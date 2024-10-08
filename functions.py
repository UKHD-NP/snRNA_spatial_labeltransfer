    # Python packages
import scanpy as sc
import anndata as ad
import scvi
#import bbknn
import scib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.neighbors import NearestNeighbors, KDTree
from matplotlib.colors import ListedColormap
import typing as tp
import squidpy as sq
import seaborn as sns
import csv
import gzip
import os
import scipy.io
import cosg as cos
import bbknn
from scib_metrics.benchmark import Benchmarker


def reannotation_KNN(adata, k=20):
    if np.all(adata.obs.groupby(['Method','leiden_scvi']).size())<=0:
        print('Error: There are clusters with no cell. Adjust clustering')
    
    else:
        import warnings

        # Unterdrücke nur FutureWarnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            # check k Single-nuclei nearest neigbor

            tree = KDTree(adata.obsm['X_scVI'], leaf_size=100)

            ind_sc_list = [] # index of nearest neighbour
            sub_sn = adata[(adata.obs['Method'] == 'xenium') ]

            for i in range(sub_sn.n_obs): # No. of Xenium cells selected

                cluster = sub_sn.obs['leiden_scvi'][(i)] # Cluster of current Xenium cell
                dist, ind = tree.query(sub_sn.obsm['X_scVI'][i,:].reshape(1, -1), k=sub_sn.n_obs) # Query and rank all cells by neighbor distance
                ind = ind.flatten()
                cells_keep = (adata.obs['Method'].iloc[ind] == 'singelCell') & (adata.obs['leiden_scvi'].iloc[ind] == cluster) #  All Single-nuclei cells in the correct cluster
                ind_sc = ind[cells_keep][:k] # Select top k nearest Single-nuclei neighbors
                ind_sc_list.append(ind_sc)

            ind_sc_matrix = np.stack(ind_sc_list, axis=0)
            ind_sc_df = pd.DataFrame(ind_sc_matrix,
                index=sub_sn.obs_names)

			# extracting annotation from sn neighbor
            adata_annotated=adata.copy()

            for row in ind_sc_df.index:
                cell_counts=adata_annotated.obs['annot_v1'].iloc[ind_sc_df.loc[row]].value_counts() # celltypes of the cell sn neighbor
                majority_cell_type = cell_counts.idxmax() # majority celltypes of sn neigbor
                adata_annotated.obs.loc[row, 'annot_v1'] = majority_cell_type
                adata_annotated.obs['annot_v1'] = adata_annotated.obs['annot_v1'].cat.remove_unused_categories()
            return adata_annotated
        
        
        
def flatten(l: tp.List[tp.Any]) -> tp.List[tp.Any]:
    return [item for sublist in l for item in sublist]

def _get_values_to_plot(
    adata,
    values_to_plot,
    gene_names,
    groups=None,
    key="rank_genes_groups",
    gene_symbols=None,
):
    """
    If rank_genes_groups has been called, this function
    prepares a dataframe containing scores, pvalues, logfoldchange etc to be plotted
    as dotplot or matrixplot.
    The dataframe index are the given groups and the columns are the gene_names
    used by rank_genes_groups_dotplot
    Parameters
    ----------
    adata
    values_to_plot
        name of the value to plot
    gene_names
        gene names
    groups
        groupby categories
    key
        adata.uns key where the rank_genes_groups is stored.
        By default 'rank_genes_groups'
    gene_symbols
        Key for field in .var that stores gene symbols.
    Returns
    -------
    pandas DataFrame index=groups, columns=gene_names
    """
    valid_options = [
        "scores",
        "logfoldchanges",
        "pvals",
        "pvals_adj",
        "log10_pvals",
        "log10_pvals_adj",
    ]
    if values_to_plot not in valid_options:
        raise ValueError(
            f"given value_to_plot: '{values_to_plot}' is not valid. Valid options are {valid_options}"
        )

    values_df = None
    check_done = False
    if groups is None:
        groups = adata.uns[key]["names"].dtype.names
    if values_to_plot is not None:
        df_list = []
        for group in groups:
            df = rank_genes_groups_df(adata, group, key=key, gene_symbols=gene_symbols)
            if gene_symbols is not None:
                df["names"] = df[gene_symbols]
            # check that all genes are present in the df as sc.tl.rank_genes_groups
            # can be called with only top genes
            if not check_done:
                if df.shape[0] < adata.shape[1]:
                    message = (
                        "Please run `sc.tl.rank_genes_groups` with "
                        "'n_genes=adata.shape[1]' to save all gene "
                        f"scores. Currently, only {df.shape[0]} "
                        "are found"
                    )
                    logg.error(message)
                    raise ValueError(message)
            df["group"] = group
            df_list.append(df)

        values_df = pd.concat(df_list)
        if values_to_plot.startswith("log10"):
            column = values_to_plot.replace("log10_", "")
        else:
            column = values_to_plot
        values_df = pd.pivot(
            values_df, index="names", columns="group", values=column
        ).fillna(1)

        if values_to_plot in ["log10_pvals", "log10_pvals_adj"]:
            values_df = -1 * np.log10(values_df)

        values_df = values_df.loc[gene_names].T

    return values_df


def rank_genes_groups_df(
    adata,
    group=None,
    *,
    key,
    pval_cutoff=None,
    log2fc_min=None,
    log2fc_max=None,
    gene_symbols=None,
) -> pd.DataFrame:
    """\
    :func:`scanpy.tl.rank_genes_groups` results in the form of a
    :class:`~pandas.DataFrame`.
    Params
    ------
    adata
        Object to get results from.
    group
        Which group (as in :func:`scanpy.tl.rank_genes_groups`'s `groupby`
        argument) to return results from. Can be a list. All groups are
        returned if groups is `None`.
    key
        Key differential expression groups were stored under.
    pval_cutoff
        Return only adjusted p-values below the  cutoff.
    log2fc_min
        Minimum logfc to return.
    log2fc_max
        Maximum logfc to return.
    gene_symbols
        Column name in `.var` DataFrame that stores gene symbols. Specifying
        this will add that column to the returned dataframe.
    Example
    -------
    >>> import scanpy as sc
    >>> pbmc = sc.datasets.pbmc68k_reduced()
    >>> sc.tl.rank_genes_groups(pbmc, groupby="louvain", use_raw=True)
    >>> dedf = sc.get.rank_genes_groups_df(pbmc, group="0")
    """
    if isinstance(group, str):
        group = [group]
    if group is None:
        group = list(adata.uns[key]["names"].dtype.names)
    colnames = ["names", "scores"]

    d = [pd.DataFrame(adata.uns[key][c])[group] for c in colnames]
    d = pd.concat(d, axis=1, names=[None, "group"], keys=colnames)
    d = d.stack(level=1).reset_index()
    d["group"] = pd.Categorical(d["group"], categories=group)
    d = d.sort_values(["group", "level_0"]).drop(columns="level_0")

    return d.reset_index(drop=True)


def _rank_genes_groups_plot(
    adata,
    plot_type="heatmap",
    groups=None,
    n_genes=None,
    groupby=None,
    values_to_plot=None,
    var_names=None,
    min_logfoldchange=None,
    key=None,
    show=None,
    save=None,
    return_fig=False,
    gene_symbols=None,
    **kwds,
):
    """\
    Common function to call the different rank_genes_groups_* plots
    """
    if var_names is not None and n_genes is not None:
        raise ValueError(
            "The arguments n_genes and var_names are mutually exclusive. Please "
            "select only one."
        )

    if var_names is None and n_genes is None:
        # set n_genes = 10 as default when none of the options is given
        n_genes = 10

    if key is None:
        key = "rank_genes_groups"

    if groupby is None:
        groupby = str(adata.uns[key]["params"]["groupby"])
    group_names = adata.uns[key]["names"].dtype.names if groups is None else groups

    if var_names is not None:
        if isinstance(var_names, Mapping):
            # get a single list of all gene names in the dictionary
            var_names_list = sum([list(x) for x in var_names.values()], [])
        elif isinstance(var_names, str):
            var_names_list = [var_names]
        else:
            var_names_list = var_names
    else:
        # dict in which each group is the key and the n_genes are the values
        var_names = {}
        var_names_list = []
        for group in group_names:
            df = rank_genes_groups_df(
                adata,
                group,
                key=key,
                gene_symbols=gene_symbols,
                log2fc_min=min_logfoldchange,
            )

            if gene_symbols is not None:
                df["names"] = df[gene_symbols]

            genes_list = df.names[df.names.notnull()].tolist()

            if len(genes_list) == 0:
                logg.warning(f"No genes found for group {group}")
                continue
            if n_genes < 0:
                genes_list = genes_list[n_genes:]
            else:
                genes_list = genes_list[:n_genes]
            var_names[group] = genes_list
            var_names_list.extend(genes_list)

    # by default add dendrogram to plots
    kwds.setdefault("dendrogram", True)

    if plot_type in ["dotplot", "matrixplot"]:
        # these two types of plots can also
        # show score, logfoldchange and pvalues, in general any value from rank
        # genes groups
        title = None
        values_df = None
        if values_to_plot is not None:
            values_df = _get_values_to_plot(
                adata,
                values_to_plot,
                var_names_list,
                key=key,
                gene_symbols=gene_symbols,
            )
            title = values_to_plot
            if values_to_plot == "logfoldchanges":
                title = "log fold change"
            else:
                title = values_to_plot.replace("_", " ").replace("pvals", "p-value")

        if plot_type == "dotplot":
            from scanpy.pl import dotplot

            _pl = dotplot(
                adata,
                var_names,
                groupby,
                dot_color_df=values_df,
                return_fig=True,
                gene_symbols=gene_symbols,
                **kwds,
            )
            if title is not None and "colorbar_title" not in kwds:
                _pl.legend(colorbar_title=title)
        _pl.make_figure()
        if show:
            plt.show()


def rank_genes_groups_dotplot(
    adata,
    groups=None,
    n_genes=None,
    groupby=None,
    values_to_plot=None,
    var_names=None,
    gene_symbols=None,
    min_logfoldchange=None,
    key=None,
    show=None,
    save=None,
    return_fig=False,
    **kwds,
):
    """\
    Plot ranking of genes using dotplot plot (see :func:`~scanpy.pl.dotplot`)
    Parameters
    ----------
    {params}
    {vals_to_plot}
    {show_save_ax}
    return_fig
        Returns :class:`DotPlot` object. Useful for fine-tuning
        the plot. Takes precedence over `show=False`.
    **kwds
        Are passed to :func:`~scanpy.pl.dotplot`.
    Returns
    -------
    If `return_fig` is `True`, returns a :class:`DotPlot` object,
    else if `show` is false, return axes dict
    Examples
    --------
    .. plot::
        :context: close-figs
        import scanpy as sc
        adata = sc.datasets.pbmc68k_reduced()
        sc.tl.rank_genes_groups(adata, 'bulk_labels', n_genes=adata.raw.shape[1])
    Plot top 2 genes per group.
    .. plot::
        :context: close-figs
        sc.pl.rank_genes_groups_dotplot(adata,n_genes=2)
    Plot with scaled expressions for easier identification of differences.
    .. plot::
        :context: close-figs
        sc.pl.rank_genes_groups_dotplot(adata, n_genes=2, standard_scale='var')
    Plot `logfoldchanges` instead of gene expression. In this case a diverging colormap
    like `bwr` or `seismic` works better. To center the colormap in zero, the minimum
    and maximum values to plot are set to -4 and 4 respectively.
    Also, only genes with a log fold change of 3 or more are shown.
    .. plot::
        :context: close-figs
        sc.pl.rank_genes_groups_dotplot(
            adata,
            n_genes=4,
            values_to_plot="logfoldchanges", cmap='bwr',
            vmin=-4,
            vmax=4,
            min_logfoldchange=3,
            colorbar_title='log fold change'
        )
    Also, the last genes can be plotted. This can be useful to identify genes
    that are lowly expressed in a group. For this `n_genes=-4` is used
    .. plot::
        :context: close-figs
        sc.pl.rank_genes_groups_dotplot(
            adata,
            n_genes=-4,
            values_to_plot="logfoldchanges",
            cmap='bwr',
            vmin=-4,
            vmax=4,
            min_logfoldchange=3,
            colorbar_title='log fold change',
        )
    A list specific genes can be given to check their log fold change. If a
    dictionary, the dictionary keys will be added as labels in the plot.
    .. plot::
        :context: close-figs
        var_names = {{'T-cell': ['CD3D', 'CD3E', 'IL32'],
                      'B-cell': ['CD79A', 'CD79B', 'MS4A1'],
                      'myeloid': ['CST3', 'LYZ'] }}
        sc.pl.rank_genes_groups_dotplot(
            adata,
            var_names=var_names,
            values_to_plot="logfoldchanges",
            cmap='bwr',
            vmin=-4,
            vmax=4,
            min_logfoldchange=3,
            colorbar_title='log fold change',
        )
    .. currentmodule:: scanpy
    See also
    --------
    tl.rank_genes_groups
    """
    return _rank_genes_groups_plot(
        adata,
        plot_type="dotplot",
        groups=groups,
        n_genes=n_genes,
        groupby=groupby,
        values_to_plot=values_to_plot,
        var_names=var_names,
        gene_symbols=gene_symbols,
        key=key,
        min_logfoldchange=min_logfoldchange,
        show=show,
        save=save,
        return_fig=return_fig,
        **kwds,
    )


def ranked_genes_stats(adata, group, d=None, key=None):
    """\
    A function to calculate basics stats for selected markers.
    Params
    ------
    adata
        Object to calculate stats from.
    group
        Which group (as in :func:`scanpy.tl.rank_genes_groups`'s `groupby`
        argument) to return results from. Is needed to calculate expression
        values in different cell clusters.
    key
        Key differential expression groups were stored under.
    d
        A melted DataFrame with columns 'group', 'names', 'scores' (optional).
        If not provided, marker genes are found in adata under the `key`.
    Returns
    -------
    A DataFrame object with columns  'group', 'names', 'scores' (optional),
    'pct_nz', 'mean', 'mean_nz'.
    """
    if d is None and not key:
        print(
            "Please provide a dataframe with markers or a key that was used to store them in adata."
        )
        return
    elif d is None and key:
        d = rank_genes_groups_df(adata, key=key)

    d["pct_nz"] = 0
    d["mean"] = 0
    d["mean_nz"] = 0

    for g in d.group.unique():
        genes = d[d["group"] == g]["names"].values
        x = adata[adata.obs[group] == g].copy()

        d.loc[d["group"] == g, "pct_nz"] = (
            (x[:, genes].X > 0).sum(axis=0) / x[:, genes].X.shape[0]
        ).A1
        d.loc[d["group"] == g, "mean"] = x[:, genes].X.mean(axis=0).A1
        d.loc[d["group"] == g, "mean_nz"] = np.mean(
            np.asarray((x[:, genes].X).todense()),
            axis=0,
            where=np.asarray((x[:, genes].X > 0).todense()),
        )

    return d