# snRNA_spatial_labeltransfer

We propose a workflow to transfer cell type labels from snRNA to low-coverage spatial Xenium data in two steps. First, snRNA and Xenium data are integrated using a conditional variational autoencoder (scVI). Second, Xenium cells are re-annotated by assigning each cell to the majority cell type among its 20 nearest neighbors from the snRNA dataset within the same Leiden cluster. The re-annotated Xenium cells can then be mapped back to their spatial context.

![SCVI+KNN (1)](https://github.com/user-attachments/assets/e2ba4278-5311-4744-a59a-3f7772d4c9b6)
