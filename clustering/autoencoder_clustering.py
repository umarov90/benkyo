# pip install 'scanpy[leiden]'
# sudo apt-get install python3-tk
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import seaborn as sns
import umap
plt.rcParams.update({'font.size': 25})
# matplotlib.use('Agg')

# # Preprocessing and clustering 3k PBMCs

# In May 2017, this started out as a demonstration that Scanpy would allow to reproduce most of Seurat's [guided clustering tutorial](http://satijalab.org/seurat/pbmc3k_tutorial.html) ([Satija et al., 2015](https://doi.org/10.1038/nbt.3192)).
#
# We gratefully acknowledge Seurat's authors for the tutorial! In the meanwhile, we have added and removed a few pieces.
#
# The data consist of *3k PBMCs from a Healthy Donor* and are freely available from 10x Genomics ([here](http://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz) from this [webpage](https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.1.0/pbmc3k)). On a unix system, you can uncomment and run the following to download and unpack the data. The last line creates a directory for writing processed data.

# In[1]:


# !mkdir data
# !wget http://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz -O data/pbmc3k_filtered_gene_bc_matrices.tar.gz
# !cd data; tar -xzf pbmc3k_filtered_gene_bc_matrices.tar.gz
# !mkdir write


# <div class="alert alert-info">
#
# **Note**
#
# Download the notebook by clicking on the _Edit on GitHub_ button. On GitHub, you can download using the _Raw_ button via right-click and _Save Link As_. Alternatively, download the whole [scanpy-tutorial](https://github.com/theislab/scanpy-tutorials) repository.
#
# </div>

# <div class="alert alert-info">
#
# **Note**
#
# In Jupyter notebooks and lab, you can see the documentation for a python function by hitting ``SHIFT + TAB``. Hit it twice to expand the view.
#
# </div>

# In[2]:


import numpy as np
import pandas as pd
import scanpy as sc


# In[3]:


sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor='white')


# In[4]:


results_file = '/home/user/PycharmProjects/scanpy_test/data/write/pbmc3k.h5ad'  # the file that will store the analysis results


# Read in the count matrix into an [AnnData](https://anndata.readthedocs.io/en/latest/anndata.AnnData.html) object, which holds many slots for annotations and different representations of the data. It also comes with its own HDF5-based file format: `.h5ad`.

# In[5]:


adata = sc.read_10x_mtx(
    'data/filtered_gene_bc_matrices/hg19/',  # the directory with the `.mtx` file
    var_names='gene_symbols',                # use gene symbols for the variable names (variables-axis index)
    cache=True)                              # write a cache file for faster subsequent reading


# <div class="alert alert-info">
#
# **Note**
#
# See [anndata-tutorials/getting-started](https://anndata-tutorials.readthedocs.io/en/latest/getting-started.html) for a more comprehensive introduction to `AnnData`.
#


# ## Preprocessing

# Show those genes that yield the highest fraction of counts in each single cell, across all cells.

# In[8]:


# sc.pl.highest_expr_genes(adata, n_top=20, )


# Basic filtering:

# In[9]:


sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)


# Let's assemble some information about mitochondrial genes, which are important for quality control.
#
# Citing from "Simple Single Cell" workflows [(Lun, McCarthy & Marioni, 2017)](https://master.bioconductor.org/packages/release/workflows/html/simpleSingleCell.html#examining-gene-level-metrics):
#
# > High proportions are indicative of poor-quality cells (Islam et al. 2014; Ilicic et al. 2016), possibly because of loss of cytoplasmic RNA from perforated cells. The reasoning is that mitochondria are larger than individual transcript molecules and less likely to escape through tears in the cell membrane.

# With `pp.calculate_qc_metrics`, we can compute many metrics very efficiently.

# In[10]:


adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)


# A violin plot of some of the computed quality measures:
#
# * the number of genes expressed in the count matrix
# * the total counts per cell
# * the percentage of counts in mitochondrial genes

# In[11]:


sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
              jitter=0.4, multi_panel=True)


# Remove cells that have too many mitochondrial genes expressed or too many total counts:

# In[12]:


# sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
# sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')


# Actually do the filtering by slicing the `AnnData` object.

# In[13]:


adata = adata[adata.obs.n_genes_by_counts < 2500, :]
adata = adata[adata.obs.pct_counts_mt < 5, :]

# Total-count normalize (library-size correct) the data matrix $\mathbf{X}$ to 10,000 reads per cell, so that counts become comparable among cells.

# In[14]:


sc.pp.normalize_total(adata, target_sum=1e4)


# Logarithmize the data:

# In[15]:


sc.pp.log1p(adata)

DL_data = adata.X.A

# Identify highly-variable genes.

# In[16]:


sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)


# In[17]:


# sc.pl.highly_variable_genes(adata)


# Set the `.raw` attribute of the AnnData object to the normalized and logarithmized raw gene expression for later use in differential testing and visualizations of gene expression. This simply freezes the state of the AnnData object.

# <div class="alert alert-info">
#
# **Note**
#
# You can get back an `AnnData` of the object in `.raw` by calling `.raw.to_adata()`.
#
# </div>

# In[18]:


adata.raw = adata


# <div class="alert alert-info">
#
# **Note**
#
# If you don't proceed below with correcting the data with `sc.pp.regress_out` and scaling it via `sc.pp.scale`,
# you can also get away without using `.raw` at all.
#
# The result of the previous highly-variable-genes detection is stored as an annotation in `.var.highly_variable` and
# auto-detected by PCA and hence, `sc.pp.neighbors` and subsequent manifold/graph tools. In that case,
# the step *actually do the filtering* below is unnecessary, too.
#
# </div>

# Actually do the filtering

# In[19]:


adata = adata[:, adata.var.highly_variable]


# Regress out effects of total counts per cell and the percentage of mitochondrial genes expressed. Scale the data to
# unit variance.

# In[20]:


sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])


# Scale each gene to unit variance. Clip values exceeding standard deviation 10.

# In[21]:


sc.pp.scale(adata, max_value=10)


# ## Principal component analysis

# Reduce the dimensionality of the data by running principal component analysis (PCA), which reveals the main axes of
# variation and denoises the data.

# In[22]:


sc.tl.pca(adata, svd_solver='arpack')
print(adata)

# We can make a scatter plot in the PCA coordinates, but we will not use that later on.

# In[23]:


# sc.pl.pca(adata, color='CST3')


# Let us inspect the contribution of single PCs to the total variance in the data. This gives us information about how many PCs we should consider in order to compute the neighborhood relations of cells, e.g. used in the clustering function  `sc.tl.louvain()` or tSNE `sc.tl.tsne()`. In our experience, often a rough estimate of the number of PCs does fine.

# In[24]:


# sc.pl.pca_variance_ratio(adata, log=True)


# Save the result.

# In[25]:


adata.write(results_file)


# In[26]:


adata


# ## Computing the neighborhood graph

# Let us compute the neighborhood graph of cells using the PCA representation of the data matrix. You might simply use default values here. For the sake of reproducing Seurat's results, let's take the following values.

# In[27]:





# ## Embedding the neighborhood graph

# We suggest embedding the graph in two dimensions using UMAP ([McInnes et al., 2018](https://arxiv.org/abs/1802.03426)), see below. It is potentially more faithful to the global connectivity of the manifold than tSNE, i.e., it better preserves trajectories. In some ocassions, you might still observe disconnected clusters and similar connectivity violations. They can usually be remedied by running:
#
# ```
# tl.paga(adata)
# pl.paga(adata, plot=False)  # remove `plot=False` if you want to see the coarse-grained graph
# tl.umap(adata, init_pos='paga')
# ```

# In[28]:




# In[29]:

# sc.pl.umap(adata, color=['CST3', 'NKG7', 'PPBP'])


# As we set the `.raw` attribute of `adata`, the previous plots showed the "raw" (normalized, logarithmized, but uncorrected) gene expression. You can also plot the scaled and corrected gene expression by explicitly stating that you don't want to use `.raw`.

# In[30]:


# sc.pl.umap(adata, color=['CST3', 'NKG7', 'PPBP'], use_raw=False)


# ## Clustering the neighborhood graph

# As with Seurat and many other frameworks, we recommend the Leiden graph-clustering method (community detection based on optimizing modularity) by [Traag *et al.* (2018)](https://scanpy.readthedocs.io/en/latest/references.html#traag18). Note that Leiden clustering directly clusters the neighborhood graph of cells, which we already computed in the previous section.

# In[31]:





# Plot the clusters, which agree quite well with the result of Seurat.

# In[32]:

sc.pp.neighbors(adata, n_neighbors=10, n_pcs=50)
sc.tl.umap(adata)
sc.tl.leiden(adata)
sc.pl.umap(adata, color=['leiden', 'CST3', 'NKG7'])


def build(input_size, latent_dim):
    layer_units = [1024, 512]
    inputs = Input(shape=input_size)
    x = inputs
    x = Dropout(0.2)(x)
    for i, f in enumerate(layer_units):
        x = Dense(f, activation="relu")(x)
        if i == 0:
            x = Dropout(0.2)(x)

    latent = Dense(latent_dim, use_bias=False)(x)
    encoder = Model(inputs, latent, name="encoder")
    latent_inputs = Input(shape=(latent_dim,))
    x = latent_inputs
    for i, f in enumerate(layer_units[::-1]):
        x = Dense(f, activation="relu")(x)
        if i == 0:
            x = Dropout(0.2)(x)

    outputs = Dense(input_size)(x)
    decoder = Model(latent_inputs, outputs, name="decoder")
    autoencoder = Model(inputs, decoder(encoder(inputs)), name="autoencoder")
    return autoencoder


autoencoder = build(len(DL_data[0]), 50)
# autoencoder = tf.keras.models.load_model("autoencoder.h5")
autoencoder.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(lr=1e-4))
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
autoencoder.fit(DL_data, DL_data, epochs=1000, batch_size=16, validation_split=0.1, callbacks=[callback])
autoencoder.save("autoencoder_a.h5")
encoder = autoencoder.get_layer("encoder")
latent_vectors = encoder.predict(np.asarray(DL_data))
np.savetxt("latent_vectors.csv", latent_vectors, delimiter=",", fmt="%s")
reducer = umap.UMAP()
latent_umap = reducer.fit_transform(latent_vectors)
fig, axs = plt.subplots(1,1,figsize=(8,4))
print("Plotting")
sns.scatterplot(x=latent_umap[:, 0], y=latent_umap[:, 1], s=5, alpha=0.2, ax=axs)
axs.set_title("Latent space")
axs.set_xlabel("A1")
axs.set_ylabel("A2")
plt.tight_layout()
plt.savefig("clustering.png")

adata.obsm["X_pca"] = latent_vectors
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=50)
sc.tl.umap(adata)
sc.tl.leiden(adata)
sc.pl.umap(adata, color=['leiden', 'CST3', 'NKG7'])


# Save the result.

# In[33]:


adata.write(results_file)


# ## Finding marker genes

# Let us compute a ranking for the highly differential genes in each cluster. For this, by default, the `.raw` attribute of AnnData is used in case it has been initialized before. The simplest and fastest method to do so is the t-test.

# In[34]:


# sc.tl.rank_genes_groups(adata, 'leiden', method='t-test')
# sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)


# In[35]:


sc.settings.verbosity = 2  # reduce the verbosity


# The result of a [Wilcoxon rank-sum (Mann-Whitney-U)](https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test) test is very similar. We recommend using the latter in publications, see e.g., [Sonison & Robinson (2018)](https://doi.org/10.1038/nmeth.4612). You might also consider much more powerful differential testing packages like MAST, limma, DESeq2 and, for python, the recent diffxpy.

# In[36]:


sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, fontsize=18)


# Save the result.

# In[37]:


adata.write(results_file)


# As an alternative, let us rank genes using logistic regression. For instance, this has been suggested by [Natranos et al. (2018)](https://doi.org/10.1101/258566). The essential difference is that here, we use a multi-variate appraoch whereas conventional differential tests are uni-variate. [Clark et al. (2014)](https://doi.org/10.1186/1471-2105-15-79) has more details.

# In[38]:


# sc.tl.rank_genes_groups(adata, 'leiden', method='logreg')
# sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)


# Reload the object that has been save with the Wilcoxon Rank-Sum test result.

# In[40]:


adata = sc.read(results_file)
adata.uns['log1p']['base'] = None


# Show the 10 top ranked genes per cluster 0, 1, ..., 7 in a dataframe.

# In[41]:


pd.DataFrame(adata.uns['rank_genes_groups']['names']).head(5)


# Get a table with the scores and groups.

# In[42]:


result = adata.uns['rank_genes_groups']
groups = result['names'].dtype.names
pd.DataFrame(
    {group + '_' + key[:1]: result[key][group]
    for group in groups for key in ['names', 'pvals']}).head(5)


# Compare to a single cluster:

# In[43]:


sc.tl.rank_genes_groups(adata, 'leiden', groups=['0'], reference='1', method='wilcoxon')
sc.pl.rank_genes_groups(adata, groups=['0'], n_genes=20)


# If we want a more detailed view for a certain group, use `sc.pl.rank_genes_groups_violin`.

# In[44]:


sc.pl.rank_genes_groups_violin(adata, groups='0', n_genes=8)


# Reload the object with the computed differential expression (i.e. DE via a comparison with the rest of the groups):

# In[45]:


adata = sc.read(results_file)
adata.uns['log1p']['base'] = None

# In[46]:


sc.pl.rank_genes_groups_violin(adata, groups='0', n_genes=8)


# If you want to compare a certain gene across groups, use the following.

# In[47]:


sc.pl.violin(adata, ['CST3', 'NKG7', 'PPBP'], groupby='leiden')


# Actually mark the cell types.

# In[48]:
# new_cluster_names = [
#     'CD4 T', 'CD14 Monocytes',
#     'B', 'CD8 T',
#     'NK', 'FCGR3A Monocytes',
#     'Dendritic', 'Megakaryocytes', "temp"]

new_cluster_names = [
    'CD4 T', 'RPS',
    'cd14+ monocytes', 'B',
    'CD8 T', 'NK2',
    'f monocyte', 'monocyte 2', "t cells", "dendritic", "megakaryocytes"]
adata.rename_categories('leiden', new_cluster_names)


# In[49]:


sc.pl.umap(adata, color='leiden', legend_loc='on data', title='')

