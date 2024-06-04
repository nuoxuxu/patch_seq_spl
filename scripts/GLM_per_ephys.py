import pandas as pd
import anndata
import scquint.differential_splicing as ds
import sys
    
adata = anndata.read_h5ad("proc/adata_filtered.h5ad")

df_list = []
for intron_group in adata.var.intron_group.unique().to_numpy():
    df, _ = ds.run_regression(adata, intron_group, sys.argv[1], subclass=True)
    df_list.append(df)

df = pd.concat(df_list)
df.to_csv(f"proc/GLM_subclass/{sys.argv[1]}.csv")