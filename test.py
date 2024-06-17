import anndata
from src.extended_anndata import *
from typing import Literal
import src.differential_splicing as ds
import torch
import torch.optim as optim

from importlib import reload

adata = anndata.read_h5ad("results/preprocessed_adata_three.h5ad")
adata = ExtendedAnnData(adata)
adata = adata.add_predictors()

# def run_regression_wrapper(adata:ExtendedAnnData, predictor: str, intron_group: str,
#                    model: Literal['simple', 'multiple'], 
#                    device: Literal['cpu', 'gpu']='cpu') -> tuple:
    
#     if predictor == "cpm":
#         predictor = intron_group.split("_")[0]
#     if model == "simple":
#         reduced = "1"
#         full = f"{predictor}"
#     if model == "multiple":
#         reduced = "subclass"
#         full = f"subclass + {predictor}"
#     df, psi = run_regression(adata, intron_group, reduced, full)
#     return df, psi

# run_regression_wrapper(adata, 'v_baseline', 'Kdm5d_Y_898549_+', 'simple')

intron_group='Kdm5d_Y_898549_+'
reduced = '1'
full = '1 + v_baseline'
device="cpu"

#### Testing ds.run_regression ####
from patsy import dmatrix
y = adata[:, adata.var.intron_group == intron_group].X.toarray()
cells_to_keep = np.flatnonzero(y.sum(axis=1) != 0)
y = y[cells_to_keep]

df = adata.obsm["predictors"]
x_reduced = np.asarray(dmatrix(reduced, df))[cells_to_keep, :]
x_full = np.asarray(dmatrix(full, df))[cells_to_keep, :]

n_cells, n_classes = y.shape

pseudocounts = 10.0
init_A_null = np.tile(ds.alr(y.sum(axis=0) + pseudocounts, denominator_idx=-1), (x_reduced.shape[1], 1))
model_null = lambda: ds.DirichletMultinomialGLM(x_reduced.shape[1], n_classes, init_A=init_A_null)

X = torch.tensor(x_reduced, dtype=torch.float64, device=device)
Y = torch.tensor(y, dtype=torch.float64, device=device)
lr = 1

model = model_null()
model.to(device)
optimizer = optim.LBFGS(model.parameters(), lr=lr, max_iter=10000, line_search_fn="strong_wolfe")
optimizer.zero_grad()

alpha = torch.exp(model.log_alpha)
A = model.get_full_A()
P = torch.softmax(X @ A, dim=1)