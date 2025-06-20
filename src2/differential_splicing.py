import anndata
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pyro.distributions import DirichletMultinomial, Gamma, Multinomial
import scanpy as sc
from scipy.stats import chi2, mannwhitneyu
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from tqdm import tqdm
from .data import make_intron_group_summation_cpu, filter_min_cells_per_feature, filter_min_cells_per_intron_group, regroup, filter_min_global_proportion
import platform
import scipy.special as sp

dtype = torch.float64
device = "cpu"

# original: from skbio.stats.composition import closure
def closure(mat):
    """
    Performs closure to ensure that all elements add up to 1.
    Parameters
    ----------
    mat : array_like
       a matrix of proportions where
       rows = compositions
       columns = components
    Returns
    -------
    array_like, np.float64
       A matrix of proportions where all of the values
       are nonzero and each composition (row) adds up to 1
    Raises
    ------
    ValueError
       Raises an error if any values are negative.
    ValueError
       Raises an error if the matrix has more than 2 dimension.
    ValueError
       Raises an error if there is a row that has all zeros.
    Examples
    --------
    >>> import numpy as np
    >>> from skbio.stats.composition import closure
    >>> X = np.array([[2, 2, 6], [4, 4, 2]])
    >>> closure(X)
    array([[ 0.2,  0.2,  0.6],
           [ 0.4,  0.4,  0.2]])
    """
    mat = np.atleast_2d(mat)
    if np.any(mat < 0):
        raise ValueError("Cannot have negative proportions")
    if mat.ndim > 2:
        raise ValueError("Input matrix can only have two dimensions or less")
    if np.all(mat == 0, axis=1).sum() > 0:
        raise ValueError("Input matrix cannot have rows with all zeros")
    mat = mat / mat.sum(axis=1, keepdims=True)
    return mat.squeeze()

# original function: from skbio.stats.composition import alr
def alr(mat, denominator_idx=0):
    r"""
    Performs additive log ratio transformation.
    This function transforms compositions from a D-part Aitchison simplex to
    a non-isometric real space of D-1 dimensions. The argument
    `denominator_col` defines the index of the column used as the common
    denominator. The :math: `alr` transformed data are amenable to multivariate
    analysis as long as statistics don't involve distances.
    :math:`alr: S^D \rightarrow \mathbb{R}^{D-1}`
    The alr transformation is defined as follows
    .. math::
        alr(x) = \left[ \ln \frac{x_1}{x_D}, \ldots,
        \ln \frac{x_{D-1}}{x_D} \right]
    where :math:`D` is the index of the part used as common denominator.
    Parameters
    ----------
    mat: numpy.ndarray
       a matrix of proportions where
       rows = compositions and
       columns = components
    denominator_idx: int
       the index of the column (2D-matrix) or position (vector) of
       `mat` which should be used as the reference composition. By default
       `denominator_idx=0` to specify the first column or position.
    Returns
    -------
    numpy.ndarray
         alr-transformed data projected in a non-isometric real space
         of D-1 dimensions for a D-parts composition
    Examples
    --------
    >>> import numpy as np
    >>> from skbio.stats.composition import alr
    >>> x = np.array([.1, .3, .4, .2])
    >>> alr(x)
    array([ 1.09861229,  1.38629436,  0.69314718])
    """
    mat = closure(mat)
    if mat.ndim == 2:
        mat_t = mat.T
        numerator_idx = list(range(0, mat_t.shape[0]))
        del numerator_idx[denominator_idx]
        lr = np.log(mat_t[numerator_idx, :]/mat_t[denominator_idx, :]).T
    elif mat.ndim == 1:
        numerator_idx = list(range(0, mat.shape[0]))
        del numerator_idx[denominator_idx]
        lr = np.log(mat[numerator_idx]/mat[denominator_idx])
    else:
        raise ValueError("mat must be either 1D or 2D")
    return lr



def lrtest(llmin, llmax, df):
    lr = 2 * (llmax - llmin)
    p = chi2.sf(lr, df)
    return p

def normalize(x):
    return x / x.sum(axis = 1)[: , None]

#TODO: debug this function under osx-arm64, use https://lectures.scientific-python.org/advanced/debugging/index.html#debugging-segmentation-faults-using-gdb
def run_regression(adata, intron_group, reduced, full, device="cpu"):
    from patsy import dmatrix
    y = adata[:, adata.var.intron_group == intron_group].X.toarray()
    cells_to_keep = np.flatnonzero(y.sum(axis=1) != 0)
    y = y[cells_to_keep]

    df = adata.obsm["predictors"]
    x_reduced = np.asarray(dmatrix(reduced, df))[cells_to_keep, :]
    x_full = np.asarray(dmatrix(full, df))[cells_to_keep, :]

    n_cells, n_classes = y.shape

    pseudocounts = 10.0
    init_A_null = np.tile(alr(y.sum(axis=0) + pseudocounts, denominator_idx=-1), (x_reduced.shape[1], 1))
    model_null = lambda: DirichletMultinomialGLM(x_reduced.shape[1], n_classes, init_A=init_A_null)
    ll_null, model_null = fit_model(model_null, x_reduced, y, device)

    init_A = np.zeros((x_full.shape[1], n_classes - 1), dtype=np.float64)
    model = lambda: DirichletMultinomialGLM(x_full.shape[1], n_classes, init_A=init_A)
    ll, model = fit_model(model, x_full, y, device)
    if ll+1e-2 < ll_null:
        return pd.DataFrame(dict(intron_group=[intron_group], p_value=[None], ll_null=[None], ll=[None], n_classes=[n_classes])), pd.DataFrame()

    p_value = lrtest(ll_null, ll, n_classes - 1)
    A = model.get_full_A().cpu().detach().numpy()
    log_alpha = model.log_alpha.cpu().detach().numpy()

    conc = np.exp(log_alpha)
    beta = A.T
    psi = normalize(conc * sp.softmax(x_full @ A))
    if np.isnan(p_value): p_value = 1.0

    df_intron_group = pd.DataFrame(dict(intron_group=[intron_group], p_value=[p_value], ll_null=[ll_null], ll=[ll], n_classes=[n_classes]))

    return df_intron_group, psi

def _run_differential_splicing(
    adata,
    cell_idx_a,
    cell_idx_b,
    device="cpu",
    min_cells_per_intron_group=30,
    min_total_cells_per_intron=30,
    n_jobs=None,
    do_regroup=False,
    min_global_proportion=1e-3,
):
    n_a = len(cell_idx_a)
    n_b = len(cell_idx_b)
    cell_idx_all = np.concatenate([cell_idx_a, cell_idx_b])
    adata = adata[cell_idx_all].copy()
    cell_idx_a = np.arange(0, n_a)
    cell_idx_b = np.arange(n_a, n_a + n_b)
    print(adata.shape)
    if min_total_cells_per_intron is not None:
        adata = filter_min_cells_per_feature(adata, min_total_cells_per_intron)
        print(adata.shape)
    if min_global_proportion is not None:
        adata = filter_min_global_proportion(adata, min_global_proportion)
        print(adata.shape)
    if do_regroup:
        adata = regroup(adata)
        print(adata.shape)
    if min_cells_per_intron_group is not None:
        adata = filter_min_cells_per_intron_group(adata, min_cells_per_intron_group, cell_idx_a)
        if adata.shape[1] == 0:
            pass
        else:
            adata = filter_min_cells_per_intron_group(adata, min_cells_per_intron_group, cell_idx_b)
        print(adata.shape)
    if adata.shape[1] == 0: return pd.DataFrame(), pd.DataFrame()

    print("Number of intron groups: ", len(adata.var.intron_group.unique()))
    print("Number of introns: ", len(adata.var))

    intron_groups = adata.var.intron_group.values
    all_intron_groups = pd.unique(intron_groups)
    intron_group_introns = defaultdict(list)
    for i, c in enumerate(intron_groups):
        intron_group_introns[c].append(i)

    X = adata.X.toarray()  # for easier parallelization using Python's libraries

    if n_jobs is not None and n_jobs != 1:
        dfs_intron_group, dfs_intron = zip(
            *Parallel(n_jobs=n_jobs)(
                delayed(run_regression)((c, X[:, intron_group_introns[c]], cell_idx_a, cell_idx_b))
                for c in tqdm(all_intron_groups)
            )
        )
    else:
        dfs_intron_group, dfs_intron = zip(*[
            run_regression((c, X[:, intron_group_introns[c]], cell_idx_a, cell_idx_b))
            for c in tqdm(all_intron_groups)
        ])
    df_intron_group = pd.concat(dfs_intron_group, ignore_index=True)
    df_intron = pd.concat(dfs_intron, ignore_index=True)
    positions = np.concatenate([intron_group_introns[c] for c in all_intron_groups])
    df_intron = pd.concat([adata.var.iloc[positions].reset_index(drop=False), df_intron], axis=1).set_index("index")
    return df_intron_group, df_intron


class MultinomialGLM(nn.Module):
    def __init__(self, n_covariates, n_classes):
        super(MultinomialGLM, self).__init__()
        self.A = nn.Parameter(torch.zeros((n_covariates, n_classes-1), dtype=dtype))
        self.register_buffer("constant_column", torch.zeros((n_covariates, 1), dtype=dtype))
        self.ll = None

    def get_full_A(self):
        return torch.cat([self.A, self.constant_column], 1)

    def forward(self, X):
        A = self.get_full_A()
        logits = X @ A
        return logits

    def loss_function(self, X, Y):
        logits = self.forward(X)
        ll = Multinomial(logits=logits).log_prob(Y).sum()
        self.ll = ll
        if torch.isnan(ll):
            print("A: ", self.A)
            print("ll: ", ll)
            raise Exception("debug")
        return -ll


class DirichletMultinomialGLM(nn.Module):
    def __init__(self, n_covariates, n_classes, init_A=None, init_log_alpha=None):
        super(DirichletMultinomialGLM, self).__init__()
        self.n_covariates = n_covariates
        self.n_classes = n_classes
        if init_A is None:
            init_A = np.zeros((n_covariates, n_classes - 1))
        if init_log_alpha is None:
            init_log_alpha = np.ones(1) * 1.0
        self.A = nn.Parameter(torch.tensor(init_A, dtype=dtype))
        self.log_alpha = nn.Parameter(torch.tensor(init_log_alpha, dtype=dtype))
        self.register_buffer("constant_column", torch.zeros((n_covariates, 1), dtype=dtype))
        self.register_buffer("conc_shape", torch.tensor(1 + 1e-4, dtype=dtype))
        self.register_buffer("conc_rate", torch.tensor(1e-4, dtype=dtype))
        self.ll = None

    def get_full_A(self):
        return torch.cat([self.A, self.constant_column], 1)

    def forward(self, X):
        alpha = torch.exp(self.log_alpha)
        A = self.get_full_A()
        P = torch.softmax(X @ A, dim=1)
        concentration = torch.mul(alpha, P)
        return A, alpha, concentration, P

    def loss_function(self, X, Y):
        A, alpha, concentration, P = self.forward(X)
        ll = DirichletMultinomial(concentration, validate_args=False).log_prob(Y).sum()
        res = (
            - ll
            - Gamma(self.conc_shape, self.conc_rate).log_prob(alpha).sum()
        )
        self.ll = ll
        return res


def fit_model(model_initializer, X, Y, device="cpu"):
    X = torch.tensor(X, dtype=dtype, device=device)
    Y = torch.tensor(Y, dtype=dtype, device=device)

    initial_lr = 1.0

    def try_optimization(lr):
        model = model_initializer()
        model.to(device)
        optimizer = optim.LBFGS(model.parameters(), lr=lr, max_iter=10000, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            loss = model.loss_function(X, Y)
            if torch.isnan(loss):
                raise ValueError("nan encountered")
            loss.backward()
            return loss

        optimizer.step(closure)
        return model.ll.cpu().detach().numpy(), model

    lr = initial_lr
    try_number = 0
    while True:
        try_number += 1
        if try_number > 10:
            print("WARNING: optimization failed, too many tries")
            return -np.inf, model_initializer()
        try:
            ll, model = try_optimization(lr)
            break
        except ValueError as ve:
            lr /= 10.0

    return ll, model