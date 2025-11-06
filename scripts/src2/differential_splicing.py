import pandas as pd

# Note: computation-heavy imports (numpy, torch, pyro, scipy, etc.)
# have been moved inside functions/factories to avoid importing them
# at module import time.

def closure(mat):
    """
    Performs closure to ensure that all elements add up to 1.
    """
    import numpy as np

    mat = np.atleast_2d(mat)
    if np.any(mat < 0):
        raise ValueError("Cannot have negative proportions")
    if mat.ndim > 2:
        raise ValueError("Input matrix can only have two dimensions or less")
    if np.all(mat == 0, axis=1).sum() > 0:
        raise ValueError("Input matrix cannot have rows with all zeros")
    mat = mat / mat.sum(axis=1, keepdims=True)
    return mat.squeeze()

def alr(mat, denominator_idx=0):
    """
    Performs additive log ratio transformation.
    """
    # uses closure which imports numpy internally
    mat = closure(mat)
    import numpy as np
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
    import scipy.stats as stats
    lr = 2 * (llmax - llmin)
    p = stats.chi2.sf(lr, df)
    return p

def normalize(x):
    import numpy as np
    return x / x.sum(axis = 1)[: , None]

def make_MultinomialGLM(n_covariates, n_classes):
    """
    Factory that returns a MultinomialGLM class instance.
    Imports heavy dependencies locally.
    """
    import torch
    import torch.nn as nn
    from pyro.distributions import Multinomial  # local import
    dtype = torch.float64

    class MultinomialGLM(nn.Module):
        def __init__(self):
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

    return MultinomialGLM()

def make_DirichletMultinomialGLM(n_covariates, n_classes, init_A=None, init_log_alpha=None):
    """
    Factory returning a DirichletMultinomialGLM instance.
    Imports heavy dependencies locally.
    """
    import torch
    import torch.nn as nn
    import numpy as np
    from pyro.distributions import DirichletMultinomial, Gamma
    dtype = torch.float64

    if init_A is None:
        init_A = np.zeros((n_covariates, n_classes - 1))
    if init_log_alpha is None:
        init_log_alpha = np.ones(1) * 1.0

    class DirichletMultinomialGLM(nn.Module):
        def __init__(self):
            super(DirichletMultinomialGLM, self).__init__()
            self.n_covariates = n_covariates
            self.n_classes = n_classes
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

    return DirichletMultinomialGLM()

def fit_model(model_initializer, X, Y, device="cpu"):
    import torch
    import torch.optim as optim
    import numpy as np

    X = torch.tensor(X, dtype=torch.float64, device=device)
    Y = torch.tensor(Y, dtype=torch.float64, device=device)

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

#TODO: debug this function under osx-arm64, use https://lectures.scientific-python.org/advanced/debugging/index.html#debugging-segmentation-faults-using-gdb
def run_regression(ratio_matrix, ephys_data, intron_group, reduced, full, device="cpu"):
    from patsy import dmatrix
    import polars as pl
    import numpy as np
    import scipy.special as sp

    y = ratio_matrix.select(pl.selectors.starts_with(intron_group)).to_numpy()
    cells_to_keep = np.flatnonzero(y.sum(axis=1) != 0)
    y = y[cells_to_keep]

    x_reduced = np.asarray(dmatrix(reduced, ephys_data))[cells_to_keep, :]
    x_full = np.asarray(dmatrix(full, ephys_data))[cells_to_keep, :]

    n_cells, n_classes = y.shape

    pseudocounts = 10.0
    init_A_null = np.tile(alr(y.sum(axis=0) + pseudocounts, denominator_idx=-1), (x_reduced.shape[1], 1))
    model_null = lambda: make_DirichletMultinomialGLM(x_reduced.shape[1], n_classes, init_A=init_A_null)
    ll_null, model_null = fit_model(model_null, x_reduced, y, device)

    init_A = np.zeros((x_full.shape[1], n_classes - 1), dtype=np.float64)
    model = lambda: make_DirichletMultinomialGLM(x_full.shape[1], n_classes, init_A=init_A)
    ll, model = fit_model(model, x_full, y, device)
    if ll+1e-2 < ll_null:
        return pd.DataFrame(dict(intron_group=[intron_group], p_value=[None], ll_null=[None], ll=[None], n_classes=[n_classes])), pd.DataFrame()

    p_value = lrtest(ll_null, ll, n_classes - 1)
    A = model.get_full_A().cpu().detach().numpy()
    import torch  # needed for model.log_alpha tensor -> numpy
    log_alpha = model.log_alpha.cpu().detach().numpy()

    conc = np.exp(log_alpha)
    beta = A.T
    psi = normalize(conc * sp.softmax(x_full @ A))
    if np.isnan(p_value): p_value = 1.0

    df_intron_group = pd.DataFrame(dict(intron_group=[intron_group], p_value=[p_value], ll_null=[ll_null], ll=[ll], n_classes=[n_classes]))

    return df_intron_group, psi