from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np
import random

from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from .base import VariationalMixin
from .utils import EPS, prod, inverse_softplus, vec_to_chol
from ....distributions import MatrixNormal


__all__ = [
    "InducingMixin"
]


def _jittered_cholesky(m):
    j = EPS * m.detach().diagonal().mean() * torch.eye(m.shape[-1], device=m.device)
    return m.add(j).cholesky()

class InducingMixin(VariationalMixin):
    def __init__(self, *args, inducing_rows=None, inducing_cols=None, prior_sd=1., init_lamda=1e-4, learn_lamda=True,
                 max_lamda=None, q_inducing="diagonal", whitened_u=True, max_sd_u=None, sqrt_width_scaling=False,
                 cache_cholesky=False, ensemble_size=8, key_layers, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_bias = self.bias is not None
        self.whitened_u = whitened_u
        self._weight_shape = self.weight.shape
        self._u = None
        self._caching_mode = False
        self.reset_cache()

        # Delete original weights
        del self.weight
        if self.has_bias:
            del self.bias

        # Set dimensions
        self._d_out = self._weight_shape[0]
        self._d_in = prod(self._weight_shape[1:]) + int(self.has_bias)
        self.inducing_rows = inducing_rows or int(self._d_out ** 0.5)
        self.inducing_cols = inducing_cols or int(self._d_in ** 0.5)

        # Set prior parameters
        self.prior_sd = prior_sd / (self._d_in ** 0.5) if sqrt_width_scaling else prior_sd
        self.max_sd_u = max_sd_u
        self.max_lamda = max_lamda
        self._lamda = nn.Parameter(torch.tensor(inverse_softplus(init_lamda))) if learn_lamda else \
                     torch.tensor(inverse_softplus(init_lamda))

        # Set posterior parameters
        self.inducing_mean = nn.Parameter(torch.randn(self.inducing_rows, self.inducing_cols))
        self._inducing_sd = nn.Parameter(torch.full((self.inducing_rows, self.inducing_cols), inverse_softplus(1e-3)))

        # Set manifold transformation
        self.row_flow = MaskedAffineAutoregressiveTransform(
            features=self.inducing_cols, hidden_features=16, dropout_probability=0.1)
        for layer in self.row_flow.autoregressive_net.modules():
            if isinstance(layer, nn.Linear):
                torch.nn.utils.spectral_norm(layer)

        # Set noise control parameters
        self.alpha_row = self.alpha_col = 0.01
        self.z_row = nn.Parameter(torch.randn(self.inducing_rows, self._d_out) * self.inducing_rows ** -0.5)
        self.z_col = nn.Parameter(torch.randn(self.inducing_cols, self._d_in) * self.inducing_cols ** -0.5)
        self.z_row_rho = nn.Parameter(torch.full((self.inducing_rows,), inverse_softplus(0.001)))
        self.z_col_rho = nn.Parameter(torch.full((self.inducing_cols,), inverse_softplus(0.001)))
        self._d_row = nn.Parameter(torch.full((self.inducing_rows,), inverse_softplus(1e-3)))
        self._d_col = nn.Parameter(torch.full((self.inducing_cols,), inverse_softplus(1e-3)))

        # Set random seed
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        np.random.seed(42)
        random.seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Initialize weights
        self.weight, self.bias = self.sample_shaped_parameters()
        self._caching_mode = cache_cholesky

    def reset_cache(self):
        self._L_r_cached = None
        self._L_c_cached = None
        self._row_transform_cached = None
        self._col_transform_cached = None
        self._u_cache = None
        self._u_middle = None

    def forward(self, x):
        self.weight, self.bias = self.sample_shaped_parameters()
        return super().forward(x)

    def sample_shaped_parameters(self):
        u = self._u if self._u is not None else self.sample_u()
        self._u = None
        row_chol = self.prior_inducing_row_scale_tril()
        col_chol = self.prior_inducing_col_scale_tril()
        if self.whitened_u:
            u = row_chol @ u @ col_chol.t()
        self._u_middle = u

        # p = u.numel()
        # if p > 2:
          
        #     norm_sq = u.pow(2).sum()
        #     # Prevent division by zero
        #     shrinkage = torch.relu(1 - (p - 2) / (norm_sq + EPS))
        #     u = shrinkage * u
        mean, noise = self.conditional_mean_and_noise(u, row_chol, col_chol)
        parameters = self.prior_sd * (mean + self.lamda() * noise)
        if self.has_bias:
            w, b = parameters[:, :-1], parameters[:, -1]
        else:
            w, b = parameters, None
        return w.view(self._weight_shape), b

    def sample_u(self):
        # Original sampling
        u = self.inducing_dist().rsample()
        # James-Stein Shrinkage: shrink towards zero to reduce variance
        # Only meaningful when dimension > 2
        
        u1, logdet_row = self.row_flow(u)
        self._u_cache = {"u0":u, "u1":u1, "logdet_row": logdet_row} 
        result = u1.view(self.inducing_rows, self.inducing_cols)
        return result

    def conditional_mean_and_noise(self, u, row_chol, col_chol):
        row_transform = self.compute_row_transform(row_chol)
        col_transform = self.compute_col_transform(col_chol)
        M_w = row_transform @ u @ col_transform

        # Matheron rule sampling
        e1 = torch.randn(self._d_out, self._d_in, device=u.device)
        e2, e3, e4 = (torch.randn(self.inducing_rows, self.inducing_cols, device=u.device) for _ in range(3))
        
        if self._L_r_cached is None:
            L_r = _jittered_cholesky(self.z_row.mm(self.z_row.t()))
            if self.caching_mode():
                self._L_r_cached = L_r
        else:
            L_r = self._L_r_cached

        if self._L_c_cached is None:
            L_c = _jittered_cholesky(self.z_col.mm(self.z_col.t()))
            if self.caching_mode():
                self._L_c_cached = L_c
        else:
            L_c = self._L_c_cached

        t1 = self.z_row @ e1 @ self.z_col.t()
        t2 = L_r @ e2 @ self.d_col()
        t3 = self.d_row() @ e3 @ L_c.t()
        t4 = self.d_row() @ e4 @ self.d_col()
        u_bar = t1 + t2 + t3 + t4

        return M_w, e1 - row_transform @ u_bar @ col_transform

    def compute_row_transform(self, row_chol):
        if self._row_transform_cached is None:
            T = self.z_row.cholesky_solve(row_chol).t()
            if self.caching_mode(): self._row_transform_cached = T
            return T
        return self._row_transform_cached

    def compute_col_transform(self, col_chol):
        if self._col_transform_cached is None:
            T = self.z_col.cholesky_solve(col_chol)
            if self.caching_mode(): self._col_transform_cached = T
            return T
        return self._col_transform_cached

    def kl_divergence(self):
        device = self.d_col().device
        if self.inducing_dist() is None:
            base_kl = torch.tensor(0., device=device)
        else:
            cache = self._u_cache
            u0  = cache["u0"]         # before flow
            u   = cache["u1"]         # after flow
            ldj = cache["logdet_row"] # log|det dT/du0|

            q0 = self.inducing_dist()
            p  = self.inducing_prior_dist()

            log_q0 = q0.log_prob(u0).sum()
            log_pu = p.log_prob(u).sum()

            base_kl = log_q0 - ldj.sum() - log_pu
        return base_kl + self.conditional_kl_divergence()

    def conditional_kl_divergence(self):
        return self._d_in * self._d_out * (0.5 * self.lamda() ** 2 - self.lamda().log() - 0.5)

    def caching_mode(self):
        return self._caching_mode

    def set_caching_mode(self, mode):
        self._caching_mode = mode
        if not mode: self.reset_cache()

    def d_row(self):
        return F.softplus(self._d_row).diag_embed()

    def d_col(self):
        return F.softplus(self._d_col).diag_embed()

    def get_z_row(self):
        eps = torch.rand_like(self.z_row)
        return self.z_row + self.alpha_row * eps * self.get_z_row_rho()

    def get_z_col(self):
        eps = torch.rand_like(self.z_col)
        return self.z_col + self.alpha_col * eps * self.get_z_col_rho()

    def get_z_row_rho(self):
        return F.softplus(self.z_row_rho).view(-1, 1)

    def get_z_col_rho(self):
        return F.softplus(self.z_col_rho).view(-1, 1)

    def lamda(self):
        return F.softplus(self._lamda).clamp(0., self.max_lamda) if self._lamda is not None else None

    def prior_inducing_row_cov(self):
        z_row = self.get_z_row()
        return z_row @ z_row.t() + self.d_row().pow(2)

    def prior_inducing_row_scale_tril(self):
        if self._L_r_cached is None:
            L = _jittered_cholesky(self.prior_inducing_row_cov())
            if self.caching_mode(): self._L_r_cached = L
            return L
        return self._L_r_cached

    def prior_inducing_col_cov(self):
        z_col = self.get_z_col()
        return z_col @ z_col.t() + self.d_col().pow(2)

    def prior_inducing_col_scale_tril(self):
        if self._L_c_cached is None:
            L = _jittered_cholesky(self.prior_inducing_col_cov())
            if self.caching_mode(): self._L_c_cached = L
            return L
        return self._L_c_cached

    def inducing_sd(self):
        return F.softplus(self._inducing_sd).clamp(0., self.max_sd_u) if self._inducing_sd is not None else None

    def inducing_prior_dist(self):
        loc = self.z_row.new_zeros(self.inducing_rows, self.inducing_cols)
        if self.whitened_u:
            scale = self.z_row.new_ones(self.inducing_rows, self.inducing_cols)
            return dist.Normal(loc, scale)
        return MatrixNormal(loc, self.prior_inducing_row_scale_tril(), self.prior_inducing_col_scale_tril())

    def inducing_dist(self):
        return dist.Normal(self.inducing_mean, self.inducing_sd())


