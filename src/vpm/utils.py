import copy
import os
import sys
import dill as pickle
from typing import Any, Callable, Sequence, Union

import flax
import flax.linen as nn
import jax
import matplotlib as mpl
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import optax
from flax import core, struct, traverse_util
from flax.core import freeze, unfreeze
from flax.training.train_state import TrainState
from jax import numpy as jnp
from jax import random

import extq


def build_dga_krylov(model, params: Sequence[Any], data, guess, coeffs):
    """Reconstruct DGA estimate of function from Krylov vectors

    Arguments
    ---------
        model: nn.Module to apply parameters
        params: list of parameters of length n_basis
        data: points to compute DGA estimate on
        guess: guess function at points
        coeffs: coefficients for the basis functions
    """
    # assert len(params) == len(coeffs)
    basis = []
    for p in params:
        basis.append(model.apply({"params": p["params"]}, data))
    X = np.concatenate(basis, axis=-1)
    u = X @ coeffs + guess
    return u


def make_grid(num_grid, xmin, xmax, ymin, ymax):
    x = np.linspace(xmin, xmax, num_grid)
    y = np.linspace(ymin, ymax, num_grid)
    XX, YY = np.meshgrid(x, y)
    x = XX.flatten()
    y = YY.flatten()
    coordinate = np.stack((x, y))
    return coordinate.T


def _fdm_gen(potential, kT, A_fn, B_fn, res):
    x = np.linspace(-2.0, 2.0, res)
    y = np.linspace(-1.5, 2.5, res)
    X, Y = np.meshgrid(x, y)
    DGrid = np.asarray(np.concatenate([X[:, :, None], Y[:, :, None]], axis=-1))
    DGrid_Reshape = np.copy(DGrid).reshape((res ** 2, 2))
    InA_Reshape = A_fn(DGrid_Reshape)
    InB_Reshape = B_fn(DGrid_Reshape)
    InD_Reshape = ~(InA_Reshape | InB_Reshape)
    InA = A_fn(DGrid)
    InB = B_fn(DGrid)
    InD = ~(InA | InB)
    L = extq.fdm.generator_reversible_2d(
        potential(DGrid[..., 0], DGrid[..., 1]), kT, x, y
    )
    return L, DGrid, np.array(InA), np.array(InB), np.array(InD)


def fdm_q(potential, kT, A_fn, B_fn, res=100):
    L, Dgrid, InA, InB, InD = _fdm_gen(potential, kT, A_fn, B_fn, res)
    q_ref = extq.fdm.tpt.forward_committor(
        L, np.ones_like(InD.flatten()), InD.flatten(), InB.flatten().astype(float)
    )
    return np.reshape(q_ref, (res, res))


def fdm_rate(potential, kT, A_fn, B_fn, res=100):
    L, Dgrid, InA, InB, InD = _fdm_gen(potential, kT, A_fn, B_fn, res)
    qp_ref = extq.fdm.tpt.forward_committor(
        L, np.ones_like(InD.flatten()), InD.flatten(), InB.flatten().astype(float)
    )
    weights = np.exp(-potential(Dgrid[..., 0], Dgrid[..., 1] / kT))
    weights /= np.sum(weights)

    qm_ref = extq.fdm.tpt.backward_committor(
        L, weights.flatten(), InD.flatten(), InA.flatten().astype(float)
    )
    rate = extq.fdm.tpt.rate(L, qp_ref, qm_ref, weights.flatten(), rxn_coords=qp_ref)

    return rate


def rmse(ref, pred):
    return jnp.sqrt(jnp.mean((ref - pred) ** 2))


def rmse_log(ref, pred):
    inD = ~((ref == 0) | (ref == 1))
    log_ref = jnp.log(1 / (ref * (1 - ref)))
    log_pred = jnp.log(1 / (pred * (1 - pred)))
    return jnp.sqrt(jnp.mean((log_ref[inD] - log_pred[inD]) ** 2))


def plot_muller_brown(state, ax=None, num_grid=50):
    if ax is None:
        ax = plt.gca()

    test_input = make_grid(num_grid, -1.5, 1.0, -0.5, 2.0)
    qp = np.reshape(state.apply_fn(state.params, test_input), (num_grid, num_grid))

    pc = ax.pcolormesh(x, y, qp, cmap="RdYlBu_r")
    return ax, pc


def plot_threehole(state, transform_fn=lambda x: x, ax=None, num_grid=50):
    if ax is None:
        ax = plt.gca()

    test_input = make_grid(num_grid, -2.0, 2.0, -1.5, 2.5)
    qp = np.reshape(
        transform_fn(state.apply_fn(state.params, test_input)), (num_grid, num_grid)
    )

    x = np.linspace(-2.0, 2.0, num_grid)
    y = np.linspace(-1.5, 2.5, num_grid)
    pc = ax.pcolormesh(x, y, qp, cmap="RdYlBu_r")
    return ax, pc
