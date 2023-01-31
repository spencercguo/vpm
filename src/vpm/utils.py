from typing import Any, Sequence

import numpy as np
from jax import numpy as jnp

import extq


def split_indices(arrays):
    """Gets the indices for np.split from a
    list of arrays.

    Parameters
    ----------
    arrays : ndarray or list/tuple of ndarray
    Arrays from which to get indices

    Returns
    -------
    traj_inds : list of int
    Frame separators to use in np.split
    """
    traj_lens = [len(traj) for traj in arrays]
    traj_inds = []
    subtot = 0
    for length in traj_lens[:-1]:
        subtot += length
        traj_inds.append(subtot)
    return traj_inds


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


def rmse_logit(ref, pred, eps=np.exp(-20)):
    inD = ~((ref == 0) | (ref == 1))
    log_ref = jnp.log((ref + eps) / (1 - ref + eps))
    log_pred = jnp.log((pred + eps) / (1 - pred + eps))
    return jnp.sqrt(jnp.mean((log_ref[inD] - log_pred[inD]) ** 2))
