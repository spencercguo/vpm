import copy
from typing import Any, Callable, Sequence, Tuple

import flax
import flax.linen as nn
import jax
import numpy as np
import scipy
import optax
from flax import traverse_util
from flax.core import freeze, unfreeze
from flax.training.train_state import TrainState
from jax import numpy as jnp
from jax import random


@jax.jit
def loss_jacobi(params, state, params_prev, batch, alpha, lamb=1):
    xt = batch[0, ...]
    xtp = batch[1, ...]
    q_xt_prev = state.apply_fn(params_prev, xt)
    q_xtp_prev = state.apply_fn(params_prev, xtp)
    q_xt_curr = state.apply_fn(params, xt)
    # q_xtp_curr = state.apply_fn(params, xtp)

    term1 = 0.5 * jnp.mean(q_xt_curr**2)
    term2 = jnp.mean(q_xt_curr * q_xt_prev)
    term3 = jnp.mean(q_xtp_prev * q_xt_curr)
    return term1 - (1 - alpha) * term2 - alpha * term3


@jax.jit
def loss_sigmoid(params, state, params_prev, batch, alpha, lamb=1):
    xt = batch[0, ...]
    xtp = batch[1, ...]
    q_xt_prev = nn.sigmoid(state.apply_fn(params_prev, xt))
    q_xtp_prev = nn.sigmoid(state.apply_fn(params_prev, xtp))
    q_xt_curr = state.apply_fn(params, xt)
    # q_xtp_curr = state.apply_fn(params, xtp)

    term1 = nn.softplus(q_xt_curr)
    term2 = q_xt_curr * q_xt_prev
    term3 = q_xtp_prev * q_xt_curr
    return jnp.mean(term1 - (1 - alpha) * term2 - alpha * term3)


@jax.jit
def loss_pi(params, state, params_prev, batch, alpha, lamb=1):
    xt = batch[0, ...]
    xtp = batch[1, ...]
    z_xt_prev = state.apply_fn(params_prev, xt)
    z_xt_curr = state.apply_fn(params, xt)
    z_xtp_curr = state.apply_fn(params, xtp)

    # TODO: check loss terms
    term1 = jnp.exp(z_xt_curr)
    term2 = z_xt_curr * jnp.exp(z_xt_prev - 1)
    term3 = z_xtp_curr * jnp.exp(z_xt_prev - 1)
    v = params["v"]
    norm = jnp.sum(2 * v * (jnp.mean(jnp.exp(z_xt_curr - 1)**2, axis=0) - 1) - v**2)
    return jnp.mean(term1 - (1 - alpha) * term2 - alpha * term3) + lamb * norm


def train_step(state, loss_fn, batch, params_prev, alpha, lamb=1):
    """Performs a single optimization step for a given loss function.

    Arguments
    ---------
    state : train state
    loss_fn : loss function with respect to which the optimization should be done. Must
        have the signature loss_fn(params, state, params_prev, batch, alpha)
    batch : (xt, xtp) pairs of data points sampled from p and the transition kernel
    params_prev : parameters to evaluate the model from the previous power iteration step
    alpha : damping factor
    lamb : normalization penalty

    Returns
    -------
    new_state : updated training state
    """
    # compute gradient of loss w.r.t current parameters (theta) and v
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params, state, params_prev, batch, alpha, lamb=lamb)
    new_state = state.apply_gradients(grads=grads)

    return new_state, loss


train_step_jit = jax.jit(train_step, static_argnames=("loss_fn"))


def initialize(rng, model, lr, **kwargs):
    """Initialize a given model and returns a TrainState

    Arguments
    ---------
    rng : random number seed
    model : a nn.Module
    lr : learning rate
    """
    rng, key1, key2 = random.split(rng, num=3)
    params = model.init(key1, jnp.ones(shape=(1, 2)))

    tx = optax.adam(learning_rate=lr)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return state


def train_power_iteration(
    epoch: int,
    state,
    dataset,
    loss_fn: Callable,
    params_prev,
    batch_size,
    rng,
    inner_iter: int = 100,
    print_loss_every: int = 10,
    nalpha: int = None,
):
    """Performs a single variational power iteration step. This involves
    sampling batches and then variationally minimizing the loss against
    the previous parameters. More information about the loss function's
    requirements are in the `train_step` docstring.

    Arguments
    ---------
        epoch : iteration number
        state : train state containing parameters, model apply function, and optimizers
        dataset : dataset of (xt, xtp) pairs. The first axis must be 0 or 1 corresponding to the time lag,
            the second axis must be the number of data points, and the last axis is the number of features for each datapoint
        loss_fn : loss function to optimize
        params_prev : parameters from previous power iteration step
        batch_size : batch size
        rng : random seed
        inner_iter : optional, number of inner minimization steps to perform
        print_loss_every : optional, how often to print loss during inner minimization
    """
    epoch_loss = []
    if nalpha is None:
        alpha = 1.0
    else:
        if epoch < nalpha:
            alpha = 1.0
        else:
            alpha = 1 / jnp.sqrt(epoch + 1 - nalpha)
    size = dataset.shape[1]

    for i in range(inner_iter):
        # draw batch
        rng, key = random.split(rng)
        batch_idx = random.choice(key, size, shape=(batch_size,))
        batch = dataset[:, batch_idx, :]
        # perform a training step
        state, loss = train_step_jit(state, loss_fn, batch, params_prev, alpha)
        # print loss
        epoch_loss.append(loss)
        if i % print_loss_every == 0:
            print(f"Loss: {loss:>7e} [{i:>5d} / {inner_iter:>5}")

    # update previous copy of net for next power iteration step
    params_prev = copy.deepcopy(state.params)
    # print(params_prev)

    epoch_loss = np.min(epoch_loss)
    return state, params_prev, epoch_loss


def pretrain(
    rng, state, dataset, transform_fn=lambda x: x, n_iter=100, batch_size=1024
):
    @jax.jit
    def loss_pretrain(params, x):
        out = state.apply_fn(params, x)
        out = transform_fn(out)
        # difference with x coordinate
        return jnp.mean((out - x[..., 0]) ** 2)

    size = dataset.shape[1]
    for _ in range(n_iter):
        # draw batch
        rng, key = random.split(rng)
        batch_idx = random.choice(key, size, shape=(batch_size,))
        batch = dataset[0, batch_idx, :]

        # compute gradient of loss w.r.t current parameters (theta)
        grad_fn = jax.grad(loss_pretrain)
        grads = grad_fn(state.params, batch)
        state = state.apply_gradients(grads=grads)

    return state


pretrain_jit = jax.jit(pretrain, static_argnames=("n_iter"))


def normalize(state, data, n_iter=100):
    """Pretraining for Krylov/Arnoldi iteration so that initial residual
    vector (g0 - b) has norm 1
    """
    xt = data[0, ...]

    @jax.jit
    def loss_fn(params):
        q_xt_curr = state.apply_fn({"params": params["params"]}, xt)
        v = params["v"]
        norm = 2 * v * (jnp.mean(q_xt_curr**2) - 1) - v**2
        return jnp.squeeze(norm)

    @jax.jit
    def update(state):
        grads = grad_fn(state.params)
        # reverse sign of grad v
        flat_grads = traverse_util.flatten_dict(grads, sep="/")
        flat_grads["v"] = -flat_grads["v"]
        unflat_grads = traverse_util.unflatten_dict(flat_grads, sep="/")
        unflat_grads = freeze(unflat_grads)
        state = state.apply_gradients(grads=unflat_grads)
        return state

    grad_fn = jax.grad(loss_fn)
    for i in range(n_iter):
        state = update(state)
        q_xt_curr = state.apply_fn({"params": state.params["params"]}, xt)
        print(np.mean(q_xt_curr**2))

    return state


def train(
    rng,
    state,
    dataset,
    test_input,
    loss_fn,
    transform_fn=lambda x: x,
    n_power_iter=100,
    inner_iter=200,
    batch_size=1024,
    print_loss_every=50,
    lamb=1.0,
    verbose=True,
):
    metrics = {}
    metrics["loss"] = []
    metrics["pred"] = []
    params_prev = copy.deepcopy(state.params)

    # run power iteration
    for epoch in range(n_power_iter):
        rng, new_key = random.split(rng)
        print(f"Epoch {epoch + 1} / {n_power_iter}")
        print("============================")
        state, params_prev, epoch_loss = train_power_iteration(
            epoch,
            state,
            dataset,
            loss_fn,
            params_prev,
            batch_size,
            new_key,
            inner_iter=inner_iter,
            print_loss_every=print_loss_every,
        )

        # save metrics
        print(f"Min epoch loss: {epoch_loss:>7e}")
        metrics["loss"].append(epoch_loss)

        # save intermediate prediction
        probs = transform_fn(state.apply_fn(state.params, test_input))
        metrics["pred"].append(probs)

        print("===========================")

    return state, metrics


@jax.jit
def krylov_train_step(
    state: TrainState, batch, u_prev, batch_rhs, alpha: float, lamb: float = 1
):
    """Performs a single optimization step for a given loss function.

    Arguments
    ---------
        state : train state
        batch : (xt, xtp) pairs of data points sampled from p and the transition kernel
        u_prev : forecast output (minus RHS) from the previous power iteration step
        batch_rhs :
        alpha : damping factor
        lamb : normalization penalty

    Returns
    -------
        new_state : updated training state
    """
    xt = batch[0, ...]
    # xtp = batch[1, ...]
    u_xt_prev = u_prev[0, ...]
    u_xtp_prev = u_prev[1, ...]
    # John calls this "OP" in his code
    resid = (u_xtp_prev - u_xt_prev) - batch_rhs

    @jax.jit
    def loss_fn(params):
        u_xt_curr = state.apply_fn({"params": params["params"]}, xt)
        # u_xtp_curr = state.apply_fn({"params": params["params"]}, xtp)
        v = params["v"]

        term1 = 0.5 * jnp.mean(u_xt_curr**2)
        term2 = jnp.mean(u_xt_curr * u_xt_prev)
        term3 = jnp.mean(resid * u_xt_curr)
        norm = jnp.mean(2 * v * (jnp.mean(u_xt_curr**2) - 1) - v**2)
        return term1 - (1 - alpha) * term2 - alpha * term3 + lamb * norm

    # compute gradient of loss w.r.t current parameters (theta) and v
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    # reverse sign of grad v
    flat_grads = traverse_util.flatten_dict(grads, sep="/")
    flat_grads["v"] = -flat_grads["v"]
    unflat_grads = traverse_util.unflatten_dict(flat_grads, sep="/")
    unflat_grads = freeze(unflat_grads)

    new_state = state.apply_gradients(grads=unflat_grads)
    return new_state, loss


def krylov_power_iteration(
    epoch: int,
    state: TrainState,
    dataset,
    u_prev,
    guess,
    rhs,
    basis: Sequence[Any],
    batch_size: int,
    rng,
    inner_iter: int = 100,
    print_loss_every: int = 10,
    nalpha: int = None,
):
    """
    Arguments
    ---------
        epoch : iteration number
        state : train state containing parameters, model apply function, and optimizers
        dataset : dataset of (xt, xtp) pairs. The first axis must be 0 or 1 corresponding
            to the time lag, the second axis must be the number of data points, and the
            last axis is the number of features for each datapoint
        u_prev : values for forecast on the dataset from previous power iteration step
        guess : guess function for u evaluated at all (xt, xtp) points in dataset
        rhs : integral for Feynman-Kac forecast problem evaluated at (xt) points in dataset
        basis : basis set (Krylov subspace) evaluated on dataset from previous iterations
        batch_size : batch size
        rng : random seed
        inner_iter : optional, number of inner minimization steps to perform
        print_loss_every : optional, how often to print loss during inner minimization
        nalpha : optional, after how many power iterations to turn on damping (and
            then sqrt{t + 1})

    Returns
    -------
        epoch_loss : minimum loss during power iteration step
        state : new TrainState
        u_new : updated values for the forecast on the dataset for the next power
            iteration step
        coeffs : coefficients for basis functions to get u_new
        basis : updated basis set
    """
    epoch_loss = []
    if nalpha is None:
        alpha = 1.0
    else:
        if epoch < nalpha:
            alpha = 1.0
        else:
            alpha = 1 / jnp.sqrt(epoch + 1 - nalpha)
    size = dataset.shape[1]

    for i in range(inner_iter):
        # draw batch
        rng, key = random.split(rng)
        batch_idx = random.choice(key, size, shape=(batch_size,))
        batch = dataset[:, batch_idx, :]
        u_prev_batch = u_prev[:, batch_idx, :]
        # perform a training step
        state, loss = krylov_train_step(
            state, batch, u_prev_batch, rhs[batch_idx], alpha
        )
        # print loss
        epoch_loss.append(loss)
        if i % print_loss_every == 0:
            print(f"Loss: {loss:>7e} [{i:>5d} / {inner_iter:>5}")

    # add new basis function
    basis_prev = state.apply_fn({"params": state.params["params"]}, dataset)
    basis.append(basis_prev)
    # solve DGA problem in basis
    X = np.concatenate(basis, axis=-1)
    Adga = X[0, ...].T @ (X[1, ...] - X[0, ...])
    bdga = X[0, ...].T @ (guess[0, ...] - guess[1, ...] + rhs)
    coeffs = np.linalg.solve(Adga, bdga)
    u_new = X @ coeffs + guess

    epoch_loss = np.min(epoch_loss)
    return epoch_loss, state, u_new, coeffs, basis


def subspace_initialize(rng, model, lr, input_dim, output_dim, **kwargs) -> TrainState:
    """Initialize a given model for subspace iteration and returns a TrainState

    Arguments
    ---------
        rng : random number seed
        model : a nn.Module
        lr : learning rate
        input_dim : dimension of input
        output_dim : dimension of output basis. Note that the first function is constant
            and is not learned by the network

    Returns
    -------
        state : initialized state
    """
    rng, key1, key2 = random.split(rng, num=3)
    params = model.init(key1, jnp.ones(shape=(1, input_dim)))

    # initialize matrix A and v for normalization
    flat_params = traverse_util.flatten_dict(params, sep="/")
    unfrozen_params = unfreeze(params)
    unfrozen_params["A"] = jnp.eye(output_dim)
    unfrozen_params["v"] = jnp.zeros(shape=(output_dim, 1))
    params = freeze(unfrozen_params)

    tx = optax.adam(learning_rate=lr)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return state


@jax.jit
def subspace_train_step(
    state: TrainState,
    batch: jnp.ndarray,
    batch_prev: jnp.ndarray,
    alpha: float,
    gamma: float = 1.0,
    lamb: float = 1.0,
) -> Tuple[TrainState, float]:
    """Performs a single optimization step for subspace iteration.

    Arguments
    ---------
        state : train state
        batch : (xt, xtp) pairs of data points sampled from p and the transition kernel
        batch_prev : basis at the previous power iteration step evaluated at (xt, xtp).
            shape (2, n_batch, n_basis)
        alpha : damping factor
        gamma : penalty on non-diagonal terms of V/A matrix
        lamb : penalty on normalization

    Returns
    -------
        new_state : updated training state
    """
    batch_size = batch.shape[1]
    xt = batch[0, ...]
    xtp = batch[1, ...]
    phi_xt_prev = batch_prev[0, ...]  # shape: n_batch x n_basis - 1
    phi_xtp_prev = batch_prev[1, ...]

    @jax.jit
    def loss_fn(params):
        phi_xt = state.apply_fn(
            {"params": params["params"]}, xt
        )  # shape: n_batch x n_basis - 1
        # add constant function to NN output
        Phi = jnp.concatenate(
            (jnp.ones(shape=(batch_size, 1)), phi_xt), axis=-1
        )  # shape: n_batch x n_basis

        A = params["A"]
        v = params["v"]
        phi_A = Phi @ A
        term1 = 0.5 * jnp.mean(phi_A**2)
        term2 = jnp.mean(phi_A * phi_xt_prev)
        term3 = jnp.mean(phi_A * phi_xtp_prev)
        # sum of off diagonal elements squared
        term4 = jnp.sum(jnp.where(~jnp.eye(len(A), dtype=bool), A**2, 0.0))
        # normalization term
        norm = jnp.sum(2 * v * (jnp.mean(Phi**2, axis=0) - 1) - v**2)
        return term1 - (1 - alpha) * term2 - alpha * term3 + gamma * term4 + lamb * norm

    # compute gradient of loss w.r.t current parameters (theta) and A
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    flat_grads = traverse_util.flatten_dict(grads, sep="/")
    # enforce A to be upper triangular
    flat_grads["A"] = jnp.triu(flat_grads["A"])
    # gradient ascent on normalization term
    flat_grads["v"] = -flat_grads["v"]
    unflat_grads = traverse_util.unflatten_dict(flat_grads, sep="/")
    unflat_grads = freeze(unflat_grads)

    new_state = state.apply_gradients(grads=unflat_grads)
    return new_state, loss


def subspace_iteration(
    epoch: int,
    state: TrainState,
    dataset: jnp.ndarray,
    basis_prev: jnp.ndarray,
    batch_size: int,
    rng,
    inner_iter: int = 100,
    print_loss_every: int = 10,
    nalpha: int = None,
    gamma: float = 1.0,
    lamb: float = 1.0,
) -> Tuple[float, TrainState, jnp.ndarray, jnp.ndarray]:
    """Performs a single step of subspace iteration for the eigenproblem.

    Arguments
    ---------
        epoch : iteration number
        state : train state containing parameters, model apply function, and optimizers
        dataset : dataset of (xt, xtp) pairs. The first axis must be 0 or 1 corresponding
            to the time lag, the second axis must be the number of data points, and the
            last axis is the number of features for each datapoint
        basis_prev : basis set (including constant function) from previous power iteration step.
            shape (2, n, n_basis)
        batch_size : batch size
        rng : random seed
        inner_iter : optional, number of inner minimization steps to perform
        print_loss_every : optional, how often to print loss during inner minimization
        nalpha : optional, after how many power iterations to turn on damping (and then sqrt{t + 1})

    Returns
    -------
        epoch_loss : minimum loss during inner iterations
        state : updated TrainState
        Phi_tilde : new basis (orthonormalized). shape (2, n, n_basis)
        R : orthogonalizing matrix of shape (n_basis, n_basis)
    """
    epoch_loss = []
    if nalpha is None:
        nalpha = 0
    else:
        if epoch < nalpha:
            alpha = 1.0
        else:
            alpha = 1 / jnp.sqrt(epoch + 1 - nalpha)
    size = dataset.shape[1]
    k = basis_prev.shape[-1]

    for i in range(inner_iter):
        # draw batch
        rng, key = random.split(rng)
        batch_idx = random.choice(key, size, shape=(batch_size,))
        batch = dataset[:, batch_idx, :]
        batch_prev = basis_prev[:, batch_idx, :]
        # perform a training step
        state, loss = subspace_train_step(
            state, batch, batch_prev, alpha, gamma=gamma, lamb=lamb
        )
        # print loss
        epoch_loss.append(loss)
        if i % print_loss_every == 0:
            print(f"Loss: {loss:>7e} [{i:>5d} / {inner_iter:>5}]")

    # add new basis function
    Phi = state.apply_fn({"params": state.params["params"]}, dataset)
    # include constant function for orthogonalization
    Phi = jnp.concatenate((jnp.ones(shape=(2, size, 1)), Phi), axis=-1)
    # convert to double precision
    Phi = np.array(Phi, dtype=float, copy=True)
    Phi0 = Phi[0, ...]
    #  orthogonalize using QR, but no normalization
    D = np.linalg.norm(Phi0, axis=0)
    Q, R = np.linalg.qr(Phi0)
    Phi_tilde = Phi @ (np.linalg.inv(R) @ np.diag(D))
    epoch_loss = np.min(epoch_loss)

    return epoch_loss, state, Phi_tilde, R
