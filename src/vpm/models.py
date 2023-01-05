from typing import Any, Callable, Sequence

import flax.linen as nn
from jax import numpy as jnp


class CommittorNet(nn.Module):
    stateA: Callable[Any, bool]
    stateB: Callable[Any, bool]
    layer_widths: Sequence[int]
    act: Callable

    @nn.compact
    def __call__(self, x):
        g = nn.Dense(features=self.layer_widths[0])(x)
        g = self.act(g)
        for w in self.layer_widths[1:]:
            g = nn.Dense(features=w)(g)
            g = self.act(g)
        g = nn.Dense(features=1)(g)
        g = nn.sigmoid(g)
        inA = self.stateA(x).astype(float)
        inB = self.stateB(x).astype(float)
        return g * (1 - inA - inB) + inB


class CommittorNetSigmoid(nn.Module):
    stateA: Callable[Any, bool]
    stateB: Callable[Any, bool]
    layer_widths: Sequence[int]
    act: Callable
    ab_value: float = 50

    @nn.compact
    def __call__(self, x):
        g = nn.Dense(features=self.layer_widths[0])(x)
        g = self.act(g)
        for w in self.layer_widths[1:]:
            g = nn.Dense(features=w)(g)
            g = self.act(g)
        g = nn.Dense(features=1)(g)
        # set A and B to be large positive/negative values
        # so that after sigmoid function they go to 0 and 1
        g = jnp.where(self.stateA(x), -self.ab_value, g)
        g = jnp.where(self.stateB(x), self.ab_value, g)
        return g


class MFPTNet(nn.Module):
    stateB: Callable[Any, bool]
    layer_widths: Sequence[int]
    act: Callable

    @nn.compact
    def __call__(self, x):
        g = nn.Dense(features=self.layer_widths[0])(x)
        g = self.act(g)
        for w in self.layer_widths[1:]:
            g = nn.Dense(features=w)(g)
            g = self.act(g)
        g = nn.Dense(features=1)(g)
        inB = self.stateB(x).astype(float)
        # don't include guess (0 in B)
        return g * (1 - inB)


class MLP(nn.Module):
    """Simple MLP with fully-connected layers and nonlinear
    activations in between"""

    output_dim: int
    layer_widths: Sequence[int]
    act: Callable

    @nn.compact
    def __call__(self, x):
        for w in self.layer_widths:
            x = nn.Dense(features=w)(x)
        x = self.act(x)
        x = nn.Dense(features=self.output_dim)(x)
        return x
