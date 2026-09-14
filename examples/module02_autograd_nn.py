"""Module 02 - Autograd & nn companion example.

Runs against the SneppX_ALG Python bindings of the `sneppx-alg` repo.

Usage:
    1) From a `sneppx-alg` checkout:
         $env:PYTHONPATH="bindings/python"
         python <this-file>
    2) Or point at a checkout explicitly:
         $env:SNEPPX_ALG_PATH="C:\\path\\to\\sneppx-alg\\bindings\\python"
         python <this-file>
"""

import os
import sys

_SEARCH = os.environ.get("SNEPPX_ALG_PATH")
if _SEARCH and _SEARCH not in sys.path:
    sys.path.insert(0, _SEARCH)

import numpy as np
import SneppX_ALG as sx
from SneppX_ALG.interface_bindings import nn, optim
from SneppX_ALG.interface_bindings.autograd import no_grad, set_grad_enabled, grad


# 1. autograd: graph, no_grad, gradient accumulation
def demo_autograd():
    x = sx.Tensor([2.0, 3.0], requires_grad=True)
    y = (x * x).sum()          # y = x0^2 + x1^2
    y.backward()
    print("grad:", np.asarray(x.grad.data), "(expected [4., 6.])")

    # gradient accumulation across segments (engine supports multiple
    # backward() calls before zero_grad())
    x2 = sx.Tensor([2.0, 3.0], requires_grad=True)
    seg1 = x2[0] ** 2
    seg1.backward()
    seg2 = x2[1] ** 3
    seg2.backward()
    print("acc grads:", np.asarray(x2.grad.data), "(expected [4., 27.])")
    x2.zero_grad_() if hasattr(x2, "zero_grad_") else None

    # no_grad context: ops don't build a graph
    with no_grad():
        z = x * 2.0
    print("no_grad result keeps no grad_fn:", not z.requires_grad)


# 2. nn modules: custom module + state_dict
class MLP(nn.Module):
    def __init__(self, in_d, hidden, out_d):
        super().__init__()
        self.fc1 = nn.Linear(in_d, hidden)
        self.fc2 = nn.Linear(hidden, out_d)

    def forward(self, x):
        h = self.fc1(x).relu()   # use tensor-method activation
        return self.fc2(h)


def demo_nn_modules():
    model = MLP(2, 4, 1)
    print("parameters:", sum(p.data.size for p in model.parameters()))
    x = sx.Tensor([[1.0, 2.0]], requires_grad=False)
    out = model(x)
    print("forward out shape:", out.shape)

    keys = list(model.state_dict().keys())
    print("state_dict keys:", keys)


# 3. one full training loop with a real loss + optimizer
def demo_training_loop():
    model = MLP(2, 4, 1)
    opt = optim.Adam(model.parameters(), lr=0.02)

    xs = sx.Tensor(np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]))
    ys = sx.Tensor(np.array([[0.0], [1.0], [1.0], [0.0]]))  # XOR

    for step in range(100):
        opt.zero_grad()
        pred = model(xs)
        loss = ((pred - ys) ** 2).mean()
        loss.backward()
        opt.step()

    print("final XOR loss:", float(np.asarray(loss.data).ravel()[0]))
    pred = model(xs)
    print("predictions:", np.round(np.asarray(pred.data).ravel(), 2))


# 4. higher-order gradient (create_graph) on a parameterized expression
def demo_second_order():
    w = sx.Tensor([1.0, 2.0, 3.0], requires_grad=True)
    loss = (w * w).sum()                # sum of squares
    g = grad(loss, w, create_graph=True)   # first-order, graph kept
    print("g:", np.asarray(g[0].data), "(expected [2., 4., 6.])")
    gg = grad(g[0], w, create_graph=True)  # second-order d/dw of g0
    print("d g0/dw:", np.asarray(gg[0].data), "(expected [2., 2., 2.])")


if __name__ == "__main__":
    demo_autograd()
    demo_nn_modules()
    demo_training_loop()
    demo_second_order()
    print("module02 example OK")