"""Module 01 - Tensor engine companion example.

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

# SneppX_ALG lives in the `sneppx-alg` repo under bindings/python
_SEARCH = os.environ.get("SNEPPX_ALG_PATH")
if _SEARCH and _SEARCH not in sys.path:
    sys.path.insert(0, _SEARCH)

import numpy as np
import SneppX_ALG as sx
from SneppX_ALG.interface_bindings import nn, optim


def main():
    # 1. create tensors
    x = sx.Tensor([3.0, 4.0], requires_grad=True)
    print("x:", x.data, "shape:", x.shape)

    # 2. ops + autograd
    y = (x * x).sum()
    y.backward()
    print("d(x^T x)/dx =", np.asarray(x.grad.data), "(expected [6., 8.])")

    # 3. matmul + trace
    a = sx.Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
    b = sx.Tensor(np.array([[5.0, 6.0], [7.0, 8.0]]))
    print("trace(A@B) =", (a @ b).trace().data)

    # 4. reductions
    m = sx.Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    print("logsumexp(dim=1):", m.logsumexp(1).data)
    print("prod(dim=0):", m.prod(0).data)

    # 5. one training step
    model = nn.Linear(2, 1)
    optim_sgd = optim.SGD(model.parameters(), lr=0.01)
    optim_sgd.zero_grad()
    out = model(sx.Tensor([1.0, 2.0]))
    loss = ((out - sx.Tensor([5.0])) ** 2).mean()
    loss.backward()
    optim_sgd.step()
    print("training step ok; loss =", float(np.asarray(loss.data).ravel()[0]))


if __name__ == "__main__":
    main()