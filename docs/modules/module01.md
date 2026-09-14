# Module 01 - Tensor Engine with SneppX_ALG

Status: **outlined** (learning content draft). Estimated: 2 hours.

## 1. What is a tensor?

A tensor is a multi-dimensional array - a generalization of scalars (0-D),
vectors (1-D), matrices (2-D) and higher-rank blocks (N-D). Deep learning is
almost entirely tensor math: model weights are tensors, inputs are tensors,
and the forward pass is a sequence of tensor operations.

SneppX_ALG (`SneppX_ALG` Python package) exposes a NumPy-based tensor engine
that is a faithful, auditable stand-in for the C/C++ core `_Tensor` engine.
It provides dtype handling, shape/broadcast semantics, autograd records, and
the `nn` module - all in dependency-light Python.

## 2. Creating tensors

```python
import numpy as np
import SneppX_ALG as sx

x = sx.Tensor([1.0, 2.0, 3.0])                     # 1-D tensor
m = sx.Tensor([[1.0, 2.0], [3.0, 4.0]])            # 2-D matrix
z = sx.Tensor(np.zeros((2, 3)))                    # via NumPy
```

Key properties: `.shape`, `.dtype`, `.ndim`. Ops are chained and lazy-free:
results are new tensors.

> Hands-on: run `examples/module01_tensor.py` against a local `sneppx-alg`
> checkout (see the example header for how to point Python at the bindings).

## 3. Core operations

- Arithmetic: `+ - * /`, `@` (matmul), `square()`, `reciprocal()`, `remainder()`, `clamp()`.
- Shape: `reshape()`, `transpose()`, `unsqueeze()`, `expand()`.
- Reductions: `sum()`, `mean()`, `prod(dim)`, `logsumexp(dim)`, `amin()`, `amax()`, `trace()`, `diagonal()`.
- Indexing: `x[i]`, advanced slicing, `gather`-style ops.

Broadcasting follows NumPy rules: trailing dimensions align, size-1 dims
broadcast.

## 4. Autograd in five lines

```python
x = sx.Tensor([3.0, 4.0], requires_grad=True)
y = (x * x).sum()          # y = x0^2 + x1^2
y.backward()               # reverse-mode gradient accumulation
print(x.grad.data)         # [6. 8.]
```

Everything in module 01 uses first-order gradients; module 02 covers the
autograd graph, `nn` modules and higher-order (`create_graph`) derivatives.

## 5. A minimal trainable layer

```python
from SneppX_ALG.interface_bindings import nn, optim

model = nn.Linear(2, 1)
optimizer = optim.SGD(model.parameters(), lr=0.01)
for step in range(20):
    optimizer.zero_grad()
    loss = ((model(sx.Tensor([1.0, 2.0])) - sx.Tensor([5.0])) ** 2).mean()
    loss.backward()
    optimizer.step()
```

The training loop shape is identical to PyTorch hybrids: forward, loss,
backward, update.

## 6. Exercises

1. Create a 3x4 tensor and compute `logsumexp` along dim 1 - print the shape you get.
2. Build a 2-layer MLP with `sx.nn.Linear` + `relu` and run one forward pass on a 5x2 batch.
3. Verify `d(x^T x)/dx = 2x` numerically with `.backward()` and a difference check.
4. Compute the trace of a product `A@B` two ways: `trace()` and element-wise dot of `A` with `B^T`.

## 7. Check your understanding

Take the quiz in `module01_quiz.md`. Pass it to unlock module 02.

## Resources
- `examples/module01_tensor.py` - runnable companion (see its header).
- `sneppx-alg` bindings under `bindings/python/SneppX_ALG/interface_bindings`.
- This module is free content; certification is a paid add-on (see the repo README).