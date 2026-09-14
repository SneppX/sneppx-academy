# Module 02 - Autograd & nn

Status: **outlined**. Estimated: 3 hours.

## 1. How autograd works

Every operation on a `Tensor` with `requires_grad=True` records the input tensors and the backward function in a directed acyclic graph (the *autograd graph*). Calling `.backward()` on a scalar loss walks this graph in reverse order, computing gradients into the `.grad` attribute of every leaf tensor.

```python
from SneppX_ALG.interface_bindings import autograd

x = sx.Tensor([2.0, 3.0], requires_grad=True)
y = x * x        # Mul records (x, x)
z = y.sum()       # Sum records (y)
z.backward()      # reverse traversal: dz/dy = 1, dy/dx = 2*x
print(x.grad)     # [4., 6.]
```

### Key concepts

| Concept | Meaning |
|---------|---------|
| `.grad` | Accumulated gradient (only on leaf tensors) |
| `.requires_grad` | Flag; set by user or by operations on grad-tracked tensors |
| `.backward()` | Reverse-mode differentiation; `grad_output` scales the starting gradient |
| `grad()` | Functional API: return gradient tensors directly without writing to `.grad` |
| `no_grad` context | Disables graph recording inside the block |
| `set_grad_enabled(flag)` | Toggle globally |

### Functional gradient: `grad()`

```python
from SneppX_ALG.interface_bindings.autograd import grad

x = sx.Tensor([1.0, 2.0, 3.0], requires_grad=True)
loss = (x * x).sum()
# grad(loss, x) returns a list of tensors, one per input
g = grad(loss, x)
print(g[0].data)  # [2., 4., 6.]
```

### Gradient accumulation

Without `zero_grad()`, multiple `.backward()` calls add into `.grad`:

```python
x = sx.Tensor([1.0, 2.0], requires_grad=True)
(x[0] ** 2).backward()
(x[1] ** 3).backward()
# x.grad == [2, 12]  (2*x0 + 3*x1^2 evaluated at [1,2])
```

Always call `optimizer.zero_grad()` at the start of each training iteration.

## 2. `nn.Module` and custom networks

`sneppx-alg`'s `nn.Module` stores parameters and buffers, and provides `state_dict()` / `load_state_dict()` for serialization.

```python
from SneppX_ALG.interface_bindings import nn

class MLP(nn.Module):
    def __init__(self, in_d, hidden, out_d):
        super().__init__()
        self.fc1 = nn.Linear(in_d, hidden)
        self.fc2 = nn.Linear(hidden, out_d)

    def forward(self, x):
        return self.fc2(self.fc1(x).relu())

model = MLP(2, 4, 1)
print(list(model.state_dict().keys()))
# ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias']
```

`ModuleList` holds a sequence of modules; `register_buffer()` adds non-trainable persistent state (e.g. running statistics).

## 3. Losses and optimizers

```python
from SneppX_ALG.interface_bindings import optim

model = MLP(2, 4, 1)
opt = optim.Adam(model.parameters(), lr=0.02)

for step in range(100):
    opt.zero_grad()
    pred = model(xs)
    loss = ((pred - ys) ** 2).mean()   # MSE loss
    loss.backward()
    opt.step()
```

Built-in losses: `MSELoss`, `CrossEntropyLoss`, `NLLLoss`, `BCEWithLogitsLoss`, `HuberLoss`, `SmoothL1Loss`, `KLDivLoss`, and 14 more (see `nn.Loss`).

Built-in optimizers: `SGD`, `Adam`, `AdamW`, `AdaGrad`, `RMSProp`, `LBFGS`, `SparseAdam`, and more.

## 4. Higher-order gradients (create_graph)

To compute Hessians or Jacobians, pass `create_graph=True` to `.backward()` or `grad()`:

```python
from SneppX_ALG.interface_bindings.autograd import grad

w = sx.Tensor([1.0, 2.0, 3.0], requires_grad=True)
loss = (w * w).sum()
g = grad(loss, w, create_graph=True)     # g = 2w
gg = grad(g[0], w, create_graph=True)    # d/dw (2w) = 2
print(gg[0].data)                         # [2., 2., 2.]
```

This enables exact Jacobian-vector products without finite differences, useful for second-order optimizers and meta-learning.

## 5. Gradient checkpointing

`sneppx-alg` provides `gradient_checkpointing.checkpoint()` to trade compute for memory:

```python
from SneppX_ALG.interface_bindings.gradient_checkpointing import checkpoint

def expensive_block(x):
    return x.linear(W1).relu().linear(W2).relu()

# Forward stores minimal state; backward recomputes intermediate activations
out = checkpoint(expensive_block, x)
```

Useful for very deep models where forward activations don't fit in GPU memory.

## 6. Exercises

1. Write a 3-layer MLP (784 → 256 → 128 → 10) and print the total parameter count using `sum(p.data.size for p in model.parameters())`.
2. Use `grad()` to compute the gradient of `logsumexp(x)` w.r.t. `x` at `x = [1.0, 2.0, 3.0]`; verify it matches the analytical softmax.
3. Implement a manual SGD step (`w -= lr * g`) and compare the result after 10 steps to `optim.SGD` with the same `lr` on a simple linear regression.
4. Use `create_graph=True` to compute the Hessian of `x^T A x` for a 3×3 matrix `A` and verify the diagonal matches `2 * diag(A)`.
5. Explain why `zero_grad()` is required at the start of every training iteration.

## 7. Check your understanding

Take the quiz in `module02_quiz.md`.

## Resources
- `examples/module02_autograd_nn.py` - runnable companion example.
- `docs/modules/module01.md` - Tensor Engine (prerequisite).
- `sneppx-alg` source: `bindings/python/SneppX_ALG/interface_bindings/autograd.py`, `nn.py`, `optim.py`, `gradient_checkpointing.py`.
- This module is free content; certification is a paid add-on (see `docs/certification.md`).