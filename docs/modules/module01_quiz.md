# Module 01 - Quiz

Answer each question, then check the answer key at the bottom. You need 7/8
to pass the module.

## Questions

1. What is the rank (number of dimensions) of a tensor created from `[[1,2],[3,4]]`?
2. True or false: broadcasting aligns dimensions from the left (outermost), like NumPy.
3. `y = (x * x).sum(); y.backward()`. What should `x.grad.data` be for `x = Tensor([3.0, 4.0])`?
4. Which op would you use to compute the sum of the main diagonal of a square matrix?
5. For `tensor([[1.,2.,3.],[4.,5.,6.]])`, what shape does `logsumexp(dim=1)` produce?
6. Give one reason the SneppX_ALG Python bindings are useful in addition to the C core.
7. What does `optimizer.zero_grad()` do before `loss.backward()` and `optimizer.step()`?
8. True or false: `a @ b` performs matrix multiplication when both operands are 2-D.

## Answer key

1. 2 (a matrix).
2. True - trailing (inner) dims align and size-1 dims broadcast.
3. `[6.0, 8.0]` since d(y)/dx = 2x.
4. `trace()`.
5. Shape `(2,)` - one value per row.
6. Any: auditable NumPy implementation, no compiler needed, dependency-light,
   CPU-only teaching/diffing surface for the C/C++ core.
7. It clears accumulated gradients from the previous step so the new
   `backward()` does not sum on top of stale grads.
8. True.