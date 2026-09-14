# Module 02 Quiz - Autograd & nn

**Pass condition: 5/8 correct**

---

**Q1.** What attribute stores the accumulated gradient on a leaf tensor after `.backward()`?

- A) `.derivative`
- B) `.grad`
- C) `.backward_grad`
- D) `.gradient`

---

**Q2.** What does `no_grad` do?

- A) Deletes all recorded gradients.
- B) Disables gradient computation globally.
- C) Disables graph recording inside its block, so ops are treated as if `requires_grad=False`.
- D) Raises an error if a tensor requires grad.

---

**Q3.** The `grad()` function returns:

- A) A single tensor with the same shape as the input.
- B) A list of tensors, one per element in `inputs`.
- C) A scalar.
- D) A tuple `(gradient, hessian)`.

---

**Q4.** What is the purpose of `create_graph=True` in `.backward()`?

- A) It creates a separate graph for each backward pass.
- B) It keeps the computational graph alive so gradients can be differentiated again (enabling higher-order derivatives).
- C) It creates a DAG visualization of the model.
- D) It doubles the memory usage for no benefit.

---

**Q5.** Why is `optimizer.zero_grad()` called at the start of each training step?

- A) To reset `.grad` to zero so gradients don't accumulate across iterations.
- B) To free GPU memory.
- C) To detach the model from the autograd graph.
- D) It is optional and has no effect.

---

**Q6.** Given `x = sx.Tensor([2.0], requires_grad=True)` and `y = x ** 3`, what is `x.grad` after `y.backward()`?

- A) `[2.0]`
- B) `[3.0]`
- C) `[12.0]`
- D) `[8.0]`

---

**Q7.** Which is true about `Module.state_dict()`?

- A) It returns a dict of tensor values (copied data).
- B) It returns a dict of tensor references (live data).
- C) It only includes parameters, never buffers.
- D) It is only available on `nn.Linear`.

---

**Q8.** What does gradient checkpointing trade off?

- A) Accuracy for speed.
- B) Memory for compute (forward is re-executed during backward).
- C) GPU for CPU.
- D) Nothing; it is always strictly better.

---

### Answer Key

1. **B** — `.grad`
2. **C** — disables graph recording inside the block
3. **B** — a list, one per input
4. **B** — enables higher-order derivatives
5. **A** — prevents cross-iteration accumulation
6. **C** — derivative of x^3 is 3x^2 = 3 * 4 = 12
7. **A** — copies tensor data into a plain dict
8. **B** — recomputes activations during backward to save memory