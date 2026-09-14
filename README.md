# SNEPPX Academy - Certification & Courses

Free course content (tensor engine, autograd, distributed training, AI
security) with a paid certification on top. Content is free forever;
certificates are an optional paid add-on.

> Status: module 01 outlined; modules 02-04 drafted.

## Modules
- Module 01 - Tensor engine with SneppX_ALG (**outlined** - lesson, runnable
  example, quiz in `docs/modules/module01.md`).
- Module 02 - Autograd & nn (draft).
- Module 03 - Distributed training (NCCL/DDP/ZeRO) (draft).
- Module 04 - AI security layers (S0-S9) (draft).

## Try module 01 today

```powershell
# from a sneppx-alg checkout
$env:PYTHONPATH="bindings/python"
python C:\path\to\sneppx-academy\examples\module01_tensor.py
```

The example creates tensors, runs autograd, matmul/trace, reductions and one
`nn.Linear` training step against the real `SneppX_ALG` bindings.

## Layout
- `docs/syllabus.md` - program outline + status legend
- `docs/modules/` - per-module learning content
- `examples/` - runnable companion scripts
- `tests/` - content-structure validation

## Roadmap
- [x] module 01: tensor engine with SneppX_ALG (outlined)
- [ ] module 01: reviewed + published
- [ ] module 02: autograd + nn
- [ ] module 03: distributed training
- [ ] module 04: AI security layers
- [ ] paid certification flow

Part of the SneppX ecosystem. Sponsor the free content: see the org
`SPONSORING.md`.