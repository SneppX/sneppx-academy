# Module 03: Distributed Training

This module covers the principles of distributed AI training using NCCL/Gloo backends, and how to use the `sneppx-dist` CLI to manage training clusters.

## Key Concepts
- **Data Parallelism**: Dividing data across GPUs.
- **Backends**: NCCL (optimal on GPU/Linux), Gloo (general purpose).
- **Cluster Lifecycle**: Initialize (config) -> Start (validate) -> Launch (run training) -> Teardown.

## Using sneppx-dist
The `sneppx-dist` CLI manages the `sneppx-dist.json` configuration file, which torchrun uses to coordinate multi-node training:

```sh
sneppx-dist init --world-size 2 --backend nccl
sneppx-dist start
sneppx-dist launch train.py --epochs 10
```

See the `examples/module03_distributed.py` file for a simulation of cluster creation and launcher script rendering.

After the lesson, complete the [Module 03 Quiz](module03_quiz.md).
