# Module 03 Quiz

1. Which backend is optimized for Linux + NVIDIA environments?
   a) Gloo
   b) NCCL
   c) MPI

   **Answer: b**

2. What does `sneppx-dist init --world-size 4` create?
   a) A running training cluster
   b) A `sneppx-dist.json` configuration with 4 processes/GPUs
   c) A Python runtime

   **Answer: b**

3. What is the central config file `sneppx-dist` manages?
   a) `cluster.yaml`
   b) `sneppx-dist.json`
   c) `torchrun.toml`

   **Answer: b**

4. In `torchrun`, what does `--nproc_per_node` control?
   a) The number of CPU cores
   b) The number of processes to launch per node
   c) The network rank

   **Answer: b**

5. Which command would TRANSITION a `created` cluster config to `running`?
   a) `sneppx-dist start`
   b) `sneppx-dist launch`
   c) `sneppx-dist teardown`

   **Answer: a**

6. How does `sneppx-dist` choose a backend by default on a non-Linux machine?
   a) Always NCCL
   b) Gloo (via `detect_backend()`)
   c) It refuses to run

   **Answer: b**

7. What does the `sneppx-dist launcher` command generate?
   a) A training loss graph
   b) A runnable Bash or PowerShell launcher script containing the torchrun command
   c) A Docker image

   **Answer: b**

8. Which command destroys the cluster state in the config?
   a) `sneppx-dist start`
   b) `sneppx-dist status`
   c) `sneppx-dist teardown`

   **Answer: c**