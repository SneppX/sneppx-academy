"""Module 03 - Distributed training companion example.

Runs against the `sneppx-dist` cluster/launcher API.

Usage:
    1) From a `sneppx-dist` checkout:
         $env:PYTHONPATH="C:\\path\\to\\sneppx-dist\\src"
         python <this-file>
    2) Or point at a checkout explicitly:
         $env:SNEPPX_DIST_PATH="C:\\path\\to\\sneppx-dist\\src"
         python <this-file>
"""

import os
import pathlib
import sys
import tempfile

_SEARCH = os.environ.get("SNEPPX_DIST_PATH")
if _SEARCH and _SEARCH not in sys.path:
    sys.path.insert(0, _SEARCH)

from sneppx_dist.cluster import Cluster, detect_backend


def demo_cluster_lifecycle():
    tmp = pathlib.Path(tempfile.mkdtemp())
    cluster = Cluster(config=str(tmp / "cluster.json"))

    # 1. init: pick backend by platform (nccl on Linux, gloo elsewhere)
    info = cluster.init(world_size=2, backend=None, master_addr="127.0.0.1")
    print("backend:", info["backend"], "detected:", detect_backend())
    print("world_size:", info["world_size"], "state:", info["state"])

    # 2. launch command (declarative, no execution)
    cmd = cluster.launch_command("train.py", ["--epochs", "10"])
    print("command:", " ".join(cmd))

    # 3. render a runnable launcher script
    launcher = cluster.render_launcher(tmp / "train.sh", "train.py")
    print("launcher:", launcher.name)
    print(launcher.read_text(encoding="utf-8"))

    # 4. lifecycle transitions
    cluster.start()
    print("after start:", cluster.status()["state"])
    cluster.teardown()
    print("after teardown:", cluster.status()["state"])
    print("cluster lifecycle OK")


if __name__ == "__main__":
    demo_cluster_lifecycle()