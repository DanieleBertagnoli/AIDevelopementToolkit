#!/usr/bin/env bash
set -euo pipefail

# Single GPU, single node
# Launch with one process on one GPU.
torchrun --nproc_per_node=1 examples/mnist_training.py

# Multiple GPUs, single node
# Launch with one process per GPU on the current node.
torchrun --nproc_per_node=4 examples/mnist_training.py

# Multiple GPUs, multiple nodes
# Use the same rendezvous id and endpoint on all nodes.
# Replace NODE0_HOST with the address of the rendezvous host.
# Replace <NUM_NODES>, <NPROC_PER_NODE>, and <NODE_RANK> for your cluster.

# Node 0
# torchrun \
#   --nnodes=2 \
#   --nproc_per_node=4 \
#   --node_rank=0 \
#   --rdzv_backend=c10d \
#   --rdzv_endpoint=NODE0_HOST:29500 \
#   --rdzv_id=my_run \
#   examples/mnist_training.py

# Node 1
# torchrun \
#   --nnodes=2 \
#   --nproc_per_node=4 \
#   --node_rank=1 \
#   --rdzv_backend=c10d \
#   --rdzv_endpoint=NODE0_HOST:29500 \
#   --rdzv_id=my_run \
#   examples/mnist_training.py
