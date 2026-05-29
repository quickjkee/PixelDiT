#!/bin/bash

NUM_NODES=1
NUM_GPUS=8
MASTER_ADDR=localhost
MASTER_PORT=29501
NODE_RANK=0
CONFIG_FILE="configs/pix256_xl.yaml"
CKPT_PATH=""
OUTPUT_DIR=""
RESUME=""
AUTO_RESUME=""
C2I_ROOT=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$C2I_ROOT/.." && pwd)
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --num-nodes)
            NUM_NODES="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --master-addr)
            MASTER_ADDR="$2"
            shift 2
            ;;
        --master-port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --node-rank)
            NODE_RANK="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --ckpt-path)
            CKPT_PATH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Starting distributed training with torchrun..."
if [[ "$CONFIG_FILE" != /* ]]; then
    CONFIG_FILE="$C2I_ROOT/$CONFIG_FILE"
fi

# --resume: a path ending in .ckpt resumes that exact checkpoint; any other
# (directory) path enables auto-resume from the latest checkpoint found there,
# and doubles as the output dir if --output-dir was not given.
if [[ -n "$RESUME" ]]; then
    if [[ "$RESUME" == *.ckpt ]]; then
        CKPT_PATH="$RESUME"
    else
        AUTO_RESUME=true
        [[ -z "$OUTPUT_DIR" ]] && OUTPUT_DIR="$RESUME"
    fi
fi

# --output-dir: pin the exact checkpoint/log directory (no tag suffix).
if [[ -n "$OUTPUT_DIR" ]]; then
    export PIXELDIT_OUTPUT_DIR="$OUTPUT_DIR"
fi

echo "Config: $CONFIG_FILE"
echo "Nodes: $NUM_NODES, GPUs per node: $NUM_GPUS"
echo "Master: $MASTER_ADDR:$MASTER_PORT, Node rank: $NODE_RANK"
echo "Output dir: ${OUTPUT_DIR:-<config default>}"
echo "Resume: ${RESUME:-<none>} (auto_resume=${AUTO_RESUME:-config default})"

CMD=(torchrun
    --nnodes="$NUM_NODES"
    --nproc_per_node="$NUM_GPUS"
    --master_addr="$MASTER_ADDR"
    --master_port="$MASTER_PORT"
    --node_rank="$NODE_RANK"
    "$C2I_ROOT/main.py" fit
    -c "$CONFIG_FILE"
    --trainer.num_nodes="$NUM_NODES"
    --trainer.devices="$NUM_GPUS")

if [[ -n "$CKPT_PATH" ]]; then
    CMD+=("--ckpt_path=$CKPT_PATH")
fi

if [[ -n "$AUTO_RESUME" ]]; then
    CMD+=("--auto_resume=true")
fi

CUDA_VISIBLE_DEVICES=3 "${CMD[@]}"
