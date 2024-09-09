#!/bin/bash

GPU_ID=0
num_pts=1
port=$((RANDOM % 64512 + 1024))

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -g|--gpu_id) GPU_ID="$2"; shift ;;
        -n|--num_pts) num_pts="$2"; shift ;;
        -p|--port) port="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
            -s /data/XiaoqianLiang/nerf_synthetic/lego \
            -m /output/lego_mlp_${num_pts}_pts_globle_a \
            --num_pts $num_pts \
            --port $port \
            --position_lr_init 0.0 \
