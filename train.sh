#!/bin/bash

num_pts=1
port=$((RANDOM % 64512 + 1024))
EXP_index=1

gpus=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
free_gpu_index=-1
index=0
while IFS= read -r gpu; do
    processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader --id=$(nvidia-smi --query-gpu=index --id=$gpu --format=csv,noheader))
    if [ -z "$processes" ]; then
        free_gpu_index=$index
        break
    fi
    index=$((index + 1))
done <<< "$gpus"
if [ $free_gpu_index -eq -1 ]; then
    echo "No free GPUs found."
    exit 1
else
    echo "First free GPU index: $free_gpu_index"
fi

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -n|--num_pts) num_pts="$2"; shift ;;
        -p|--port) port="$2"; shift ;;
        -e|--exp_index) EXP_index="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

rm -rf /output/${EXP_index}_lego_texturegs_${num_pts}_pts
CUDA_VISIBLE_DEVICES=$free_gpu_index python train.py \
            -s /data/XiaoqianLiang/nerf_synthetic/lego \
            -m /output/${EXP_index}_lego_texturegs_${num_pts}_pts \
            --num_pts $num_pts \
            --port $port \
            --scaling_lr 0.0 \

# #             # --checkpoint_iterations 7000 \
# #             # --start_checkpoint /output/${EXP_index}_lego_2dgs_${num_pts}_pts/chkpnt7000.pth

# CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
#             -s /data/XiaoqianLiang/mipnerf360/kitchen \
#             -r 4 \
#             -m /output/${EXP_index}_kitchen_2dgs_${num_pts}_pts \
#             --num_pts $num_pts \
#             --port $port \