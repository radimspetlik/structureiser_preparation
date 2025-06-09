#!/bin/bash

source "./config.sh"

cd "${SCRIPT_DIR}" || exit

if [ -z "${SLURM_JOB_ID}" ]; then
  SLURM_JOB_ID="0000000"
fi

if [ $# -eq 0 ]; then
  SCRIPT_NAME="structureiser.py"
  CONFIG_FILE="confs/lynx.yml"
  ARGS="${SCRIPT_NAME} ${CONFIG_FILE}"
else
  ARGS="${@}"
fi

if [ -z "${SLURM_JOB_ID}" ]; then
  NNODES=1
  GPUS_PER_NODE=1
else
  NNODES=${SLURM_NNODES}
  GPUS_PER_NODE=$(scontrol show job ${SLURM_JOB_ID} | grep -oP 'gres/gpu:\K[^ ]+')
fi

export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3

# get node list and set MASTER_NODE/master_address
declare -A node_global_rank
node_list=$(scontrol show hostnames "${SLURM_NODELIST}")
index=0
for node in ${node_list[@]}; do
    node_global_rank["${node}"]=${index}
    index=$((index+1))
done

echo "node_list: ${node_list[@]}"

MASTER_NODE="$(scontrol show hostnames "${SLURM_NODELIST}" | head -1)"

for node in ${node_list[@]}; do
  singularity exec --nv --cleanenv \
    -B"${SCRATCH}:${SCRATCH}" \
    -B"${PROJECT_DIR}:${PROJECT_DIR}" \
    "${PROJECT_DIR}/singularity/${SINGULARITY_IMAGE_NAME}.sif" \
    torchrun \
      --nnodes=${NNODES} \
      --nproc-per-node=${GPUS_PER_NODE} \
      --rdzv-id=${SLURM_JOB_ID} \
      --rdzv-backend=c10d \
      --rdzv-endpoint=${MASTER_NODE}:29529 \
      ${ARGS}
done
