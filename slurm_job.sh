#!/bin/bash

cd "${HOME}/structureiser" || exit

# if env variable SLURM_JOB_ID is not defined, then define it
if [ -z "${SLURM_JOB_ID}" ]; then
  SLURM_JOB_ID="0000000"
fi

# if number of arguments is zero, echo "zero"
if [ $# -eq 0 ]; then
  SCRIPT_NAME="structureiser.py"
  CONFIG_FILE="confs/lynx.yml"
else
  SCRIPT_NAME="${2}"
  CONFIG_FILE="${1}"
fi

SINGULARITY_IMAGE_NAME="structureiser"

PROJECT_DIR="/mnt/data/vrg/spetlrad/data"
SCRATCH="/data/temporary/"

echo "PROJECT_DIR: ${PROJECT_DIR}"

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
      "${SCRIPT_NAME}" ${SLURM_JOB_ID} "${CONFIG_FILE}"
done
