#!/bin/bash

source "config.sh"

singularity build "${PROJECT_DIR}/singularity/structureiser.sif" "singularity.def"