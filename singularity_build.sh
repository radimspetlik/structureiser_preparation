#!/bin/bash

PROJECT_DIR="/mnt/data/vrg/spetlrad/data"

singularity build "${PROJECT_DIR}/singularity/structureiser.sif" "singularity.def"