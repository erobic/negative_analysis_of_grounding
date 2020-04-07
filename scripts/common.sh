#!/bin/bash

ROOT_DIR=/hdd/robik
PROJ_DIR=${ROOT_DIR}/projects/negative_analysis_of_grounding
# nag = negative analysis of grounding
DATA_DIR=${ROOT_DIR}/nag/data
SAVE_DIR=${ROOT_DIR}/nag/saved
ENV_NAME=negative_analysis_of_grounding

mkdir -p ${DATA_DIR}
mkdir -p ${SAVE_DIR}
cd ${PROJ_DIR}
export PYTHONPATH=${PROJ_DIR}

source activate ${ENV_NAME}