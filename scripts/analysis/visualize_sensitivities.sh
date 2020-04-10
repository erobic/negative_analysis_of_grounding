#!/bin/bash
set -e
source scripts/common.sh

python -u analysis/visualize_sensitivities.py \
--data_dir ${DATA_DIR} \
--save_dir ${SAVE_DIR}

