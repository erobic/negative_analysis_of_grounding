#!/usr/bin/env bash
set -e
source scripts/common.sh

python tools/create_dictionary.py --data_dir ${DATA_DIR}
python tools/compute_softscore.py --data_dir ${DATA_DIR}

#cd data
#python create_vqacp_dataset.py
#python create_vqx_dataset.py
##python create_vqx_hint.py
#python create_random_hint_for_some.py
#cd ..
