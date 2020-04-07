#!/usr/bin/env bash
set -e
source scripts/common.sh

## Script for downloading data

# GloVe Vectors
wget -P ${DATA_DIR} http://nlp.stanford.edu/data/glove.6B.zip --no-check-certificate
unzip ${DATA_DIR}/glove.6B.zip -d ${DATA_DIR}/glove
rm ${DATA_DIR}/glove.6B.zip

# Questions
wget -P ${DATA_DIR} https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip
unzip ${DATA_DIR}/v2_Questions_Train_mscoco.zip -d ${DATA_DIR}
rm ${DATA_DIR}/v2_Questions_Train_mscoco.zip

wget -P ${DATA_DIR} https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
unzip ${DATA_DIR}/v2_Questions_Val_mscoco.zip -d ${DATA_DIR}
rm ${DATA_DIR}/v2_Questions_Val_mscoco.zip

wget -P ${DATA_DIR} https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip
unzip ${DATA_DIR}/v2_Questions_Test_mscoco.zip -d ${DATA_DIR}
rm ${DATA_DIR}/v2_Questions_Test_mscoco.zip

# Annotations
wget -P ${DATA_DIR} https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip
unzip ${DATA_DIR}/v2_Annotations_Train_mscoco.zip -d ${DATA_DIR}
rm ${DATA_DIR}/v2_Annotations_Train_mscoco.zip

wget -P ${DATA_DIR} https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip
unzip ${DATA_DIR}/v2_Annotations_Val_mscoco.zip -d ${DATA_DIR}
rm ${DATA_DIR}/v2_Annotations_Val_mscoco.zip

wget -P ${DATA_DIR} https://computing.ece.vt.edu/~aish/vqacp/vqacp_v2_train_annotations.json
wget -P ${DATA_DIR} https://computing.ece.vt.edu/~aish/vqacp/vqacp_v2_test_annotations.json
wget -P ${DATA_DIR} https://computing.ece.vt.edu/~aish/vqacp/vqacp_v2_train_questions.json
wget -P ${DATA_DIR} https://computing.ece.vt.edu/~aish/vqacp/vqacp_v2_test_questions.json



#wget -P ${DATA_DIR} http://www.cs.utexas.edu/~jialinwu/dataset/VQA_caption_traindataset.pkl
#wget -P ${DATA_DIR} http://www.cs.utexas.edu/~jialinwu/dataset/VQA_caption_valdataset.pkl

