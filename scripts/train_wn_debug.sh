#!/usr/bin/env bash
# run with >> OUTPUT_DIR=./checkpoint/wn18rr_debug/ bash scripts/train_wn_debug.sh

set -x
set -e

TASK="WN18RR"

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/checkpoint/${TASK}_$(date +%F-%H%M.%S)"
fi
# if [ -z "$DATA_DIR" ]; then
#   DATA_DIR="${DIR}/data/${TASK}"
# fi
DATA_DIR="/home/irene/Projects/2022Fall/simkgc_data/data/${TASK}"
echo "data directory: ${DATA_DIR}"
# OUTPUT_DIR=./checkpoint/wn18rr_gat_pt_3/
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -u main.py \
--model-dir "${OUTPUT_DIR}" \
--pretrained-model bert-base-uncased \
--pooling mean \
--lr 5e-5 \
--use-link-graph \
--train-path "${DATA_DIR}/train.txt.json" \
--valid-path "${DATA_DIR}/valid.txt.json" \
--task ${TASK} \
--batch-size 1000 \
--print-freq 50 \
--additive-margin 0.02 \
--use-amp \
--use-self-negative \
--pre-batch 0 \
--finetune-t \
--epochs 25 \
--workers 0 \
--max-to-keep 10 "$@" \
--model sage \
--n-neighbor 3 \
--n-hop-graph 3


# --freeze-lm
# --lr 5e-5 \
# --pretrained-model /home/irene/Play/HF/bert-base-uncased-WN18RR-1016/ \
# --train-extract-path "${DATA_DIR}/v2.extraction.desc.neg.pos.pt.json" \
# --model gcns,gat,vgae \
