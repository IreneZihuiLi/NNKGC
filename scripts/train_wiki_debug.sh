#!/usr/bin/env bash
# OUTPUT_DIR=./checkpoint/wiki_trans/ bash scripts/train_wiki_debug.sh wiki5m_trans

set -x
set -e

TASK="wiki5m_ind"
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    TASK=$1
    shift
fi

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
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -u main.py \
--model-dir "${OUTPUT_DIR}" \
--pretrained-model bert-base-uncased \
--pooling mean \
--lr 3e-5 \
--train-path "${DATA_DIR}/train.txt.json" \
--valid-path "${DATA_DIR}/valid.txt.json" \
--task "${TASK}" \
--batch-size 700 \
--print-freq 100 \
--additive-margin 0.02 \
--use-amp \
--use-self-negative \
--finetune-t \
--pre-batch 0 \
--epochs 1 \
--workers 1 \
--max-to-keep 10 "$@" \
--model gcns \
--n-neighbor 3 \
--n-hop-graph 1

# --train-extract-path "${DATA_DIR}/v2.extraction.desc.pos.pt.json" \
