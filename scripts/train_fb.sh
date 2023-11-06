#!/usr/bin/env bash
# OUTPUT_DIR=./checkpoint/fb15k237_debug/ bash scripts/train_fb.sh
# OUTPUT_DIR=./checkpoint/fb15k237_vgae_gat/ bash scripts/train_fb.sh


set -x
set -e

TASK="FB15k237"

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
--pooling mean \
--lr 3e-5 \
--use-link-graph \
--train-path "$DATA_DIR/train.txt.json" \
--valid-path "$DATA_DIR/valid.txt.json" \
--task ${TASK} \
--batch-size 500 \
--print-freq 50 \
--additive-margin 0.02 \
--use-amp \
--use-self-negative \
--finetune-t \
--pre-batch 2 \
--epochs 7 \
--workers 1 \
--max-to-keep 5 "$@" \
--model gat \
--n-neighbor 2 \
--n-hop-graph 2 \
--pretrained-model bert-base-uncased
# --freeze-lm

# --train-extract-path "${DATA_DIR}/v2.extraction.desc.neg.pos.pt.json" \
# --pretrained-model /home/irene/Play/HF/bert-base-uncased-own/checkpoint-3000/
# --batch-size 400 \ n-neighbor 5, max-to-keep 5

# comment: 4 GPUs run with 800 as batch size


# before vgae, large lr
# CUDA_VISIBLE_DEVICES=0 python3 -u main.py \
# --model-dir "${OUTPUT_DIR}" \
# --pooling mean \
# --lr 1e-5 \
# --use-link-graph \
# --train-path "$DATA_DIR/train.txt.json" \
# --valid-path "$DATA_DIR/valid.txt.json" \
# --task ${TASK} \
# --batch-size 100 \
# --print-freq 50 \
# --additive-margin 0.02 \
# --use-amp \
# --use-self-negative \
# --finetune-t \
# --pre-batch 2 \
# --epochs 7 \
# --workers 1 \
# --max-to-keep 5 "$@" \
# --model vgae \
# --n-neighbor 5 \
# --n-hop-graph 1 \
# --pretrained-model bert-base-uncased
