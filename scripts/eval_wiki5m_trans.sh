#!/usr/bin/env bash
# >> bash scripts/eval_wiki5m_trans.sh ./checkpoint/wiki5m_trans_gans/checkpoint_at_step_20000.mdl
set -x
set -e

model_path="bert"
TASK="wiki5m_trans"
DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"
# if [ -z "$DATA_DIR" ]; then
#   DATA_DIR="${DIR}/data/${task}"
# fi
DATA_DIR="/home/irene/Projects/2022Fall/simkgc_data/data/${TASK}"
echo "data directory: ${DATA_DIR}"

if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    model_path=$1
    shift
fi
test_path="$DATA_DIR/test.txt.json"
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    test_path=$1
    shift
fi

neighbor_weight=0.05
echo "??? ${model_path}"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -u eval_wiki5m_trans.py \
--task "${TASK}" \
--is-test \
--batch-size 64 \
--eval-model-path "${model_path}" \
--neighbor-weight "${neighbor_weight}" \
--train-path "$DATA_DIR/train.txt.json" \
--valid-path "${test_path}" "$@"
