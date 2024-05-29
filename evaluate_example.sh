#!/usr/bin/env bash
set -x

WORK_DIR=$(dirname $(readlink -f $0))
echo "${WORK_DIR}"
UUID=$(uuidgen)
echo "${UUID}"
CONDA_HOME=/home/tanminghuan/anaconda3
CONDA_ENV=base

OUTPUT_DIR="${WORK_DIR}"/output/${UUID}
mkdir -p "${OUTPUT_DIR}"
log_file="${OUTPUT_DIR}"/logs.txt
exec &> >(tee -a "$log_file")

PID=$BASHPID
echo "$PID"

#METHOD=baseline
METHOD=multi_scorer
TASK=wassa2023
SPLIT=validation

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('${CONDA_HOME}/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "${CONDA_HOME}/etc/profile.d/conda.sh" ]; then
        . "${CONDA_HOME}/etc/profile.d/conda.sh"
    else
        export PATH="${CONDA_HOME}/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate $CONDA_ENV

start_controller_script="cd ${WORK_DIR}/src && LOG_DIR=${OUTPUT_DIR} STATIC_PATH=${WORK_DIR}/static python -m wassa.controller --host 0.0.0.0 --port 5000"
echo "$start_controller_script"
#screen -dmS start_webui_$PID bash -c "$start_webui_script"
tmux new-session -d -s wassa_controller_$PID "$start_controller_script"

sleep 10

start_llm_script="cd ${WORK_DIR}/src && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=${WORK_DIR}/related_repos/LLaMA-Factory/src CONTROLLER_ADDRESS=http://localhost:5000 MODEL_NAME=Qwen1.5-7B-Chat WORKER_ADDRESS=http://localhost:40001 WORKER_HOST=0.0.0.0 WORKER_PORT=40001 python -m wassa.workers.llm  --model_name_or_path /data2/public/pretrained_models/Qwen1.5-7B-Chat   --template qwen --infer_backend vllm --vllm_enforce_eager --vllm_maxlen 8192"
echo "$start_llm_script"
#screen -dmS start_llm_$PID bash -c "$start_llm_script"
tmux new-session -d -s wassa_llm_$PID "$start_llm_script"

sleep 10

PYTHONPATH="${WORK_DIR}"/src python "${WORK_DIR}"/evaluate.py \
  --task "${TASK}" --method "${METHOD}" --split "${SPLIT}" --debug \
  --task_dir "${WORK_DIR}"/evaluations/llmeval \
  --model_name Qwen1.5-7B-Chat \
  --model_api_key "EMPTY" \
  --model_api_base http://localhost:40001/v1 \
  --evaluator_name Qwen1.5-7B-Chat \
  --evaluator_api_key "EMPTY" \
  --evaluator_api_base http://localhost:40001/v1
