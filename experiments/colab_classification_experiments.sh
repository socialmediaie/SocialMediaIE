pip install torch allennlp
# Install apex
if [ ! -d "/content/apex" ]; then
  git clone https://github.com/NVIDIA/apex.git
  pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" /content/apex
else
  echo "Apex cloned and installed";
fi

from google.colab import drive
drive.mount('/content/gdrive')

export SOCIALMEDIAIE_PATH="/content/gdrive/My Drive/SocialMediaIE/"

echo "${SOCIALMEDIAIE_PATH}"
ls -ltrh "${SOCIALMEDIAIE_PATH}/data"
realpath "${SOCIALMEDIAIE_PATH}"
cd "${SOCIALMEDIAIE_PATH}" && ls -ltrh


# Install socialmediaIE
pip install -e "${SOCIALMEDIAIE_PATH}"

ABUSIVE_DATA=(founta_abusive waseem_abusive)
UNCERTAINITY_DATA=(sarcasm_uncertainity veridicality_uncertainity)
SENTIMENT_DATA=(semeval_sentiment clarin_sentiment politics_sentiment other_sentiment)
DATASETS=("${ABUSIVE_DATA[@]}" "${UNCERTAINITY_DATA[@]}" "${SENTIMENT_DATA[@]}")
echo ${DATASETS[@]}

# Run experiments

EXPERIMENT_SCRIPTS_DIR="${SOCIALMEDIAIE_PATH}/experiments"
SCRIPT_FILE="${SOCIALMEDIAIE_PATH}/SocialMediaIE/scripts/multitask_multidataset_classification.py"
DATASET_PATHS_FILE="${EXPERIMENT_SCRIPTS_DIR}/all_classification_dataset_paths.json"
ABUSIVE_DATA=(founta_abusive waseem_abusive)
UNCERTAINITY_DATA=(sarcasm_uncertainity veridicality_uncertainity)
SENTIMENT_DATA=(semeval_sentiment clarin_sentiment politics_sentiment other_sentiment)
DATASETS=("${ABUSIVE_DATA[@]}" "${UNCERTAINITY_DATA[@]}" "${SENTIMENT_DATA[@]}")
echo ${DATASETS[@]}
TASK_NAMES=("abusive" "uncertainity" "sentiment")
DATA_STARTS=(0 2 4)
DATA_SIZES=(2 2 4)
MULTITASK_SETTINGS=("stacked" "shared")

## BiLSTM models normal

# Single dataset training
for data_key in ${DATASETS[@]}; do
    MODEL_DIR="${SOCIALMEDIAIE_PATH}/data/models_classification/${data_key}_bilstm"
    python "${SCRIPT_FILE}" \
    --task ${data_key} \
    --encoder-type bilstm \
    --batch-size 64 \
    --model-dir "${MODEL_DIR}" \
    --clean-model-dir --cuda \
    --dataset-path-prefix "${EXPERIMENT_SCRIPTS_DIR}" \
    --dataset-paths-file "${DATASET_PATHS_FILE}"
done

# Multi dataset training
for ((i=0;i<${#TASK_NAMES[@]};++i)); do
  echo $i ${TASK_NAMES[$i]} ${DATA_STARTS[$i]} ${DATA_SIZES[$i]} ${DATASETS[@]:${DATA_STARTS[$i]}:${DATA_SIZES[$i]}};
  task_key=${TASK_NAMES[$i]};
  MODEL_DIR="${SOCIALMEDIAIE_PATH}/data/models_classification/all_${task_key}_bilstm"
  python "${SCRIPT_FILE}" \
  --task ${DATASETS[@]:${DATA_STARTS[$i]}:${DATA_SIZES[$i]}} \
  --encoder-type bilstm --multi-task-mode shared \
  --batch-size 64 \
  --model-dir "${MODEL_DIR}" \
  --clean-model-dir --cuda \
  --dataset-path-prefix "${EXPERIMENT_SCRIPTS_DIR}" \
  --dataset-paths-file "${DATASET_PATHS_FILE}"
done

# Multi task training batch 32 for less memory
for multi_task_mode in ${MULTITASK_SETTINGS[@]}; do
  MODEL_DIR="${SOCIALMEDIAIE_PATH}/data/models_classification/all_multitask_${multi_task_mode}_bilstm"
  python "${SCRIPT_FILE}" \
  --task ${DATASETS[@]} \
  --encoder-type bilstm --multi-task-mode ${multi_task_mode} \
  --batch-size 16 \
  --model-dir "${MODEL_DIR}" \
  --clean-model-dir --cuda \
  --dataset-path-prefix "${EXPERIMENT_SCRIPTS_DIR}" \
  --dataset-paths-file "${DATASET_PATHS_FILE}"
done

## BiLSTM models L2=0 LR=1e-3

# Single dataset training
for data_key in ${DATASETS[@]}; do
    MODEL_DIR="${SOCIALMEDIAIE_PATH}/data/models_classification/${data_key}_bilstm_l2_0_lr_1e-3"
    python "${SCRIPT_FILE}" \
    --task ${data_key} \
    --encoder-type bilstm \
    --weight-decay 0 --lr 1e-3 \
    --batch-size 64 \
    --model-dir "${MODEL_DIR}" \
    --clean-model-dir --cuda \
    --dataset-path-prefix "${EXPERIMENT_SCRIPTS_DIR}" \
    --dataset-paths-file "${DATASET_PATHS_FILE}"
done

# Multi dataset training
for ((i=0;i<${#TASK_NAMES[@]};++i)); do
  echo $i ${TASK_NAMES[$i]} ${DATA_STARTS[$i]} ${DATA_SIZES[$i]} ${DATASETS[@]:${DATA_STARTS[$i]}:${DATA_SIZES[$i]}};
  task_key=${TASK_NAMES[$i]};
  MODEL_DIR="${SOCIALMEDIAIE_PATH}/data/models_classification/all_${task_key}_bilstm_l2_0_lr_1e-3"
  python "${SCRIPT_FILE}" \
  --task ${DATASETS[@]:${DATA_STARTS[$i]}:${DATA_SIZES[$i]}} \
  --encoder-type bilstm --multi-task-mode shared \
  --weight-decay 0 --lr 1e-3 \
  --batch-size 64 \
  --model-dir "${MODEL_DIR}" \
  --clean-model-dir --cuda \
  --dataset-path-prefix "${EXPERIMENT_SCRIPTS_DIR}" \
  --dataset-paths-file "${DATASET_PATHS_FILE}"
done


# Multi task training batch 16 for less memory
for multi_task_mode in ${MULTITASK_SETTINGS[@]}; do
  MODEL_DIR="${SOCIALMEDIAIE_PATH}/data/models_classification/all_multitask_${multi_task_mode}_bilstm_l2_0_lr_1e-3"
  python "${SCRIPT_FILE}" \
  --task ${DATASETS[@]} \
  --encoder-type bilstm --multi-task-mode ${multi_task_mode} \
  --weight-decay 0 --lr 1e-3 \
  --batch-size 16 \
  --model-dir "${MODEL_DIR}" \
  --clean-model-dir --cuda \
  --dataset-path-prefix "${EXPERIMENT_SCRIPTS_DIR}" \
  --dataset-paths-file "${DATASET_PATHS_FILE}"
done


## CNN models L2=0 LR=1e-3

# Single dataset training
for data_key in ${DATASETS[@]}; do
    MODEL_DIR="${SOCIALMEDIAIE_PATH}/data/models_classification/${data_key}_cnn_l2_0_lr_1e-3"
    python "${SCRIPT_FILE}" \
    --task ${data_key} \
    --encoder-type cnn \
    --weight-decay 0 --lr 1e-3 \
    --batch-size 64 \
    --model-dir "${MODEL_DIR}" \
    --clean-model-dir --cuda \
    --dataset-path-prefix "${EXPERIMENT_SCRIPTS_DIR}" \
    --dataset-paths-file "${DATASET_PATHS_FILE}"
done

# Multi dataset training
for ((i=0;i<${#TASK_NAMES[@]};++i)); do
  echo $i ${TASK_NAMES[$i]} ${DATA_STARTS[$i]} ${DATA_SIZES[$i]} ${DATASETS[@]:${DATA_STARTS[$i]}:${DATA_SIZES[$i]}};
  task_key=${TASK_NAMES[$i]};
  MODEL_DIR="${SOCIALMEDIAIE_PATH}/data/models_classification/all_${task_key}_cnn_l2_0_lr_1e-3"
  python "${SCRIPT_FILE}" \
  --task ${DATASETS[@]:${DATA_STARTS[$i]}:${DATA_SIZES[$i]}} \
  --encoder-type cnn --multi-task-mode shared \
  --weight-decay 0 --lr 1e-3 \
  --batch-size 64 \
  --model-dir "${MODEL_DIR}" \
  --clean-model-dir --cuda \
  --dataset-path-prefix "${EXPERIMENT_SCRIPTS_DIR}" \
  --dataset-paths-file "${DATASET_PATHS_FILE}"
done

# Multi task training batch 32 for less memory
for multi_task_mode in ${MULTITASK_SETTINGS[@]}; do
  MODEL_DIR="${SOCIALMEDIAIE_PATH}/data/models_classification/all_multitask_${multi_task_mode}_cnn_l2_0_lr_1e-3"
  python "${SCRIPT_FILE}" \
  --task ${DATASETS[@]} \
  --encoder-type cnn --multi-task-mode ${multi_task_mode} \
  --weight-decay 0 --lr 1e-3 \
  --batch-size 16 \
  --model-dir "${MODEL_DIR}" \
  --clean-model-dir --cuda \
  --dataset-path-prefix "${EXPERIMENT_SCRIPTS_DIR}" \
  --dataset-paths-file "${DATASET_PATHS_FILE}"
done