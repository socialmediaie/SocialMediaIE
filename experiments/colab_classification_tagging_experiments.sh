pip install torch==1.0.0 allennlp==0.8.3
# Install apex
if [ ! -d "/content/apex" ]; then
  git clone https://github.com/NVIDIA/apex.git
  cd /content/apex && git checkout b82c6bd7613ffb9c4ea68e7306fa83aabe9fa9b5 && git checkout -b socialmediaIE_version
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
echo "${SOCIALMEDIAIE_PATH}"
pip install -e "${SOCIALMEDIAIE_PATH}"

# Set datasets
ABUSIVE_DATA=(founta_abusive waseem_abusive)
UNCERTAINITY_DATA=(sarcasm_uncertainity veridicality_uncertainity)
SENTIMENT_DATA=(semeval_sentiment clarin_sentiment politics_sentiment other_sentiment)

NER_DATA=(multimodal_ner broad_ner wnut17_ner neel_ner ritter_ner yodie_ner)
CHUNK_DATA=(ritter_chunk)
CCG_DATA=(ritter_ccg)
POS_DATA=(ud_pos ark_pos ptb_pos)

DATASETS=("${ABUSIVE_DATA[@]}" "${UNCERTAINITY_DATA[@]}" "${SENTIMENT_DATA[@]}" "${NER_DATA[@]}" "${POS_DATA[@]}" "${CHUNK_DATA[@]}" "${CCG_DATA[@]}")
echo ${DATASETS[@]}

# Run experiments

EXPERIMENT_SCRIPTS_DIR="${SOCIALMEDIAIE_PATH}/experiments"
SCRIPT_FILE="${SOCIALMEDIAIE_PATH}/SocialMediaIE/scripts/multitask_multidataset_classification_tagging.py"
DATASET_PATHS_FILE="${EXPERIMENT_SCRIPTS_DIR}/all_classification_tagging_dataset_paths.json"
MULTITASK_SETTINGS=("stacked", "shared")

## BiLSTM models normal

# Multitask stacked dataset training
# Multi task training batch 32 for less memory
for multi_task_mode in ${MULTITASK_SETTINGS[@]}; do
  MODEL_DIR="${SOCIALMEDIAIE_PATH}/data/models_classification_tagging/all_multitask_${multi_task_mode}_bilstm"
  python "${SCRIPT_FILE}" \
  --task ${DATASETS[@]} \
  --encoder-type bilstm --multi-task-mode ${multi_task_mode} \
  --batch-size 16 \
  --model-dir "${MODEL_DIR}" \
  --clean-model-dir --cuda \
  --dataset-path-prefix "${EXPERIMENT_SCRIPTS_DIR}" \
  --dataset-paths-file "${DATASET_PATHS_FILE}" 2> "${MODEL_DIR}.err"
done

## BiLSTM models L2=0 LR=1e-3

# Multitask stacked dataset training
# Multi task training batch 32 for less memory
for multi_task_mode in ${MULTITASK_SETTINGS[@]}; do
  MODEL_DIR="${SOCIALMEDIAIE_PATH}/data/models_classification_tagging/all_multitask_${multi_task_mode}_bilstm_l2_0_lr_1e-3"
  python "${SCRIPT_FILE}" \
  --task ${DATASETS[@]} \
  --encoder-type bilstm --multi-task-mode ${multi_task_mode} \
  --weight-decay 0 --lr 1e-3 \
  --batch-size 16 \
  --model-dir "${MODEL_DIR}" \
  --clean-model-dir --cuda \
  --dataset-path-prefix "${EXPERIMENT_SCRIPTS_DIR}" \
  --dataset-paths-file "${DATASET_PATHS_FILE}" 2> "${MODEL_DIR}.err"
done