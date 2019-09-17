export SOCIALMEDIAIE_PATH="/content/gdrive/My Drive/SocialMediaIE/"

echo "${SOCIALMEDIAIE_PATH}"
ls -ltrh "${SOCIALMEDIAIE_PATH}/data"
realpath "${SOCIALMEDIAIE_PATH}"
cd "${SOCIALMEDIAIE_PATH}" && ls -ltrh


# Install dependencies

pip install torch allennlp
echo "${SOCIALMEDIAIE_PATH}"

# Install socialmediaIE
pip install -e "${SOCIALMEDIAIE_PATH}"

# Run experiments

EXPERIMENT_SCRIPTS_DIR="${SOCIALMEDIAIE_PATH}/experiments"
SCRIPT_FILE="${EXPERIMENT_SCRIPTS_DIR}/multitask_multidataset_experiment.py"
DATASET_PATHS_FILE="${EXPERIMENT_SCRIPTS_DIR}/all_dataset_paths.json"

## BiLSTM models

# Multi task shared bilstm
MODEL_DIR="${SOCIALMEDIAIE_PATH}/data/models/all_multitask_shared"
python "${SCRIPT_FILE}" \
--task multimodal_ner broad_ner wnut17_ner neel_ner ritter_ner yodie_ner ritter_chunk ud_pos ark_pos ptb_pos ritter_ccg \
--encoder-type bilstm --multi-task-mode shared \
--weight-decay 1e-3 \
--model-dir "${MODEL_DIR}" \
--clean-model-dir --cuda \
--dataset-path-prefix "${EXPERIMENT_SCRIPTS_DIR}" \
--dataset-paths-file "${DATASET_PATHS_FILE}"


# Multi task stacked bilstm
MODEL_DIR="${SOCIALMEDIAIE_PATH}/data/models/all_multitask_stacked"
python "${SCRIPT_FILE}" \
--task multimodal_ner broad_ner wnut17_ner neel_ner ritter_ner yodie_ner ritter_chunk ud_pos ark_pos ptb_pos ritter_ccg \
--encoder-type bilstm --multi-task-mode stacked \
--weight-decay 1e-3 \
--model-dir "${MODEL_DIR}" \
--clean-model-dir --cuda \
--dataset-path-prefix "${EXPERIMENT_SCRIPTS_DIR}" \
--dataset-paths-file "${DATASET_PATHS_FILE}"


# POS all data bilstm
MODEL_DIR="${SOCIALMEDIAIE_PATH}/data/models/all_pos_bilstm"
python "${SCRIPT_FILE}" \
--task ud_pos ark_pos ptb_pos \
--encoder-type bilstm --multi-task-mode shared \
--weight-decay 1e-3 \
--model-dir "${MODEL_DIR}" \
--clean-model-dir --cuda \
--dataset-path-prefix "${EXPERIMENT_SCRIPTS_DIR}" \
--dataset-paths-file "${DATASET_PATHS_FILE}"


# NER all data bilstm
MODEL_DIR="${SOCIALMEDIAIE_PATH}/data/models/all_ner_bilstm"
python "${SCRIPT_FILE}" \
--task multimodal_ner broad_ner wnut17_ner neel_ner ritter_ner yodie_ner \
--encoder-type bilstm --multi-task-mode shared \
--weight-decay 1e-3 \
--model-dir "${MODEL_DIR}" \
--clean-model-dir --cuda \
--dataset-path-prefix "${EXPERIMENT_SCRIPTS_DIR}" \
--dataset-paths-file "${DATASET_PATHS_FILE}"

# Chunking single data bilstm
MODEL_DIR="${SOCIALMEDIAIE_PATH}/data/models/ritter_chunk_bilstm"
python "${SCRIPT_FILE}" \
--task ritter_chunk \
--encoder-type bilstm --multi-task-mode shared \
--weight-decay 1e-3 \
--model-dir "${MODEL_DIR}" \
--clean-model-dir --cuda \
--dataset-path-prefix "${EXPERIMENT_SCRIPTS_DIR}" \
--dataset-paths-file "${DATASET_PATHS_FILE}"

# CCG single data bilstm
MODEL_DIR="${SOCIALMEDIAIE_PATH}/data/models/ritter_ccg_bilstm"
python "${SCRIPT_FILE}" \
--task ritter_ccg \
--encoder-type bilstm --multi-task-mode shared \
--weight-decay 1e-3 \
--model-dir "${MODEL_DIR}" \
--clean-model-dir --cuda \
--dataset-path-prefix "${EXPERIMENT_SCRIPTS_DIR}" \
--dataset-paths-file "${DATASET_PATHS_FILE}"

# POS single data bilstm
for data_key in ud_pos ark_pos ptb_pos; do
    MODEL_DIR="${SOCIALMEDIAIE_PATH}/data/models/${data_key}_bilstm"
    python "${SCRIPT_FILE}" \
    --task ${data_key} \
    --encoder-type bilstm --multi-task-mode shared \
    --weight-decay 1e-3 \
    --model-dir "${MODEL_DIR}" \
    --clean-model-dir --cuda \
    --dataset-path-prefix "${EXPERIMENT_SCRIPTS_DIR}" \
    --dataset-paths-file "${DATASET_PATHS_FILE}"
done

# NER single data bilstm
for data_key in multimodal_ner broad_ner wnut17_ner neel_ner ritter_ner yodie_ner; do
    MODEL_DIR="${SOCIALMEDIAIE_PATH}/data/models/${data_key}_bilstm"
    python "${SCRIPT_FILE}" \
    --task ${data_key} \
    --encoder-type bilstm --multi-task-mode shared \
    --weight-decay 1e-3 \
    --model-dir "${MODEL_DIR}" \
    --clean-model-dir --cuda \
    --dataset-path-prefix "${EXPERIMENT_SCRIPTS_DIR}" \
    --dataset-paths-file "${DATASET_PATHS_FILE}"
done


## BiLSTM models LR=1e-3 L2=0

# Multi task shared bilstm
MODEL_DIR="${SOCIALMEDIAIE_PATH}/data/models/all_multitask_shared_l2_0_lr_1e-3"
python "${SCRIPT_FILE}" \
--task multimodal_ner broad_ner wnut17_ner neel_ner ritter_ner yodie_ner ritter_chunk ud_pos ark_pos ptb_pos ritter_ccg \
--encoder-type bilstm --multi-task-mode shared \
--weight-decay 0 --lr 1e-3 \
--model-dir "${MODEL_DIR}" \
--clean-model-dir --cuda \
--dataset-path-prefix "${EXPERIMENT_SCRIPTS_DIR}" \
--dataset-paths-file "${DATASET_PATHS_FILE}"


# Multi task stacked bilstm
MODEL_DIR="${SOCIALMEDIAIE_PATH}/data/models/all_multitask_stacked_l2_0_lr_1e-3"
python "${SCRIPT_FILE}" \
--task multimodal_ner broad_ner wnut17_ner neel_ner ritter_ner yodie_ner ritter_chunk ud_pos ark_pos ptb_pos ritter_ccg \
--encoder-type bilstm --multi-task-mode stacked \
--weight-decay 0 --lr 1e-3 \
--model-dir "${MODEL_DIR}" \
--clean-model-dir --cuda \
--dataset-path-prefix "${EXPERIMENT_SCRIPTS_DIR}" \
--dataset-paths-file "${DATASET_PATHS_FILE}"


# POS all data bilstm
MODEL_DIR="${SOCIALMEDIAIE_PATH}/data/models/all_pos_bilstm_l2_0_lr_1e-3"
python "${SCRIPT_FILE}" \
--task ud_pos ark_pos ptb_pos \
--encoder-type bilstm --multi-task-mode shared \
--weight-decay 0 --lr 1e-3 \
--model-dir "${MODEL_DIR}" \
--clean-model-dir --cuda \
--dataset-path-prefix "${EXPERIMENT_SCRIPTS_DIR}" \
--dataset-paths-file "${DATASET_PATHS_FILE}"


# NER all data bilstm
MODEL_DIR="${SOCIALMEDIAIE_PATH}/data/models/all_ner_bilstm_l2_0_lr_1e-3"
python "${SCRIPT_FILE}" \
--task multimodal_ner broad_ner wnut17_ner neel_ner ritter_ner yodie_ner \
--encoder-type bilstm --multi-task-mode shared \
--weight-decay 0 --lr 1e-3 \
--model-dir "${MODEL_DIR}" \
--clean-model-dir --cuda \
--dataset-path-prefix "${EXPERIMENT_SCRIPTS_DIR}" \
--dataset-paths-file "${DATASET_PATHS_FILE}"

# Chunking single data bilstm
MODEL_DIR="${SOCIALMEDIAIE_PATH}/data/models/ritter_chunk_bilstm_l2_0_lr_1e-3"
python "${SCRIPT_FILE}" \
--task ritter_chunk \
--encoder-type bilstm --multi-task-mode shared \
--weight-decay 0 --lr 1e-3 \
--model-dir "${MODEL_DIR}" \
--clean-model-dir --cuda \
--dataset-path-prefix "${EXPERIMENT_SCRIPTS_DIR}" \
--dataset-paths-file "${DATASET_PATHS_FILE}"

# CCG single data bilstm
MODEL_DIR="${SOCIALMEDIAIE_PATH}/data/models/ritter_ccg_bilstm_l2_0_lr_1e-3"
python "${SCRIPT_FILE}" \
--task ritter_ccg \
--encoder-type bilstm --multi-task-mode shared \
--weight-decay 0 --lr 1e-3 \
--model-dir "${MODEL_DIR}" \
--clean-model-dir --cuda \
--dataset-path-prefix "${EXPERIMENT_SCRIPTS_DIR}" \
--dataset-paths-file "${DATASET_PATHS_FILE}"

# POS single data bilstm
for data_key in ud_pos ark_pos ptb_pos; do
    MODEL_DIR="${SOCIALMEDIAIE_PATH}/data/models/${data_key}_bilstm_l2_0_lr_1e-3"
    python "${SCRIPT_FILE}" \
    --task ${data_key} \
    --encoder-type bilstm --multi-task-mode shared \
    --weight-decay 0 --lr 1e-3 \
    --model-dir "${MODEL_DIR}" \
    --clean-model-dir --cuda \
    --dataset-path-prefix "${EXPERIMENT_SCRIPTS_DIR}" \
    --dataset-paths-file "${DATASET_PATHS_FILE}"
done

# NER single data bilstm
for data_key in multimodal_ner broad_ner wnut17_ner neel_ner ritter_ner yodie_ner; do
    MODEL_DIR="${SOCIALMEDIAIE_PATH}/data/models/${data_key}_bilstm_l2_0_lr_1e-3"
    python "${SCRIPT_FILE}" \
    --task ${data_key} \
    --encoder-type bilstm --multi-task-mode shared \
    --weight-decay 0 --lr 1e-3 \
    --model-dir "${MODEL_DIR}" \
    --clean-model-dir --cuda \
    --dataset-path-prefix "${EXPERIMENT_SCRIPTS_DIR}" \
    --dataset-paths-file "${DATASET_PATHS_FILE}"
done



## Stacked Self-Attention LR=1e-3 L2=0

# Multi task shared ssa
MODEL_DIR="${SOCIALMEDIAIE_PATH}/data/models/all_multitask_shared_ssa_l2_0_lr_1e-3"
python "${SCRIPT_FILE}" \
--task multimodal_ner broad_ner wnut17_ner neel_ner ritter_ner yodie_ner ritter_chunk ud_pos ark_pos ptb_pos ritter_ccg \
--encoder-type stacked_self_attention --multi-task-mode shared \
--weight-decay 0 --lr 1e-3 \
--model-dir "${MODEL_DIR}" \
--clean-model-dir --cuda \
--dataset-path-prefix "${EXPERIMENT_SCRIPTS_DIR}" \
--dataset-paths-file "${DATASET_PATHS_FILE}"


# Multi task stacked ssa
MODEL_DIR="${SOCIALMEDIAIE_PATH}/data/models/all_multitask_stacked_ssa_l2_0_lr_1e-3"
python "${SCRIPT_FILE}" \
--task multimodal_ner broad_ner wnut17_ner neel_ner ritter_ner yodie_ner ritter_chunk ud_pos ark_pos ptb_pos ritter_ccg \
--encoder-type stacked_self_attention --multi-task-mode stacked \
--weight-decay 0 --lr 1e-3 \
--model-dir "${MODEL_DIR}" \
--clean-model-dir --cuda \
--dataset-path-prefix "${EXPERIMENT_SCRIPTS_DIR}" \
--dataset-paths-file "${DATASET_PATHS_FILE}"


# POS all data ssa
MODEL_DIR="${SOCIALMEDIAIE_PATH}/data/models/all_pos_ssa_l2_0_lr_1e-3"
python "${SCRIPT_FILE}" \
--task ud_pos ark_pos ptb_pos \
--encoder-type stacked_self_attention --multi-task-mode shared \
--weight-decay 0 --lr 1e-3 \
--model-dir "${MODEL_DIR}" \
--clean-model-dir --cuda \
--dataset-path-prefix "${EXPERIMENT_SCRIPTS_DIR}" \
--dataset-paths-file "${DATASET_PATHS_FILE}"


# NER all data ssa
MODEL_DIR="${SOCIALMEDIAIE_PATH}/data/models/all_ner_ssa_l2_0_lr_1e-3"
python "${SCRIPT_FILE}" \
--task multimodal_ner broad_ner wnut17_ner neel_ner ritter_ner yodie_ner \
--encoder-type stacked_self_attention --multi-task-mode shared \
--weight-decay 0 --lr 1e-3 \
--model-dir "${MODEL_DIR}" \
--clean-model-dir --cuda \
--dataset-path-prefix "${EXPERIMENT_SCRIPTS_DIR}" \
--dataset-paths-file "${DATASET_PATHS_FILE}"

# Chunking single data ssa
MODEL_DIR="${SOCIALMEDIAIE_PATH}/data/models/ritter_chunk_ssa_l2_0_lr_1e-3"
python "${SCRIPT_FILE}" \
--task ritter_chunk \
--encoder-type stacked_self_attention --multi-task-mode shared \
--weight-decay 0 --lr 1e-3 \
--model-dir "${MODEL_DIR}" \
--clean-model-dir --cuda \
--dataset-path-prefix "${EXPERIMENT_SCRIPTS_DIR}" \
--dataset-paths-file "${DATASET_PATHS_FILE}"

# CCG single data ssa
MODEL_DIR="${SOCIALMEDIAIE_PATH}/data/models/ritter_ccg_ssa_l2_0_lr_1e-3"
python "${SCRIPT_FILE}" \
--task ritter_ccg \
--encoder-type stacked_self_attention --multi-task-mode shared \
--weight-decay 0 --lr 1e-3 \
--model-dir "${MODEL_DIR}" \
--clean-model-dir --cuda \
--dataset-path-prefix "${EXPERIMENT_SCRIPTS_DIR}" \
--dataset-paths-file "${DATASET_PATHS_FILE}"

# POS single data ssa
for data_key in ud_pos ark_pos ptb_pos; do
    MODEL_DIR="${SOCIALMEDIAIE_PATH}/data/models/${data_key}_ssa_l2_0_lr_1e-3"
    python "${SCRIPT_FILE}" \
    --task ${data_key} \
    --encoder-type stacked_self_attention --multi-task-mode shared \
    --weight-decay 0 --lr 1e-3 \
    --model-dir "${MODEL_DIR}" \
    --clean-model-dir --cuda \
    --dataset-path-prefix "${EXPERIMENT_SCRIPTS_DIR}" \
    --dataset-paths-file "${DATASET_PATHS_FILE}"
done

# NER single data bilstm
for data_key in multimodal_ner broad_ner wnut17_ner neel_ner ritter_ner yodie_ner; do
    MODEL_DIR="${SOCIALMEDIAIE_PATH}/data/models/${data_key}_ssa_l2_0_lr_1e-3"
    python "${SCRIPT_FILE}" \
    --task ${data_key} \
    --encoder-type stacked_self_attention --multi-task-mode shared \
    --weight-decay 0 --lr 1e-3 \
    --model-dir "${MODEL_DIR}" \
    --clean-model-dir --cuda \
    --dataset-path-prefix "${EXPERIMENT_SCRIPTS_DIR}" \
    --dataset-paths-file "${DATASET_PATHS_FILE}"
done



## No NEEL experiments

# Multi task stacked ssa
MODEL_DIR="${SOCIALMEDIAIE_PATH}/data/models/all_multitask_stacked_ssa_l2_0_lr_1e-3_no_neel"
python "${SCRIPT_FILE}" \
--task multimodal_ner broad_ner wnut17_ner ritter_ner yodie_ner ritter_chunk ud_pos ark_pos ptb_pos ritter_ccg \
--encoder-type stacked_self_attention --multi-task-mode stacked \
--weight-decay 0 --lr 1e-3 \
--model-dir "${MODEL_DIR}" \
--clean-model-dir --cuda \
--dataset-path-prefix "${EXPERIMENT_SCRIPTS_DIR}" \
--dataset-paths-file "${DATASET_PATHS_FILE}"

# Multi task shared ssa
MODEL_DIR="${SOCIALMEDIAIE_PATH}/data/models/all_multitask_shared_ssa_l2_0_lr_1e-3_no_neel"
python "${SCRIPT_FILE}" \
--task multimodal_ner broad_ner wnut17_ner ritter_ner yodie_ner ritter_chunk ud_pos ark_pos ptb_pos ritter_ccg \
--encoder-type stacked_self_attention --multi-task-mode shared \
--weight-decay 0 --lr 1e-3 \
--model-dir "${MODEL_DIR}" \
--clean-model-dir --cuda \
--dataset-path-prefix "${EXPERIMENT_SCRIPTS_DIR}" \
--dataset-paths-file "${DATASET_PATHS_FILE}"


# NER all data ssa
MODEL_DIR="${SOCIALMEDIAIE_PATH}/data/models/all_ner_ssa_l2_0_lr_1e-3_no_neel"
python "${SCRIPT_FILE}" \
--task multimodal_ner broad_ner wnut17_ner ritter_ner yodie_ner \
--encoder-type stacked_self_attention --multi-task-mode shared \
--weight-decay 0 --lr 1e-3 \
--model-dir "${MODEL_DIR}" \
--clean-model-dir --cuda \
--dataset-path-prefix "${EXPERIMENT_SCRIPTS_DIR}" \
--dataset-paths-file "${DATASET_PATHS_FILE}"


# Multi task shared bilstm
MODEL_DIR="${SOCIALMEDIAIE_PATH}/data/models/all_multitask_shared_l2_0_lr_1e-3_no_neel"
python "${SCRIPT_FILE}" \
--task multimodal_ner broad_ner wnut17_ner ritter_ner yodie_ner ritter_chunk ud_pos ark_pos ptb_pos ritter_ccg \
--encoder-type bilstm --multi-task-mode shared \
--weight-decay 0 --lr 1e-3 \
--model-dir "${MODEL_DIR}" \
--clean-model-dir --cuda \
--dataset-path-prefix "${EXPERIMENT_SCRIPTS_DIR}" \
--dataset-paths-file "${DATASET_PATHS_FILE}"


# Multi task stacked bilstm
MODEL_DIR="${SOCIALMEDIAIE_PATH}/data/models/all_multitask_stacked_l2_0_lr_1e-3_no_neel"
python "${SCRIPT_FILE}" \
--task multimodal_ner broad_ner wnut17_ner ritter_ner yodie_ner ritter_chunk ud_pos ark_pos ptb_pos ritter_ccg \
--encoder-type bilstm --multi-task-mode stacked \
--weight-decay 0 --lr 1e-3 \
--model-dir "${MODEL_DIR}" \
--clean-model-dir --cuda \
--dataset-path-prefix "${EXPERIMENT_SCRIPTS_DIR}" \
--dataset-paths-file "${DATASET_PATHS_FILE}"


# NER all data bilstm
MODEL_DIR="${SOCIALMEDIAIE_PATH}/data/models/all_ner_bilstm_l2_0_lr_1e-3_no_neel"
python "${SCRIPT_FILE}" \
--task multimodal_ner broad_ner wnut17_ner ritter_ner yodie_ner \
--encoder-type bilstm --multi-task-mode shared \
--weight-decay 0 --lr 1e-3 \
--model-dir "${MODEL_DIR}" \
--clean-model-dir --cuda \
--dataset-path-prefix "${EXPERIMENT_SCRIPTS_DIR}" \
--dataset-paths-file "${DATASET_PATHS_FILE}"



## Residual experiments

# Multi task stacked bilstm
MODEL_DIR="${SOCIALMEDIAIE_PATH}/data/models/all_multitask_stacked_bilstm_residual_l2_0_lr_1e-3_no_neel"
python "${SCRIPT_FILE}" \
--task multimodal_ner broad_ner wnut17_ner ritter_ner yodie_ner ritter_chunk ud_pos ark_pos ptb_pos ritter_ccg \
--encoder-type bilstm --multi-task-mode stacked \
--residual-connection --proj-dim 200 \
--weight-decay 0 --lr 1e-3 \
--model-dir "${MODEL_DIR}" \
--clean-model-dir --cuda \
--dataset-path-prefix "${EXPERIMENT_SCRIPTS_DIR}" \
--dataset-paths-file "${DATASET_PATHS_FILE}"

# Multi task shared bilstm
MODEL_DIR="${SOCIALMEDIAIE_PATH}/data/models/all_multitask_shared_bilstm_residual_l2_0_lr_1e-3_no_neel"
python "${SCRIPT_FILE}" \
--task multimodal_ner broad_ner wnut17_ner ritter_ner yodie_ner ritter_chunk ud_pos ark_pos ptb_pos ritter_ccg \
--encoder-type bilstm --multi-task-mode shared \
--residual-connection --proj-dim 200 \
--weight-decay 0 --lr 1e-3 \
--model-dir "${MODEL_DIR}" \
--clean-model-dir --cuda \
--dataset-path-prefix "${EXPERIMENT_SCRIPTS_DIR}" \
--dataset-paths-file "${DATASET_PATHS_FILE}"


# NER all data bilstm
MODEL_DIR="${SOCIALMEDIAIE_PATH}/data/models/all_ner_bilstm_residual_l2_0_lr_1e-3_no_neel"
python "${SCRIPT_FILE}" \
--task multimodal_ner broad_ner wnut17_ner ritter_ner yodie_ner \
--encoder-type bilstm --multi-task-mode shared \
--weight-decay 0 --lr 1e-3 \
--residual-connection --proj-dim 200 \
--model-dir "${MODEL_DIR}" \
--clean-model-dir --cuda \
--dataset-path-prefix "${EXPERIMENT_SCRIPTS_DIR}" \
--dataset-paths-file "${DATASET_PATHS_FILE}"


# POS all data bilstm
MODEL_DIR="${SOCIALMEDIAIE_PATH}/data/models/all_pos_bilstm_residual_l2_0_lr_1e-3"
python "${SCRIPT_FILE}" \
--task ud_pos ark_pos ptb_pos \
--encoder-type bilstm --multi-task-mode shared \
--weight-decay 0 --lr 1e-3 \
--residual-connection --proj-dim 200 \
--model-dir "${MODEL_DIR}" \
--clean-model-dir --cuda \
--dataset-path-prefix "${EXPERIMENT_SCRIPTS_DIR}" \
--dataset-paths-file "${DATASET_PATHS_FILE}"


# NER single data bilstm
for data_key in multimodal_ner broad_ner wnut17_ner neel_ner ritter_ner yodie_ner; do
    MODEL_DIR="${SOCIALMEDIAIE_PATH}/data/models/${data_key}_bilstm_residual_l2_0_lr_1e-3"
    python "${SCRIPT_FILE}" \
    --task ${data_key} \
    --encoder-type bilstm --multi-task-mode shared \
    --weight-decay 0 --lr 1e-3 \
    --residual-connection --proj-dim 200 \
    --model-dir "${MODEL_DIR}" \
    --clean-model-dir --cuda \
    --dataset-path-prefix "${EXPERIMENT_SCRIPTS_DIR}" \
    --dataset-paths-file "${DATASET_PATHS_FILE}"
done


# POS single data bilstm
for data_key in ud_pos ark_pos ptb_pos; do
    MODEL_DIR="${SOCIALMEDIAIE_PATH}/data/models/${data_key}_bilstm_residual_l2_0_lr_1e-3"
    python "${SCRIPT_FILE}" \
    --task ${data_key} \
    --encoder-type bilstm --multi-task-mode shared \
    --weight-decay 0 --lr 1e-3 \
    --residual-connection --proj-dim 200 \
    --model-dir "${MODEL_DIR}" \
    --clean-model-dir --cuda \
    --dataset-path-prefix "${EXPERIMENT_SCRIPTS_DIR}" \
    --dataset-paths-file "${DATASET_PATHS_FILE}"
done


# Chunk and CCG single data bilstm
for data_key in ritter_chunk ritter_ccg; do
    MODEL_DIR="${SOCIALMEDIAIE_PATH}/data/models/${data_key}_bilstm_residual_l2_0_lr_1e-3"
    python "${SCRIPT_FILE}" \
    --task ${data_key} \
    --encoder-type bilstm --multi-task-mode shared \
    --weight-decay 0 --lr 1e-3 \
    --residual-connection --proj-dim 200 \
    --model-dir "${MODEL_DIR}" \
    --clean-model-dir --cuda \
    --dataset-path-prefix "${EXPERIMENT_SCRIPTS_DIR}" \
    --dataset-paths-file "${DATASET_PATHS_FILE}"
done


## Residual with hidden dim 50

# Multi task stacked bilstm
MODEL_DIR="${SOCIALMEDIAIE_PATH}/data/models/all_multitask_stacked_bilstm_residual_dim_50_l2_0_lr_1e-3_no_neel"
python "${SCRIPT_FILE}" \
--task multimodal_ner broad_ner wnut17_ner ritter_ner yodie_ner ritter_chunk ud_pos ark_pos ptb_pos ritter_ccg \
--encoder-type bilstm --multi-task-mode stacked \
--residual-connection --proj-dim 100 --hidden-dim 50 \
--weight-decay 0 --lr 1e-3 \
--model-dir "${MODEL_DIR}" \
--clean-model-dir --cuda \
--dataset-path-prefix "${EXPERIMENT_SCRIPTS_DIR}" \
--dataset-paths-file "${DATASET_PATHS_FILE}"

# Multi task shared bilstm
MODEL_DIR="${SOCIALMEDIAIE_PATH}/data/models/all_multitask_shared_bilstm_residual_dim_50_l2_0_lr_1e-3_no_neel"
python "${SCRIPT_FILE}" \
--task multimodal_ner broad_ner wnut17_ner ritter_ner yodie_ner ritter_chunk ud_pos ark_pos ptb_pos ritter_ccg \
--encoder-type bilstm --multi-task-mode shared \
--residual-connection --proj-dim 100 --hidden-dim 50 \
--weight-decay 0 --lr 1e-3 \
--model-dir "${MODEL_DIR}" \
--clean-model-dir --cuda \
--dataset-path-prefix "${EXPERIMENT_SCRIPTS_DIR}" \
--dataset-paths-file "${DATASET_PATHS_FILE}"


# NER all data bilstm
MODEL_DIR="${SOCIALMEDIAIE_PATH}/data/models/all_ner_bilstm_residual_dim_50_l2_0_lr_1e-3_no_neel"
python "${SCRIPT_FILE}" \
--task multimodal_ner broad_ner wnut17_ner ritter_ner yodie_ner \
--encoder-type bilstm --multi-task-mode shared \
--weight-decay 0 --lr 1e-3 \
--residual-connection --proj-dim 100 --hidden-dim 50 \
--model-dir "${MODEL_DIR}" \
--clean-model-dir --cuda \
--dataset-path-prefix "${EXPERIMENT_SCRIPTS_DIR}" \
--dataset-paths-file "${DATASET_PATHS_FILE}"


# POS all data bilstm
MODEL_DIR="${SOCIALMEDIAIE_PATH}/data/models/all_pos_bilstm_residual_dim_50_l2_0_lr_1e-3"
python "${SCRIPT_FILE}" \
--task ud_pos ark_pos ptb_pos \
--encoder-type bilstm --multi-task-mode shared \
--weight-decay 0 --lr 1e-3 \
--residual-connection --proj-dim 100 --hidden-dim 50 \
--model-dir "${MODEL_DIR}" \
--clean-model-dir --cuda \
--dataset-path-prefix "${EXPERIMENT_SCRIPTS_DIR}" \
--dataset-paths-file "${DATASET_PATHS_FILE}"


# All single data bilstm
for data_key in multimodal_ner broad_ner wnut17_ner neel_ner ritter_ner yodie_ner ritter_chunk ritter_ccg ud_pos ark_pos ptb_pos; do
    MODEL_DIR="${SOCIALMEDIAIE_PATH}/data/models/${data_key}_bilstm_residual_dim_50_l2_0_lr_1e-3"
    python "${SCRIPT_FILE}" \
    --task ${data_key} \
    --encoder-type bilstm --multi-task-mode shared \
    --weight-decay 0 --lr 1e-3 \
    --residual-connection --proj-dim 100 --hidden-dim 50 \
    --model-dir "${MODEL_DIR}" \
    --clean-model-dir --cuda \
    --dataset-path-prefix "${EXPERIMENT_SCRIPTS_DIR}" \
    --dataset-paths-file "${DATASET_PATHS_FILE}"
done