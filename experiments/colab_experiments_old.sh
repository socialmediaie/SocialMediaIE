pip install torch allennlp

pip install -e /content/gdrive/My\ Drive/SocialMediaIE/

python /content/gdrive/My\ Drive/SocialMediaIE/experiments/multitask_multidataset_experiment.py \
--task multimodal_ner broad_ner wnut17_ner ritter_ner yodie_ner ritter_chunk ud_pos ark_pos ptb_pos ritter_ccg \
--encoder-type bilstm --multi-task-mode stacked \
--weight-decay 0 --lr 1e-3 \
--model-dir /content/gdrive/My\ Drive/SocialMediaIE/data/models/all_multitask_stacked_l2_0_lr_0.001_no_neel_no_dimsum \
--clean-model-dir --cuda \
--dataset-paths-file /content/gdrive/My\ Drive/SocialMediaIE/experiments/all_dataset_paths.json


python /content/gdrive/My\ Drive/SocialMediaIE/experiments/multitask_multidataset_experiment.py \
--task multimodal_ner broad_ner wnut17_ner ritter_ner yodie_ner ritter_chunk ud_pos ark_pos ptb_pos ritter_ccg \
--encoder-type bilstm --multi-task-mode shared \
--weight-decay 1e-3 \
--model-dir /content/gdrive/My\ Drive/SocialMediaIE/data/models/all_multitask_shared_no_neel_no_dimsum \
--clean-model-dir --cuda \
--dataset-paths-file /content/gdrive/My\ Drive/SocialMediaIE/experiments/all_dataset_paths.json


python /content/gdrive/My\ Drive/SocialMediaIE/experiments/multitask_multidataset_experiment.py \
--task multimodal_ner broad_ner wnut17_ner ritter_ner yodie_ner \
--encoder-type bilstm \
--weight-decay 1e-3 \
--model-dir /content/gdrive/My\ Drive/SocialMediaIE/data/models/all_ner_bilstm_no_neel \
--clean-model-dir --cuda \
--dataset-paths-file /content/gdrive/My\ Drive/SocialMediaIE/experiments/all_dataset_paths.json


python /content/gdrive/My\ Drive/SocialMediaIE/experiments/multitask_multidataset_experiment.py \
--task ud_pos ark_pos ptb_pos \
--encoder-type bilstm \
--weight-decay 1e-3 \
--model-dir /content/gdrive/My\ Drive/SocialMediaIE/data/models/all_pos_bilstm_no_dimsum \
--clean-model-dir --cuda \
--dataset-paths-file /content/gdrive/My\ Drive/SocialMediaIE/experiments/all_dataset_paths.json



python /content/gdrive/My\ Drive/SocialMediaIE/experiments/multitask_multidataset_experiment.py \
--task multimodal_ner broad_ner wnut17_ner ritter_ner yodie_ner ritter_chunk ud_pos ark_pos ptb_pos ritter_ccg \
--encoder-type bilstm --multi-task-mode shared \
--weight-decay 0 --lr 1e-3 \
--model-dir /content/gdrive/My\ Drive/SocialMediaIE/data/models/all_multitask_shared_l2_0_lr_0.001_no_neel_no_dimsum \
--clean-model-dir --cuda \
--dataset-paths-file /content/gdrive/My\ Drive/SocialMediaIE/experiments/all_dataset_paths.json



python /content/gdrive/My\ Drive/SocialMediaIE/experiments/multitask_multidataset_experiment.py \
--task multimodal_ner broad_ner wnut17_ner ritter_ner yodie_ner ritter_chunk ud_pos ark_pos ptb_pos ritter_ccg \
--encoder-type bilstm --multi-task-mode stacked \
--weight-decay 1e-3 \
--model-dir /content/gdrive/My\ Drive/SocialMediaIE/data/models/all_multitask_stacked_no_neel_no_dimsum \
--clean-model-dir --cuda \
--dataset-paths-file /content/gdrive/My\ Drive/SocialMediaIE/experiments/all_dataset_paths.json