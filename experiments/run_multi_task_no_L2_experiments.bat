REM Multi task shared encoder stacked_self_attention
python multitask_multidataset_experiment.py ^
--task multimodal_ner broad_ner neel_ner wnut17_ner ritter_ner yodie_ner ritter_chunk ud_pos ark_pos ptb_pos ritter_ccg ^
--encoder-type stacked_self_attention --multi-task-mode shared ^
--weight-decay 0 --lr 1e-3 ^
--model-dir ..\data\models\all_multitask_shared_ssa_l2_0_lr_0.001 --clean-model-dir --cuda --dataset-paths-file ./all_dataset_paths.json

REM Multi task shared encoder stacked_self_attention default decay and LR
python multitask_multidataset_experiment.py ^
--task multimodal_ner broad_ner neel_ner wnut17_ner ritter_ner yodie_ner ritter_chunk ud_pos ark_pos ptb_pos ritter_ccg ^
--encoder-type stacked_self_attention --multi-task-mode shared ^
--weight-decay 1e-3 ^
--model-dir ..\data\models\all_multitask_shared_ssa --clean-model-dir --cuda --dataset-paths-file ./all_dataset_paths.json

REM Multi task shared encoder shared
python multitask_multidataset_experiment.py ^
--task multimodal_ner broad_ner neel_ner wnut17_ner ritter_ner yodie_ner ritter_chunk ud_pos ark_pos ptb_pos ritter_ccg ^
--encoder-type bilstm --multi-task-mode shared ^
--weight-decay 0 --lr 1e-3 ^
--model-dir ..\data\models\all_multitask_shared_l2_0_lr_0.001 --clean-model-dir --cuda --dataset-paths-file ./all_dataset_paths.json

REM Multi task shared encoder self_attention
python multitask_multidataset_experiment.py ^
--task multimodal_ner broad_ner neel_ner wnut17_ner ritter_ner yodie_ner ritter_chunk ud_pos ark_pos ptb_pos ritter_ccg ^
--encoder-type self_attention --multi-task-mode shared ^
--weight-decay 0 --lr 1e-3 ^
--model-dir ..\data\models\all_multitask_shared_sa_l2_0_lr_0.001 --clean-model-dir --cuda --dataset-paths-file ./all_dataset_paths.json

REM Multi task mode stacked
python multitask_multidataset_experiment.py ^
--task multimodal_ner broad_ner neel_ner wnut17_ner ritter_ner yodie_ner ritter_chunk ud_pos ark_pos ptb_pos ritter_ccg ^
--encoder-type bilstm --multi-task-mode stacked ^
--weight-decay 0 --lr 1e-3 ^
--model-dir ..\data\models\all_multitask_stacked_l2_0_lr_0.001 --clean-model-dir --cuda --dataset-paths-file ./all_dataset_paths.json