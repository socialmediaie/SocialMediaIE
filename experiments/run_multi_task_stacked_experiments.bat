REM Multi task mode
python multitask_multidataset_experiment.py ^
--task multimodal_ner broad_ner neel_ner wnut17_ner ritter_ner yodie_ner ritter_chunk ud_pos ark_pos ptb_pos ritter_ccg ^
--encoder-type bilstm --multi-task-mode stacked ^
--model-dir ..\data\models\all_multitask_stacked --clean-model-dir --cuda --dataset-paths-file ./all_dataset_paths.json --weight-decay 1e-3