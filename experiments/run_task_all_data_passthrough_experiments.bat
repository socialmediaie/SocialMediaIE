REM NER MODELS
python multitask_multidataset_experiment.py ^
--task multimodal_ner broad_ner neel_ner wnut17_ner ritter_ner yodie_ner ^
--model-dir ..\data\models\all_ner  --clean-model-dir --cuda --dataset-paths-file ./all_dataset_paths.json --weight-decay 1e-3

REM CHUNKING MODELS
REM python multitask_multidataset_experiment.py --task ritter_chunk --model-dir ..\data\models\ritter_chunk --clean-model-dir --cuda --dataset-paths-file ./all_dataset_paths.json --weight-decay 1e-3

REM POS MODELS
python multitask_multidataset_experiment.py ^
--task ud_pos ark_pos ptb_pos ^
--model-dir ..\data\models\all_pos  --clean-model-dir --cuda --dataset-paths-file ./all_dataset_paths.json --weight-decay 1e-3

REM CCG SUPERSENSE MODELS
REM python multitask_multidataset_experiment.py --task ritter_ccg --model-dir ..\data\models\ritter_ccg --clean-model-dir --cuda --dataset-paths-file ./all_dataset_paths.json --weight-decay 1e-3
