REM NER MODELS
python multitask_multidataset_experiment.py ^
--task multimodal_ner broad_ner neel_ner wnut17_ner ritter_ner yodie_ner ^
--encoder-type bilstm ^
--model-dir ..\data\models\all_ner_bilstm --clean-model-dir --cuda --dataset-paths-file ./all_dataset_paths.json --weight-decay 1e-3

REM POS MODELS
python multitask_multidataset_experiment.py ^
--task ud_pos ark_pos ptb_pos ^
--encoder-type bilstm ^
--model-dir ..\data\models\all_pos_bilstm --clean-model-dir --cuda --dataset-paths-file ./all_dataset_paths.json --weight-decay 1e-3