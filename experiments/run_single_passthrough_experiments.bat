REM NER MODELS
REM python multitask_multidataset_experiment.py --task multimodal_ner --model-dir ..\data\models\multimodal_ner  --clean-model-dir --cuda --dataset-paths-file ./all_dataset_paths.json --weight-decay 1e-3
REM python multitask_multidataset_experiment.py --task broad_ner --model-dir ..\data\models\broad_ner  --clean-model-dir --cuda --dataset-paths-file ./all_dataset_paths.json --weight-decay 1e-3

python multitask_multidataset_experiment.py --task neel_ner --model-dir ..\data\models\neel_ner --clean-model-dir --cuda --dataset-paths-file ./all_dataset_paths.json --weight-decay 1e-3
python multitask_multidataset_experiment.py --task wnut17_ner --model-dir ..\data\models\wnut17_ner --clean-model-dir --cuda --dataset-paths-file ./all_dataset_paths.json --weight-decay 1e-3
python multitask_multidataset_experiment.py --task ritter_ner --model-dir ..\data\models\ritter_ner --clean-model-dir --cuda --dataset-paths-file ./all_dataset_paths.json --weight-decay 1e-3
python multitask_multidataset_experiment.py --task yodie_ner --model-dir ..\data\models\yodie_ner --clean-model-dir --cuda --dataset-paths-file ./all_dataset_paths.json --weight-decay 1e-3

REM CHUNKING MODELS
python multitask_multidataset_experiment.py --task ritter_chunk --model-dir ..\data\models\ritter_chunk --clean-model-dir --cuda --dataset-paths-file ./all_dataset_paths.json --weight-decay 1e-3

REM POS MODELS
python multitask_multidataset_experiment.py --task ud_pos --model-dir ..\data\models\ud_pos --clean-model-dir --cuda --dataset-paths-file ./all_dataset_paths.json --weight-decay 1e-3
python multitask_multidataset_experiment.py --task ark_pos --model-dir ..\data\models\ark_pos --clean-model-dir --cuda --dataset-paths-file ./all_dataset_paths.json --weight-decay 1e-3
python multitask_multidataset_experiment.py --task ptb_pos --model-dir ..\data\models\ptb_pos --clean-model-dir --cuda --dataset-paths-file ./all_dataset_paths.json --weight-decay 1e-3

REM CCG SUPERSENSE MODELS
python multitask_multidataset_experiment.py --task ritter_ccg --model-dir ..\data\models\ritter_ccg --clean-model-dir --cuda --dataset-paths-file ./all_dataset_paths.json --weight-decay 1e-3
