REM NER MODELS
python multitask_multidataset_experiment.py --task multimodal_ner --model-dir ..\data\models\multimodal_ner_l2_0_lr_1e-3  --clean-model-dir --cuda --dataset-paths-file ./all_dataset_paths.json --weight-decay 0 --lr 1e-3
python multitask_multidataset_experiment.py --task broad_ner --model-dir ..\data\models\broad_ner_l2_0_lr_1e-3  --clean-model-dir --cuda --dataset-paths-file ./all_dataset_paths.json --weight-decay 0 --lr 1e-3

python multitask_multidataset_experiment.py --task neel_ner --model-dir ..\data\models\neel_ner_l2_0_lr_1e-3 --clean-model-dir --cuda --dataset-paths-file ./all_dataset_paths.json --weight-decay 0 --lr 1e-3
python multitask_multidataset_experiment.py --task wnut17_ner --model-dir ..\data\models\wnut17_ner_l2_0_lr_1e-3 --clean-model-dir --cuda --dataset-paths-file ./all_dataset_paths.json --weight-decay 0 --lr 1e-3
python multitask_multidataset_experiment.py --task ritter_ner --model-dir ..\data\models\ritter_ner_l2_0_lr_1e-3 --clean-model-dir --cuda --dataset-paths-file ./all_dataset_paths.json --weight-decay 0 --lr 1e-3
python multitask_multidataset_experiment.py --task yodie_ner --model-dir ..\data\models\yodie_ner_l2_0_lr_1e-3 --clean-model-dir --cuda --dataset-paths-file ./all_dataset_paths.json --weight-decay 0 --lr 1e-3

REM CHUNKING MODELS
python multitask_multidataset_experiment.py --task ritter_chunk --model-dir ..\data\models\ritter_chunk_l2_0_lr_1e-3 --clean-model-dir --cuda --dataset-paths-file ./all_dataset_paths.json --weight-decay 0 --lr 1e-3

REM POS MODELS
python multitask_multidataset_experiment.py --task ud_pos --model-dir ..\data\models\ud_pos_l2_0_lr_1e-3 --clean-model-dir --cuda --dataset-paths-file ./all_dataset_paths.json --weight-decay 0 --lr 1e-3
python multitask_multidataset_experiment.py --task ark_pos --model-dir ..\data\models\ark_pos_l2_0_lr_1e-3 --clean-model-dir --cuda --dataset-paths-file ./all_dataset_paths.json --weight-decay 0 --lr 1e-3
python multitask_multidataset_experiment.py --task ptb_pos --model-dir ..\data\models\ptb_pos_l2_0_lr_1e-3 --clean-model-dir --cuda --dataset-paths-file ./all_dataset_paths.json --weight-decay 0 --lr 1e-3

REM CCG SUPERSENSE MODELS
python multitask_multidataset_experiment.py --task ritter_ccg --model-dir ..\data\models\ritter_ccg_l2_0_lr_1e-3 --clean-model-dir --cuda --dataset-paths-file ./all_dataset_paths.json --weight-decay 0 --lr 1e-3

REM All data single task models
REM NER MODELS
python multitask_multidataset_experiment.py ^
--task multimodal_ner broad_ner neel_ner wnut17_ner ritter_ner yodie_ner ^
--weight-decay 0 --lr 1e-3 ^
--model-dir ..\data\models\all_ner_l2_0_lr_1e-3  --clean-model-dir --cuda --dataset-paths-file ./all_dataset_paths.json

REM POS MODELS
python multitask_multidataset_experiment.py ^
--task ud_pos ark_pos ptb_pos ^
--weight-decay 0 --lr 1e-3 ^
--model-dir ..\data\models\all_pos_l2_0_lr_1e-3  --clean-model-dir --cuda --dataset-paths-file ./all_dataset_paths.json

REM NER MODELS
python multitask_multidataset_experiment.py ^
--task multimodal_ner broad_ner neel_ner wnut17_ner ritter_ner yodie_ner ^
--encoder-type bilstm ^
--weight-decay 0 --lr 1e-3 ^
--model-dir ..\data\models\all_ner_bilstm_l2_0_lr_1e-3 --clean-model-dir --cuda --dataset-paths-file ./all_dataset_paths.json

REM POS MODELS
python multitask_multidataset_experiment.py ^
--task ud_pos ark_pos ptb_pos ^
--encoder-type bilstm ^
--weight-decay 0 --lr 1e-3 ^
--model-dir ..\data\models\all_pos_bilstm_l2_0_lr_1e-3 --clean-model-dir --cuda --dataset-paths-file ./all_dataset_paths.json

REM NER MODELS
python multitask_multidataset_experiment.py ^
--task multimodal_ner broad_ner neel_ner wnut17_ner ritter_ner yodie_ner ^
--encoder-type self_attention ^
--weight-decay 0 --lr 1e-3 ^
--model-dir ..\data\models\all_ner_sa_l2_0_lr_1e-3 --clean-model-dir --cuda --dataset-paths-file ./all_dataset_paths.json

REM POS MODELS
python multitask_multidataset_experiment.py ^
--task ud_pos ark_pos ptb_pos ^
--encoder-type self_attention ^
--weight-decay 0 --lr 1e-3 ^
--model-dir ..\data\models\all_pos_sa_l2_0_lr_1e-3 --clean-model-dir --cuda --dataset-paths-file ./all_dataset_paths.json