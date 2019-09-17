REM Multi task mode
python ..\SocialMediaIE\scripts\multitask_multidataset_classification_tagging.py ^
--task multimodal_ner broad_ner neel_ner wnut17_ner ritter_ner yodie_ner ritter_chunk ud_pos ark_pos ptb_pos ritter_ccg ^
founta_abusive waseem_abusive ^
sarcasm_uncertainity veridicality_uncertainity ^
semeval_sentiment clarin_sentiment politics_sentiment other_sentiment ^
--encoder-type bilstm --multi-task-mode stacked ^
--model-dir ..\data\models_classification_tagging\all_multitask_stacked --clean-model-dir --cuda ^
--dataset-paths-file all_classification_tagging_dataset_paths.json --weight-decay 1e-3 ^
--batch-size 8 --epochs 1

REM Multi task mode
python ..\SocialMediaIE\scripts\multitask_multidataset_classification_tagging.py ^
--task multimodal_ner broad_ner neel_ner wnut17_ner ritter_ner yodie_ner ritter_chunk ud_pos ark_pos ptb_pos ritter_ccg ^
founta_abusive waseem_abusive ^
sarcasm_uncertainity veridicality_uncertainity ^
semeval_sentiment clarin_sentiment politics_sentiment other_sentiment ^
--encoder-type bilstm --multi-task-mode shared ^
--model-dir ..\data\models_classification_tagging\all_multitask_stacked --clean-model-dir --cuda ^
--dataset-paths-file all_classification_tagging_dataset_paths.json --weight-decay 1e-3 ^
--batch-size 8 --epochs 1