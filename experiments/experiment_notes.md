02/18/2019 22:13:37 - INFO - allennlp.training.trainer -                                       Training &  Validation
02/18/2019 22:13:37 - INFO - allennlp.training.trainer -   ritter_chunk_accuracy           &     0.766  &     0.767
02/18/2019 22:13:37 - INFO - allennlp.training.trainer -   ritter_chunk_precision-overall  &     0.673  &     0.711
02/18/2019 22:13:37 - INFO - allennlp.training.trainer -   ritter_chunk_f1-measure-overall &     0.673  &     0.685
02/18/2019 22:13:37 - INFO - allennlp.training.trainer -   ark_pos_accuracy3               &     0.808  &     0.801
02/18/2019 22:13:37 - INFO - allennlp.training.trainer -   ptb_pos_accuracy3               &     0.719  &     0.814
02/18/2019 22:13:37 - INFO - allennlp.training.trainer -   loss                            &    21.654  &    12.325
02/18/2019 22:13:37 - INFO - allennlp.training.trainer -   cpu_memory_MB                   &     0.000  &       N/A
02/18/2019 22:13:37 - INFO - allennlp.training.trainer -   ritter_chunk_recall-overall     &     0.674  &     0.661
02/18/2019 22:13:37 - INFO - allennlp.training.trainer -   ud_pos_accuracy                 &     0.365  &     0.769
02/18/2019 22:13:37 - INFO - allennlp.training.trainer -   ptb_pos_accuracy                &     0.702  &     0.806
02/18/2019 22:13:37 - INFO - allennlp.training.trainer -   ark_pos_accuracy                &     0.795  &     0.790
02/18/2019 22:13:37 - INFO - allennlp.training.trainer -   ud_pos_accuracy3                &     0.456  &     0.819
02/18/2019 22:13:37 - INFO - allennlp.training.trainer -   ritter_chunk_accuracy3          &     0.793  &     0.785





{
  "best_epoch": 4,
  "peak_cpu_memory_MB": 0,
  "training_duration": "00:50:03",
  "training_start_epoch": 0,
  "training_epochs": 9,
  "epoch": 9,
  "training_multimodal_ner_accuracy": 0.9454035874439461,
  "training_multimodal_ner_accuracy3": 0.9541738530527768,
  "training_multimodal_ner_precision-overall": 0.7573729863692689,
  "training_multimodal_ner_recall-overall": 0.647869408522366,
  "training_multimodal_ner_f1-measure-overall": 0.6983546617915408,
  "training_loss": 36.45110553866821,
  "training_cpu_memory_MB": 0.0,
  "validation_multimodal_ner_accuracy": 0.9337371739399184,
  "validation_multimodal_ner_accuracy3": 0.9409692174558042,
  "validation_multimodal_ner_precision-overall": 0.7044235924932976,
  "validation_multimodal_ner_recall-overall": 0.656875,
  "validation_multimodal_ner_f1-measure-overall": 0.6798188874514377,
  "validation_loss": 44.978739904978916,
  "best_validation_multimodal_ner_accuracy": 0.9322536778340957,
  "best_validation_multimodal_ner_accuracy3": 0.941463716157745,
  "best_validation_multimodal_ner_precision-overall": 0.7494270435446906,
  "best_validation_multimodal_ner_recall-overall": 0.613125,
  "best_validation_multimodal_ner_f1-measure-overall": 0.6744585768304766,
  "best_validation_loss": 42.05576551528204
}


# Experiment run results


## Multimodal NER
Namespace(batch_size=16, clean_model_dir=True, cuda=True, dataset_paths_file='./all_dataset_paths.json', dropout=0.5, epochs=10, hidden_dim=50, lr=0.01, model_dir='../data/models/websci_mt/', patience=3, proj_dim=100, task=['multimodal_ner'], weight_decay=0.001)

1450it [00:00, 5174.11it/s]
multimodal_ner - ../data/processed/NER/MSM2013/test.conll: 23it [00:21,  1.18it/s]
03/05/2019 02:33:00 AM - SocialMediaIE.evaluation.evaluate_model - INFO - multimodal_ner_accuracy: 0.97198      multimodal_ner_accuracy3: 0.97924       multimodal_ner_precision-overall: 0.83372       multimodal_ner_recall-overall: 0.74014 multimodal_ner_f1-measure-overall: 0.78414
3257it [00:00, 9800.70it/s]
multimodal_ner - ../data/processed/NER/MultiModal/test.conll: 51it [00:38,  1.36it/s]
03/05/2019 02:33:39 AM - SocialMediaIE.evaluation.evaluate_model - INFO - multimodal_ner_accuracy: 0.93457      multimodal_ner_accuracy3: 0.94432       multimodal_ner_precision-overall: 0.78942       multimodal_ner_recall-overall: 0.60959 multimodal_ner_f1-measure-overall: 0.68795

Test F1 for MultiModal close to results reported in paper https://aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16432/16127
Test F1 for MSM2013 is much better than of best system in http://ceur-ws.org/Vol-1019/msm2013-challenge-report.pdf
Test F1 for MSM2013 is better than those in https://www.sciencedirect.com/science/article/pii/S088523081630002X

## Broad NER
python multitask_multidataset_experiment.py --task broad_ner --clean-model-dir --cuda --dataset-paths-file ./all_dataset_paths.json --model-dir ..\data\models\broad_ner --weight-decay 1e-3

broad_ner - ../data/processed/NER/BROAD/test.conll: 44it [00:37,  1.10it/s]
03/05/2019 04:33:06 AM - SocialMediaIE.evaluation.evaluate_model - INFO - broad_ner_accuracy: 0.94635   broad_ner_accuracy3: 0.97400    broad_ner_precision-overall: 0.75061    broad_ner_recall-overall: 0.63248       broad_ner_f1-measure-overall: 0.68650
5369it [00:00, 12045.62it/s]
broad_ner - ../data/processed/NER/Finin/test.conll: 84it [01:10,  1.16it/s]
03/05/2019 04:34:17 AM - SocialMediaIE.evaluation.evaluate_model - INFO - broad_ner_accuracy: 0.96202   broad_ner_accuracy3: 0.97006    broad_ner_precision-overall: 0.46829    broad_ner_recall-overall: 0.54967       broad_ner_f1-measure-overall: 0.50573
1545it [00:00, 11691.48it/s]
broad_ner - ../data/processed/NER/Hege/test.conll: 25it [00:18,  1.60it/s]
03/05/2019 04:34:36 AM - SocialMediaIE.evaluation.evaluate_model - INFO - broad_ner_accuracy: 0.97256   broad_ner_accuracy3: 0.98456    broad_ner_precision-overall: 0.90861    broad_ner_recall-overall: 0.82544       broad_ner_f1-measure-overall: 0.86503



## NEEL NER
*only 1 epoch*

neel_ner - ../data/processed/NER/NEEL2016/test.conll: 84it [00:48,  2.02it/s]
03/05/2019 05:00:38 AM - SocialMediaIE.evaluation.evaluate_model - INFO - neel_ner_accuracy: 0.88332    neel_ner_accuracy3: 0.89010     neel_ner_precision-overall: 0.04087     neel_ner_recall-overall: 0.19406        neel_ner_f1-measure-overall: 0.06751


*Final*
2663it [00:00, 10608.62it/s]
neel_ner - ../data/processed/NER/NEEL2016/test.conll: 84it [00:47,  2.13it/s]
03/05/2019 05:38:29 AM - SocialMediaIE.evaluation.evaluate_model - INFO - neel_ner_accuracy: 0.88521    neel_ner_accuracy3: 0.89145     neel_ner_precision-overall: 0.04587     neel_ner_recall-overall: 0.21005        neel_ner_f1-measure-overall: 0.07530

## WNUT17 NER
*only 1 epoch*

1287it [00:00, 8801.49it/s]
wnut17_ner - ../data/processed/NER/WNUT2017/test.conll: 41it [00:36,  1.67it/s]
03/05/2019 05:01:39 AM - SocialMediaIE.evaluation.evaluate_model - INFO - wnut17_ner_accuracy: 0.94516  wnut17_ner_accuracy3: 0.94956   wnut17_ner_precision-overall: 0.63939   wnut17_ner_recall-overall: 0.23946      wnut17_ner_f1-measure-overall: 0.34843

*Final*
1287it [00:00, 17405.04it/s]
wnut17_ner - ../data/processed/NER/WNUT2017/test.conll: 41it [00:34,  1.98it/s]
03/05/2019 06:05:41 AM - SocialMediaIE.evaluation.evaluate_model - INFO - wnut17_ner_accuracy: 0.95007  wnut17_ner_accuracy3: 0.95366   wnut17_ner_precision-overall: 0.71599   wnut17_ner_recall-overall: 0.28736      wnut17_ner_f1-measure-overall: 0.41012

## Ritter NER
*only 1 epoch*

254it [00:00, 8562.69it/s]
ritter_ner - ../data/processed/NER/Ritter/test.conll: 8it [00:06,  1.35it/s]
03/05/2019 05:02:10 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ritter_ner_accuracy: 0.97561  ritter_ner_accuracy3: 0.97805   ritter_ner_precision-overall: 0.82143   ritter_ner_recall-overall: 0.50365      ritter_ner_f1-measure-overall: 0.62443
3850it [00:00, 8209.68it/s]
ritter_ner - ../data/processed/NER/WNUT2016/test.conll: 121it [01:26,  1.97it/s]
03/05/2019 05:03:37 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ritter_ner_accuracy: 0.92728  ritter_ner_accuracy3: 0.93565   ritter_ner_precision-overall: 0.52990   ritter_ner_recall-overall: 0.33170      ritter_ner_f1-measure-overall: 0.40800


*Final*
254it [00:00, 10865.52it/s]
ritter_ner - ../data/processed/NER/Ritter/test.conll: 8it [00:05,  1.65it/s]
03/05/2019 06:37:27 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ritter_ner_accuracy: 0.98090  ritter_ner_accuracy3: 0.98171   ritter_ner_precision-overall: 0.78261   ritter_ner_recall-overall: 0.65693      ritter_ner_f1-measure-overall: 0.71429
3850it [00:00, 7600.42it/s]
ritter_ner - ../data/processed/NER/WNUT2016/test.conll: 121it [01:07,  2.02it/s]
03/05/2019 06:38:35 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ritter_ner_accuracy: 0.92961  ritter_ner_accuracy3: 0.93603   ritter_ner_precision-overall: 0.51033   ritter_ner_recall-overall: 0.40541      ritter_ner_f1-measure-overall: 0.45186

## YODIE NER
*only 1 epoch*

397it [00:00, 9560.48it/s]
yodie_ner - ../data/processed/NER/YODIE/test.conll: 13it [00:12,  1.21s/it]
03/05/2019 05:04:15 AM - SocialMediaIE.evaluation.evaluate_model - INFO - yodie_ner_accuracy: 0.92903   yodie_ner_accuracy3: 0.93526    yodie_ner_precision-overall: 0.48317    yodie_ner_recall-overall: 0.37361       yodie_ner_f1-measure-overall: 0.42138

*Final*

397it [00:00, 12111.16it/s]
yodie_ner - ../data/processed/NER/YODIE/test.conll: 13it [00:07,  1.82it/s]
03/05/2019 06:43:36 AM - SocialMediaIE.evaluation.evaluate_model - INFO - yodie_ner_accuracy: 0.94422   yodie_ner_accuracy3: 0.95020    yodie_ner_precision-overall: 0.65340    yodie_ner_recall-overall: 0.51859       yodie_ner_f1-measure-overall: 0.57824


# Chunking

## Ritter Chunk
*only 1 epoch*

119it [00:00, 5424.24it/s]
ritter_chunk - ../data/processed/CHUNKING/Ritter/test.conll: 4it [00:04,  1.19s/it]
03/05/2019 05:14:26 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ritter_chunk_accuracy: 0.85758        ritter_chunk_accuracy3: 0.87965 ritter_chunk_precision-overall: 0.75910 ritter_chunk_recall-overall: 0.78400           ritter_chunk_f1-measure-overall: 0.77135

*Final*

119it [00:00, 13624.56it/s]
ritter_chunk - ../data/processed/CHUNKING/Ritter/test.conll: 4it [00:02,  1.52it/s]
03/05/2019 06:48:10 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ritter_chunk_accuracy: 0.89481        ritter_chunk_accuracy3: 0.91039 ritter_chunk_precision-overall: 0.83320 ritter_chunk_recall-overall: 0.83920           ritter_chunk_f1-measure-overall: 0.83619

# POS

## UD POS

*only 1 epoch*

1201it [00:00, 7319.03it/s]
ud_pos - ../data/processed/POS/Tweetbankv2/test.conll: 38it [00:24,  1.71it/s]
03/05/2019 05:15:15 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ud_pos_accuracy: 0.88217      ud_pos_accuracy3: 0.91689
1000it [00:00, 9332.89it/s]
ud_pos - ../data/processed/POS/DiMSUM2016/test.conll: 32it [00:32,  1.32it/s]
03/05/2019 05:15:47 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ud_pos_accuracy: 0.17297      ud_pos_accuracy3: 0.26036
250it [00:00, 11982.49it/s]
ud_pos - ../data/processed/POS/Foster/test.conll: 8it [00:03,  2.04it/s]
03/05/2019 05:15:51 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ud_pos_accuracy: 0.65259      ud_pos_accuracy3: 0.71102
1318it [00:00, 10906.35it/s]
ud_pos - ../data/processed/POS/lowlands/test.conll: 42it [00:21,  2.20it/s]
03/05/2019 05:16:13 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ud_pos_accuracy: 0.60619      ud_pos_accuracy3: 0.65469


*Final*

1201it [00:00, 15131.84it/s]
ud_pos - ../data/processed/POS/Tweetbankv2/test.conll: 38it [00:21,  1.93it/s]
03/05/2019 07:20:09 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ud_pos_accuracy: 0.89605      ud_pos_accuracy3: 0.92239
1000it [00:00, 19368.13it/s]
ud_pos - ../data/processed/POS/DiMSUM2016/test.conll: 32it [00:25,  1.31it/s]
03/05/2019 07:20:35 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ud_pos_accuracy: 0.17061      ud_pos_accuracy3: 0.25085
250it [00:00, 21743.86it/s]
ud_pos - ../data/processed/POS/Foster/test.conll: 8it [00:03,  2.17it/s]
03/05/2019 07:20:39 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ud_pos_accuracy: 0.63675      ud_pos_accuracy3: 0.69623
1318it [00:00, 20663.91it/s]
ud_pos - ../data/processed/POS/lowlands/test.conll: 42it [00:21,  2.29it/s]
03/05/2019 07:21:01 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ud_pos_accuracy: 0.60276      ud_pos_accuracy3: 0.65298

## Ark POS

*only 1 epoch*


ark_pos - ../data/processed/POS/Owoputi/test.conll: 16it [00:10,  1.95it/s]
03/05/2019 05:16:54 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ark_pos_accuracy: 0.88968     ark_pos_accuracy3: 0.89583

*Final*
500it [00:00, 5895.18it/s]
ark_pos - ../data/processed/POS/Owoputi/test.conll: 16it [00:08,  1.99it/s]
03/05/2019 07:28:25 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ark_pos_accuracy: 0.90702     ark_pos_accuracy3: 0.91359

## PTB POS

*only 1 epoch*


ptb_pos - ../data/processed/POS/TwitIE/test.conll: 8it [00:05,  1.60it/s]
03/05/2019 05:17:25 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ptb_pos_accuracy: 0.87540     ptb_pos_accuracy3: 0.87927
84it [00:00, 6282.26it/s]

ptb_pos - ../data/processed/POS/Ritter/test.conll: 0it [00:00, ?it/s]03/05/2019 05:17:25 AM - allennlp.data.vocabulary - ERROR - Namespace: ptb_pos
03/05/2019 05:17:25 AM - allennlp.data.vocabulary - ERROR - Token: VPP

**TODO: VPP should be VBP** - Fix the data generation code. Currently manually fixed. 

*only 1 epoch* after fixing
250it [00:00, 19381.46it/s]
ptb_pos - ../data/processed/POS/TwitIE/test.conll: 8it [00:05,  1.62it/s]
03/05/2019 05:20:56 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ptb_pos_accuracy: 0.87540     ptb_pos_accuracy3: 0.87927
84it [00:00, 8064.68it/s]
ptb_pos - ../data/processed/POS/Ritter/test.conll: 3it [00:01,  1.87it/s]
03/05/2019 05:20:58 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ptb_pos_accuracy: 0.87953     ptb_pos_accuracy3: 0.88322


*Final*
250it [00:00, 19386.48it/s]
ptb_pos - ../data/processed/POS/TwitIE/test.conll: 8it [00:04,  2.00it/s]
03/05/2019 07:33:31 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ptb_pos_accuracy: 0.89616     ptb_pos_accuracy3: 0.90004
84it [00:00, 10865.74it/s]
ptb_pos - ../data/processed/POS/Ritter/test.conll: 3it [00:01,  1.66it/s]
03/05/2019 07:33:32 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ptb_pos_accuracy: 0.90227     ptb_pos_accuracy3: 0.90412


# CCG

## Ritter CCG
*Final*

118it [00:00, 15860.02it/s]
ritter_ccg - ../data/processed/SUPERSENSE/Ritter/test.conll: 4it [00:02,  1.77it/s]
03/05/2019 10:34:43 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ritter_ccg_accuracy: 0.82409  ritter_ccg_accuracy3: 0.82628   ritter_ccg_precision-overall: 0.61019   ritter_ccg_recall-overall: 0.58520      ritter_ccg_f1-measure-overall: 0.59744
200it [00:00, 17475.54it/s]
ritter_ccg - ../data/processed/SUPERSENSE/Johannsen2014/test.conll: 7it [00:03,  2.11it/s]
03/05/2019 10:34:47 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ritter_ccg_accuracy: 0.64132  ritter_ccg_accuracy3: 0.64621   ritter_ccg_precision-overall: 0.36609   ritter_ccg_recall-overall: 0.38447      ritter_ccg_f1-measure-overall: 0.37506



# All NER

## Passthrough

*After 1st epoch*

03/05/2019 12:14:46 PM - allennlp.training.tensorboard_writer - INFO -                                       Training |  Validation
03/05/2019 12:14:46 PM - allennlp.training.tensorboard_writer - INFO - yodie_ner_precision-overall       |     0.255  |     0.553
03/05/2019 12:14:46 PM - allennlp.training.tensorboard_writer - INFO - neel_ner_precision-overall        |     0.463  |     0.667
03/05/2019 12:14:46 PM - allennlp.training.tensorboard_writer - INFO - wnut17_ner_accuracy3              |     0.965  |     0.934
03/05/2019 12:14:46 PM - allennlp.training.tensorboard_writer - INFO - yodie_ner_recall-overall          |     0.263  |     0.308
03/05/2019 12:14:46 PM - allennlp.training.tensorboard_writer - INFO - yodie_ner_f1-measure-overall      |     0.259  |     0.395
03/05/2019 12:14:46 PM - allennlp.training.tensorboard_writer - INFO - multimodal_ner_recall-overall     |     0.577  |     0.403
03/05/2019 12:14:46 PM - allennlp.training.tensorboard_writer - INFO - ritter_ner_accuracy               |     0.958  |     0.942
03/05/2019 12:14:46 PM - allennlp.training.tensorboard_writer - INFO - broad_ner_precision-overall       |     0.698  |     0.814
03/05/2019 12:14:46 PM - allennlp.training.tensorboard_writer - INFO - multimodal_ner_precision-overall  |     0.684  |     0.788
03/05/2019 12:14:46 PM - allennlp.training.tensorboard_writer - INFO - neel_ner_recall-overall           |     0.421  |     0.034
03/05/2019 12:14:46 PM - allennlp.training.tensorboard_writer - INFO - wnut17_ner_precision-overall      |     0.525  |     0.750
03/05/2019 12:14:46 PM - allennlp.training.tensorboard_writer - INFO - wnut17_ner_accuracy               |     0.960  |     0.926
03/05/2019 12:14:46 PM - allennlp.training.tensorboard_writer - INFO - multimodal_ner_accuracy           |     0.935  |     0.907
03/05/2019 12:14:46 PM - allennlp.training.tensorboard_writer - INFO - multimodal_ner_f1-measure-overall |     0.626  |     0.533
03/05/2019 12:14:46 PM - allennlp.training.tensorboard_writer - INFO - ritter_ner_precision-overall      |     0.495  |     0.642
03/05/2019 12:14:46 PM - allennlp.training.tensorboard_writer - INFO - broad_ner_f1-measure-overall      |     0.593  |     0.636
03/05/2019 12:14:46 PM - allennlp.training.tensorboard_writer - INFO - cpu_memory_MB                     |     0.000  |       N/A
03/05/2019 12:14:46 PM - allennlp.training.tensorboard_writer - INFO - broad_ner_recall-overall          |     0.516  |     0.521
03/05/2019 12:14:46 PM - allennlp.training.tensorboard_writer - INFO - neel_ner_accuracy                 |     0.931  |     0.786
03/05/2019 12:14:46 PM - allennlp.training.tensorboard_writer - INFO - ritter_ner_recall-overall         |     0.388  |     0.146
03/05/2019 12:14:46 PM - allennlp.training.tensorboard_writer - INFO - broad_ner_accuracy                |     0.947  |     0.937
03/05/2019 12:14:46 PM - allennlp.training.tensorboard_writer - INFO - broad_ner_accuracy3               |     0.974  |     0.975
03/05/2019 12:14:46 PM - allennlp.training.tensorboard_writer - INFO - wnut17_ner_recall-overall         |     0.387  |     0.072
03/05/2019 12:14:46 PM - allennlp.training.tensorboard_writer - INFO - ritter_ner_f1-measure-overall     |     0.435  |     0.238
03/05/2019 12:14:46 PM - allennlp.training.tensorboard_writer - INFO - ritter_ner_accuracy3              |     0.961  |     0.946
03/05/2019 12:14:46 PM - allennlp.training.tensorboard_writer - INFO - yodie_ner_accuracy                |     0.891  |     0.931
03/05/2019 12:14:46 PM - allennlp.training.tensorboard_writer - INFO - neel_ner_accuracy3                |     0.941  |     0.862
03/05/2019 12:14:46 PM - allennlp.training.tensorboard_writer - INFO - wnut17_ner_f1-measure-overall     |     0.446  |     0.131
03/05/2019 12:14:46 PM - allennlp.training.tensorboard_writer - INFO - loss                              |     3.235  |     8.052
03/05/2019 12:14:46 PM - allennlp.training.tensorboard_writer - INFO - neel_ner_f1-measure-overall       |     0.441  |     0.065
03/05/2019 12:14:46 PM - allennlp.training.tensorboard_writer - INFO - multimodal_ner_accuracy3          |     0.946  |     0.931
03/05/2019 12:14:46 PM - allennlp.training.tensorboard_writer - INFO - yodie_ner_accuracy3               |     0.896  |     0.936
03/05/2019 12:14:47 PM - allennlp.training.checkpointer - INFO - Best validation performance so far. Copying weights to '..\data\models\all_ner/best.th'.


*Final*

1450it [00:00, 8305.14it/s]
multimodal_ner - ../data/processed/NER/MSM2013/test.conll: 46it [02:18,  2.66s/it]
03/06/2019 12:47:40 AM - SocialMediaIE.evaluation.evaluate_model - INFO - multimodal_ner_accuracy: 0.96975      multimodal_ner_accuracy3: 0.97690       multimodal_ner_precision-overall: 0.81156       multimodal_ner_recall-overall: 0.73537 multimodal_ner_f1-measure-overall: 0.77159
3257it [00:03, 1061.96it/s]
multimodal_ner - ../data/processed/NER/MultiModal/test.conll: 102it [04:26,  2.37s/it]
03/06/2019 12:52:09 AM - SocialMediaIE.evaluation.evaluate_model - INFO - multimodal_ner_accuracy: 0.93302      multimodal_ner_accuracy3: 0.94112       multimodal_ner_precision-overall: 0.77693       multimodal_ner_recall-overall: 0.60844 multimodal_ner_f1-measure-overall: 0.68244
2802it [00:00, 18464.77it/s]
broad_ner - ../data/processed/NER/BROAD/test.conll: 88it [04:02,  2.85s/it]
03/06/2019 12:56:12 AM - SocialMediaIE.evaluation.evaluate_model - INFO - broad_ner_accuracy: 0.94592   broad_ner_accuracy3: 0.97219    broad_ner_precision-overall: 0.74027    broad_ner_recall-overall: 0.64044       broad_ner_f1-measure-overall: 0.68675
5369it [00:00, 18710.29it/s]
broad_ner - ../data/processed/NER/Finin/test.conll: 168it [08:08,  3.04s/it]
03/06/2019 01:04:21 AM - SocialMediaIE.evaluation.evaluate_model - INFO - broad_ner_accuracy: 0.96309   broad_ner_accuracy3: 0.97195    broad_ner_precision-overall: 0.47936    broad_ner_recall-overall: 0.52058       broad_ner_f1-measure-overall: 0.49912
1545it [00:00, 17349.84it/s]
broad_ner - ../data/processed/NER/Hege/test.conll: 49it [02:09,  2.48s/it]
03/06/2019 01:06:31 AM - SocialMediaIE.evaluation.evaluate_model - INFO - broad_ner_accuracy: 0.97193   broad_ner_accuracy3: 0.98500    broad_ner_precision-overall: 0.91521    broad_ner_recall-overall: 0.81001       broad_ner_f1-measure-overall: 0.85940
2663it [00:00, 9726.58it/s]
neel_ner - ../data/processed/NER/NEEL2016/test.conll: 84it [04:30,  2.49s/it]
03/06/2019 01:11:02 AM - SocialMediaIE.evaluation.evaluate_model - INFO - neel_ner_accuracy: 0.91846    neel_ner_accuracy3: 0.92516     neel_ner_precision-overall: 0.04917     neel_ner_recall-overall: 0.12785        neel_ner_f1-measure-overall: 0.07102
1287it [00:00, 15418.69it/s]
wnut17_ner - ../data/processed/NER/WNUT2017/test.conll: 41it [02:49,  2.51s/it]
03/06/2019 01:13:51 AM - SocialMediaIE.evaluation.evaluate_model - INFO - wnut17_ner_accuracy: 0.94960  wnut17_ner_accuracy3: 0.95336   wnut17_ner_precision-overall: 0.68842   wnut17_ner_recall-overall: 0.31322      wnut17_ner_f1-measure-overall: 0.43055
254it [00:00, 9346.27it/s]
ritter_ner - ../data/processed/NER/Ritter/test.conll: 8it [00:22,  2.77s/it]
03/06/2019 01:14:14 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ritter_ner_accuracy: 0.97155  ritter_ner_accuracy3: 0.97501   ritter_ner_precision-overall: 0.73418   ritter_ner_recall-overall: 0.42336      ritter_ner_f1-measure-overall: 0.53704
3850it [00:00, 10595.25it/s]
ritter_ner - ../data/processed/NER/WNUT2016/test.conll: 121it [07:14,  2.40s/it]
03/06/2019 01:21:28 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ritter_ner_accuracy: 0.92780  ritter_ner_accuracy3: 0.93712   ritter_ner_precision-overall: 0.54517   ritter_ner_recall-overall: 0.35272      ritter_ner_f1-measure-overall: 0.42832
397it [00:00, 9189.05it/s]
yodie_ner - ../data/processed/NER/YODIE/test.conll: 13it [00:36,  2.58s/it]
03/06/2019 01:22:05 AM - SocialMediaIE.evaluation.evaluate_model - INFO - yodie_ner_accuracy: 0.92928   yodie_ner_accuracy3: 0.93588    yodie_ner_precision-overall: 0.54496    yodie_ner_recall-overall: 0.37175       yodie_ner_f1-measure-overall: 0.44199


**NOTE** - We experience catastrophic forgetting in the network. When training across diverse datasets. This is also amplified by the fact that smaller datasets are exhausted in the epoch earlier compared to larger datasets. We need a shared learned contextual layer. 

**NOTE** - One bottlenk in the runtime of the model is that it tries to run the CRF for each task which is order N^2 in length of sequence. Hence the CRF computation time becomes k times for k tasks. We don't need this for training and evaluation. We only need this for final inference.
One way to achieve this would be to only do CRF for tasks which are needed. Maybe some kind of flag in the model. **DONE**


## Self attention

1450it [00:00, 13226.18it/s]
multimodal_ner - ../data/processed/NER/MSM2013/test.conll: 46it [00:27,  1.91it/s]
03/06/2019 07:37:32 AM - SocialMediaIE.evaluation.evaluate_model - INFO - multimodal_ner_accuracy: 0.96700      multimodal_ner_accuracy3: 0.97229       multimodal_ner_precision-overall: 0.73552       multimodal_ner_recall-overall: 0.77755 multimodal_ner_f1-measure-overall: 0.75595
3257it [00:01, 3109.56it/s]
multimodal_ner - ../data/processed/NER/MultiModal/test.conll: 102it [00:50,  2.16it/s]
03/06/2019 07:38:24 AM - SocialMediaIE.evaluation.evaluate_model - INFO - multimodal_ner_accuracy: 0.93495      multimodal_ner_accuracy3: 0.94209       multimodal_ner_precision-overall: 0.72839       multimodal_ner_recall-overall: 0.64640 multimodal_ner_f1-measure-overall: 0.68495
2802it [00:00, 14161.32it/s]
broad_ner - ../data/processed/NER/BROAD/test.conll: 88it [00:47,  1.87it/s]
03/06/2019 07:39:11 AM - SocialMediaIE.evaluation.evaluate_model - INFO - broad_ner_accuracy: 0.94181   broad_ner_accuracy3: 0.97148    broad_ner_precision-overall: 0.74074    broad_ner_recall-overall: 0.58676       broad_ner_f1-measure-overall: 0.65482
5369it [00:00, 19163.48it/s]
broad_ner - ../data/processed/NER/Finin/test.conll: 168it [01:35,  1.66it/s]
03/06/2019 07:40:47 AM - SocialMediaIE.evaluation.evaluate_model - INFO - broad_ner_accuracy: 0.96275   broad_ner_accuracy3: 0.97226    broad_ner_precision-overall: 0.48674    broad_ner_recall-overall: 0.50511       broad_ner_f1-measure-overall: 0.49576
1545it [00:00, 2458.25it/s]
broad_ner - ../data/processed/NER/Hege/test.conll: 49it [00:24,  2.16it/s]
03/06/2019 07:41:12 AM - SocialMediaIE.evaluation.evaluate_model - INFO - broad_ner_accuracy: 0.97150   broad_ner_accuracy3: 0.98621    broad_ner_precision-overall: 0.93599    broad_ner_recall-overall: 0.78606       broad_ner_f1-measure-overall: 0.85450
2663it [00:00, 5767.08it/s]
neel_ner - ../data/processed/NER/NEEL2016/test.conll: 84it [00:45,  2.11it/s]
03/06/2019 07:41:59 AM - SocialMediaIE.evaluation.evaluate_model - INFO - neel_ner_accuracy: 0.92979    neel_ner_accuracy3: 0.93647     neel_ner_precision-overall: 0.04474     neel_ner_recall-overall: 0.08447        neel_ner_f1-measure-overall: 0.05850
1287it [00:00, 15958.63it/s]
wnut17_ner - ../data/processed/NER/WNUT2017/test.conll: 41it [00:33,  2.01it/s]
03/06/2019 07:42:33 AM - SocialMediaIE.evaluation.evaluate_model - INFO - wnut17_ner_accuracy: 0.94140  wnut17_ner_accuracy3: 0.94494   wnut17_ner_precision-overall: 0.51327   wnut17_ner_recall-overall: 0.27778      wnut17_ner_f1-measure-overall: 0.36047
254it [00:00, 11890.90it/s]
ritter_ner - ../data/processed/NER/Ritter/test.conll: 8it [00:04,  1.70it/s]
03/06/2019 07:42:37 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ritter_ner_accuracy: 0.96972  ritter_ner_accuracy3: 0.97236   ritter_ner_precision-overall: 0.64130   ritter_ner_recall-overall: 0.43066      ritter_ner_f1-measure-overall: 0.51528
3850it [00:00, 11498.75it/s]
ritter_ner - ../data/processed/NER/WNUT2016/test.conll: 121it [01:02,  2.15it/s]
03/06/2019 07:43:40 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ritter_ner_accuracy: 0.92500  ritter_ner_accuracy3: 0.93424   ritter_ner_precision-overall: 0.46245   ritter_ner_recall-overall: 0.33861      ritter_ner_f1-measure-overall: 0.39096
397it [00:00, 11640.58it/s]
yodie_ner - ../data/processed/NER/YODIE/test.conll: 13it [00:07,  1.77it/s]
03/06/2019 07:43:48 AM - SocialMediaIE.evaluation.evaluate_model - INFO - yodie_ner_accuracy: 0.92356   yodie_ner_accuracy3: 0.92991    yodie_ner_precision-overall: 0.56225    yodie_ner_recall-overall: 0.26022       yodie_ner_f1-measure-overall: 0.35578


## BiLSTM

1450it [00:00, 16246.79it/s]
multimodal_ner - ../data/processed/NER/MSM2013/test.conll: 46it [00:28,  1.80it/s]
03/06/2019 01:22:04 PM - SocialMediaIE.evaluation.evaluate_model - INFO - multimodal_ner_accuracy: 0.97157      multimodal_ner_accuracy3: 0.97714       multimodal_ner_precision-overall: 0.81424       multimodal_ner_recall-overall: 0.75442 multimodal_ner_f1-measure-overall: 0.78319
3257it [00:03, 1074.21it/s]
multimodal_ner - ../data/processed/NER/MultiModal/test.conll: 102it [00:51,  2.11it/s]
03/06/2019 01:22:59 PM - SocialMediaIE.evaluation.evaluate_model - INFO - multimodal_ner_accuracy: 0.93009      multimodal_ner_accuracy3: 0.93868       multimodal_ner_precision-overall: 0.76346       multimodal_ner_recall-overall: 0.60652 multimodal_ner_f1-measure-overall: 0.67600
2802it [00:00, 20166.79it/s]
broad_ner - ../data/processed/NER/BROAD/test.conll: 88it [00:48,  1.83it/s]
03/06/2019 01:23:48 PM - SocialMediaIE.evaluation.evaluate_model - INFO - broad_ner_accuracy: 0.95186   broad_ner_accuracy3: 0.97467    broad_ner_precision-overall: 0.75252    broad_ner_recall-overall: 0.69502       broad_ner_f1-measure-overall: 0.72263
5369it [00:00, 19420.62it/s]
broad_ner - ../data/processed/NER/Finin/test.conll: 168it [01:35,  1.56it/s]
03/06/2019 01:25:24 PM - SocialMediaIE.evaluation.evaluate_model - INFO - broad_ner_accuracy: 0.96530   broad_ner_accuracy3: 0.97335    broad_ner_precision-overall: 0.50409    broad_ner_recall-overall: 0.56488       broad_ner_f1-measure-overall: 0.53276
1545it [00:00, 2427.98it/s]
broad_ner - ../data/processed/NER/Hege/test.conll: 49it [00:24,  2.14it/s]
03/06/2019 01:25:49 PM - SocialMediaIE.evaluation.evaluate_model - INFO - broad_ner_accuracy: 0.97575   broad_ner_accuracy3: 0.98708    broad_ner_precision-overall: 0.93063    broad_ner_recall-overall: 0.84247       broad_ner_f1-measure-overall: 0.88436
2663it [00:00, 18408.56it/s]
neel_ner - ../data/processed/NER/NEEL2016/test.conll: 84it [00:46,  2.09it/s]
03/06/2019 01:26:36 PM - SocialMediaIE.evaluation.evaluate_model - INFO - neel_ner_accuracy: 0.90216    neel_ner_accuracy3: 0.90926     neel_ner_precision-overall: 0.04842     neel_ner_recall-overall: 0.17808        neel_ner_f1-measure-overall: 0.07613
1287it [00:00, 17923.78it/s]
wnut17_ner - ../data/processed/NER/WNUT2017/test.conll: 41it [00:33,  2.01it/s]
03/06/2019 01:27:10 PM - SocialMediaIE.evaluation.evaluate_model - INFO - wnut17_ner_accuracy: 0.94729  wnut17_ner_accuracy3: 0.95067   wnut17_ner_precision-overall: 0.67834   wnut17_ner_recall-overall: 0.29693      wnut17_ner_f1-measure-overall: 0.41306
254it [00:00, 14494.40it/s]
ritter_ner - ../data/processed/NER/Ritter/test.conll: 8it [00:04,  1.71it/s]
03/06/2019 01:27:15 PM - SocialMediaIE.evaluation.evaluate_model - INFO - ritter_ner_accuracy: 0.97155  ritter_ner_accuracy3: 0.97318   ritter_ner_precision-overall: 0.72941   ritter_ner_recall-overall: 0.45255      ritter_ner_f1-measure-overall: 0.55856
3850it [00:00, 20275.21it/s]
ritter_ner - ../data/processed/NER/WNUT2016/test.conll: 121it [01:02,  2.01it/s]
03/06/2019 01:28:18 PM - SocialMediaIE.evaluation.evaluate_model - INFO - ritter_ner_accuracy: 0.92991  ritter_ner_accuracy3: 0.93568   ritter_ner_precision-overall: 0.56898   ritter_ner_recall-overall: 0.38238      ritter_ner_f1-measure-overall: 0.45738
397it [00:00, 12493.16it/s]
yodie_ner - ../data/processed/NER/YODIE/test.conll: 13it [00:07,  1.85it/s]
03/06/2019 01:28:26 PM - SocialMediaIE.evaluation.evaluate_model - INFO - yodie_ner_accuracy: 0.93489   yodie_ner_accuracy3: 0.94148    yodie_ner_precision-overall: 0.57323    yodie_ner_recall-overall: 0.42193       yodie_ner_f1-measure-overall: 0.48608


# All POS

## Passthrough

1201it [00:00, 13757.79it/s]
ud_pos - ../data/processed/POS/Tweetbankv2/test.conll: 38it [00:22,  1.86it/s]
03/06/2019 02:13:41 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ud_pos_accuracy: 0.88945      ud_pos_accuracy3: 0.92689
1000it [00:00, 16208.18it/s]
ud_pos - ../data/processed/POS/DiMSUM2016/test.conll: 32it [00:25,  1.21it/s]
03/06/2019 02:14:07 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ud_pos_accuracy: 0.17442      ud_pos_accuracy3: 0.26394
250it [00:00, 10501.51it/s]
ud_pos - ../data/processed/POS/Foster/test.conll: 8it [00:04,  1.95it/s]
03/06/2019 02:14:11 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ud_pos_accuracy: 0.65188      ud_pos_accuracy3: 0.71278
1318it [00:00, 14515.14it/s]
ud_pos - ../data/processed/POS/lowlands/test.conll: 42it [00:22,  2.21it/s]
03/06/2019 02:14:34 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ud_pos_accuracy: 0.61680      ud_pos_accuracy3: 0.66758
500it [00:00, 11586.35it/s]
ark_pos - ../data/processed/POS/Owoputi/test.conll: 16it [00:08,  1.96it/s]
03/06/2019 02:14:43 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ark_pos_accuracy: 0.88367     ark_pos_accuracy3: 0.89220
250it [00:00, 12421.68it/s]
ptb_pos - ../data/processed/POS/TwitIE/test.conll: 8it [00:03,  2.11it/s]
03/06/2019 02:14:47 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ptb_pos_accuracy: 0.87540     ptb_pos_accuracy3: 0.88173
84it [00:00, 7697.65it/s]
ptb_pos - ../data/processed/POS/Ritter/test.conll: 3it [00:01,  1.60it/s]
03/06/2019 02:14:48 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ptb_pos_accuracy: 0.88691     ptb_pos_accuracy3: 0.89305


## Self-attention

1201it [00:00, 13583.28it/s]
ud_pos - ../data/processed/POS/Tweetbankv2/test.conll: 38it [00:20,  1.88it/s]
03/06/2019 08:25:32 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ud_pos_accuracy: 0.90045      ud_pos_accuracy3: 0.93391
1000it [00:00, 16772.45it/s]
ud_pos - ../data/processed/POS/DiMSUM2016/test.conll: 32it [00:24,  1.40it/s]
03/06/2019 08:25:56 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ud_pos_accuracy: 0.17667      ud_pos_accuracy3: 0.25752
250it [00:00, 14040.55it/s]
ud_pos - ../data/processed/POS/Foster/test.conll: 8it [00:03,  2.11it/s]
03/06/2019 08:26:00 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ud_pos_accuracy: 0.65963      ud_pos_accuracy3: 0.71911
1318it [00:00, 5556.72it/s]
ud_pos - ../data/processed/POS/lowlands/test.conll: 42it [00:21,  2.31it/s]
03/06/2019 08:26:22 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ud_pos_accuracy: 0.62388      ud_pos_accuracy3: 0.67318
500it [00:00, 17370.02it/s]
ark_pos - ../data/processed/POS/Owoputi/test.conll: 16it [00:08,  2.03it/s]
03/06/2019 08:26:30 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ark_pos_accuracy: 0.87570     ark_pos_accuracy3: 0.88227
250it [00:00, 12913.34it/s]
ptb_pos - ../data/processed/POS/TwitIE/test.conll: 8it [00:03,  2.14it/s]
03/06/2019 08:26:34 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ptb_pos_accuracy: 0.87293     ptb_pos_accuracy3: 0.87716
84it [00:00, 6881.55it/s]
ptb_pos - ../data/processed/POS/Ritter/test.conll: 3it [00:01,  1.89it/s]
03/06/2019 08:26:36 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ptb_pos_accuracy: 0.87892     ptb_pos_accuracy3: 0.88199


## BiLSTM

1201it [00:00, 10177.59it/s]
ud_pos - ../data/processed/POS/Tweetbankv2/test.conll: 38it [00:23,  1.58it/s]
03/06/2019 02:34:17 PM - SocialMediaIE.evaluation.evaluate_model - INFO - ud_pos_accuracy: 0.91134      ud_pos_accuracy3: 0.93433
1000it [00:00, 8960.64it/s]
ud_pos - ../data/processed/POS/DiMSUM2016/test.conll: 32it [00:26,  1.36it/s]
03/06/2019 02:34:44 PM - SocialMediaIE.evaluation.evaluate_model - INFO - ud_pos_accuracy: 0.17709      ud_pos_accuracy3: 0.26000
250it [00:00, 20938.86it/s]
ud_pos - ../data/processed/POS/Foster/test.conll: 8it [00:03,  1.90it/s]
03/06/2019 02:34:48 PM - SocialMediaIE.evaluation.evaluate_model - INFO - ud_pos_accuracy: 0.64097      ud_pos_accuracy3: 0.69940
1318it [00:00, 11754.17it/s]
ud_pos - ../data/processed/POS/lowlands/test.conll: 42it [00:24,  2.06it/s]
03/06/2019 02:35:12 PM - SocialMediaIE.evaluation.evaluate_model - INFO - ud_pos_accuracy: 0.61190      ud_pos_accuracy3: 0.65823
0it [00:00, ?it/s]03/06/2019 02:35:12 PM - allennlp.data.fields.sequence_label_field - WARNING - Your label namespace was 'ark_pos'. We recommend you use a namespace ending with 'labels' or 'tags', so we don't add UNK and PAD tokens by default to your vocabulary.  See documentation for `non_padded_namespaces` parameter in Vocabulary.
500it [00:00, 15029.47it/s]
ark_pos - ../data/processed/POS/Owoputi/test.conll: 16it [00:08,  1.90it/s]
03/06/2019 02:35:21 PM - SocialMediaIE.evaluation.evaluate_model - INFO - ark_pos_accuracy: 0.88479     ark_pos_accuracy3: 0.89150
0it [00:00, ?it/s]03/06/2019 02:35:21 PM - allennlp.data.fields.sequence_label_field - WARNING - Your label namespace was 'ptb_pos'. We recommend you use a namespace ending with 'labels' or 'tags', so we don't add UNK and PAD tokens by default to your vocabulary.  See documentation for `non_padded_namespaces` parameter in Vocabulary.
250it [00:00, 13812.50it/s]
ptb_pos - ../data/processed/POS/TwitIE/test.conll: 8it [00:04,  1.68it/s]
03/06/2019 02:35:25 PM - SocialMediaIE.evaluation.evaluate_model - INFO - ptb_pos_accuracy: 0.87258     ptb_pos_accuracy3: 0.87927
84it [00:00, 10851.68it/s]
ptb_pos - ../data/processed/POS/Ritter/test.conll: 3it [00:01,  1.52it/s]
03/06/2019 02:35:27 PM - SocialMediaIE.evaluation.evaluate_model - INFO - ptb_pos_accuracy: 0.86847     ptb_pos_accuracy3: 0.87277


# All tasks

## Stacked

1450it [00:00, 13851.07it/s]
multimodal_ner - ../data/processed/NER/MSM2013/test.conll: 46it [00:28,  1.83it/s]
03/07/2019 06:00:31 AM - SocialMediaIE.evaluation.evaluate_model - INFO - multimodal_ner_accuracy: 0.96714      multimodal_ner_accuracy3: 0.97587       multimodal_ner_precision-overall: 0.76912       multimodal_ner_recall-overall: 0.69796 multimodal_ner_f1-measure-overall: 0.73181
3257it [00:01, 3055.28it/s]
multimodal_ner - ../data/processed/NER/MultiModal/test.conll: 102it [00:52,  1.94it/s]
03/07/2019 06:01:25 AM - SocialMediaIE.evaluation.evaluate_model - INFO - multimodal_ner_accuracy: 0.92293      multimodal_ner_accuracy3: 0.93365       multimodal_ner_precision-overall: 0.73837       multimodal_ner_recall-overall: 0.56012 multimodal_ner_f1-measure-overall: 0.63701
2802it [00:00, 19165.10it/s]
broad_ner - ../data/processed/NER/BROAD/test.conll: 88it [00:50,  1.72it/s]
03/07/2019 06:02:15 AM - SocialMediaIE.evaluation.evaluate_model - INFO - broad_ner_accuracy: 0.94163   broad_ner_accuracy3: 0.96942    broad_ner_precision-overall: 0.69174    broad_ner_recall-overall: 0.62315       broad_ner_f1-measure-overall: 0.65566
5369it [00:00, 18955.62it/s]
broad_ner - ../data/processed/NER/Finin/test.conll: 168it [01:35,  1.75it/s]
03/07/2019 06:03:50 AM - SocialMediaIE.evaluation.evaluate_model - INFO - broad_ner_accuracy: 0.96363   broad_ner_accuracy3: 0.97268    broad_ner_precision-overall: 0.49126    broad_ner_recall-overall: 0.53054       broad_ner_f1-measure-overall: 0.51014
1545it [00:00, 18657.24it/s]
broad_ner - ../data/processed/NER/Hege/test.conll: 49it [00:24,  2.18it/s]
03/07/2019 06:04:15 AM - SocialMediaIE.evaluation.evaluate_model - INFO - broad_ner_accuracy: 0.97004   broad_ner_accuracy3: 0.98413    broad_ner_precision-overall: 0.90427    broad_ner_recall-overall: 0.79936       broad_ner_f1-measure-overall: 0.84859
2663it [00:00, 3220.71it/s]
neel_ner - ../data/processed/NER/NEEL2016/test.conll: 84it [00:45,  2.14it/s]
03/07/2019 06:05:01 AM - SocialMediaIE.evaluation.evaluate_model - INFO - neel_ner_accuracy: 0.93009    neel_ner_accuracy3: 0.93727     neel_ner_precision-overall: 0.02704     neel_ner_recall-overall: 0.06621        neel_ner_f1-measure-overall: 0.03840
1287it [00:00, 16586.83it/s]
wnut17_ner - ../data/processed/NER/WNUT2017/test.conll: 41it [00:33,  2.02it/s]
03/07/2019 06:05:35 AM - SocialMediaIE.evaluation.evaluate_model - INFO - wnut17_ner_accuracy: 0.94447  wnut17_ner_accuracy3: 0.94875   wnut17_ner_precision-overall: 0.76508   wnut17_ner_recall-overall: 0.23084      wnut17_ner_f1-measure-overall: 0.35467
254it [00:00, 11056.77it/s]
ritter_ner - ../data/processed/NER/Ritter/test.conll: 8it [00:04,  1.75it/s]
03/07/2019 06:05:39 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ritter_ner_accuracy: 0.96586  ritter_ner_accuracy3: 0.96911   ritter_ner_precision-overall: 0.77083   ritter_ner_recall-overall: 0.27007      ritter_ner_f1-measure-overall: 0.40000
3850it [00:00, 11693.56it/s]
ritter_ner - ../data/processed/NER/WNUT2016/test.conll: 121it [01:02,  2.16it/s]
03/07/2019 06:06:42 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ritter_ner_accuracy: 0.91807  ritter_ner_accuracy3: 0.92646   ritter_ner_precision-overall: 0.56972   ritter_ner_recall-overall: 0.21883      ritter_ner_f1-measure-overall: 0.31621
397it [00:00, 6744.05it/s]
yodie_ner - ../data/processed/NER/YODIE/test.conll: 13it [00:07,  1.87it/s]
03/07/2019 06:06:50 AM - SocialMediaIE.evaluation.evaluate_model - INFO - yodie_ner_accuracy: 0.91820   yodie_ner_accuracy3: 0.92468    yodie_ner_precision-overall: 0.50222    yodie_ner_recall-overall: 0.21004       yodie_ner_f1-measure-overall: 0.29620
119it [00:00, 7441.03it/s]
ritter_chunk - ../data/processed/CHUNKING/Ritter/test.conll: 4it [00:02,  1.79it/s]
03/07/2019 06:06:53 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ritter_chunk_accuracy: 0.80433        ritter_chunk_accuracy3: 0.83377 ritter_chunk_precision-overall: 0.67843 ritter_chunk_recall-overall: 0.70720           ritter_chunk_f1-measure-overall: 0.69252
1201it [00:00, 15497.42it/s]
ud_pos - ../data/processed/POS/Tweetbankv2/test.conll: 38it [00:21,  1.86it/s]
03/07/2019 06:07:14 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ud_pos_accuracy: 0.87761      ud_pos_accuracy3: 0.91589
1000it [00:00, 17874.72it/s]
ud_pos - ../data/processed/POS/DiMSUM2016/test.conll: 32it [00:24,  1.41it/s]
03/07/2019 06:07:39 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ud_pos_accuracy: 0.17776      ud_pos_accuracy3: 0.26042
250it [00:00, 12433.76it/s]
ud_pos - ../data/processed/POS/Foster/test.conll: 8it [00:04,  1.97it/s]
03/07/2019 06:07:43 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ud_pos_accuracy: 0.66631      ud_pos_accuracy3: 0.72017
1318it [00:00, 16544.93it/s]
ud_pos - ../data/processed/POS/lowlands/test.conll: 42it [00:21,  2.32it/s]
03/07/2019 06:08:04 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ud_pos_accuracy: 0.63100      ud_pos_accuracy3: 0.67616
500it [00:00, 16936.14it/s]
ark_pos - ../data/processed/POS/Owoputi/test.conll: 16it [00:08,  1.91it/s]
03/07/2019 06:08:13 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ark_pos_accuracy: 0.85109     ark_pos_accuracy3: 0.86018
250it [00:00, 13914.59it/s]
ptb_pos - ../data/processed/POS/TwitIE/test.conll: 8it [00:03,  2.11it/s]
03/07/2019 06:08:17 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ptb_pos_accuracy: 0.84372     ptb_pos_accuracy3: 0.85533
84it [00:00, 6460.47it/s]
ptb_pos - ../data/processed/POS/Ritter/test.conll: 3it [00:01,  1.89it/s]
03/07/2019 06:08:18 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ptb_pos_accuracy: 0.85249     ptb_pos_accuracy3: 0.86417
118it [00:00, 6958.86it/s]
ritter_ccg - ../data/processed/SUPERSENSE/Ritter/test.conll: 4it [00:02,  1.64it/s]
03/07/2019 06:08:21 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ritter_ccg_accuracy: 0.65954  ritter_ccg_accuracy3: 0.66608   ritter_ccg_precision-overall: 0.24533   ritter_ccg_recall-overall: 0.13871      ritter_ccg_f1-measure-overall: 0.17722
200it [00:00, 8335.43it/s]
ritter_ccg - ../data/processed/SUPERSENSE/Johannsen2014/test.conll: 7it [00:03,  2.08it/s]
03/07/2019 06:08:25 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ritter_ccg_accuracy: 0.54700  ritter_ccg_accuracy3: 0.56201   ritter_ccg_precision-overall: 0.10256   ritter_ccg_recall-overall: 0.06575      ritter_ccg_f1-measure-overall: 0.08013



## Shared L2 0 LR 1e-3

1450it [00:00, 12863.70it/s]
multimodal_ner - ../data/processed/NER/MSM2013/test.conll: 46it [00:28,  1.82it/s]
03/08/2019 03:03:49 PM - SocialMediaIE.evaluation.evaluate_model - INFO - multimodal_ner_accuracy: 0.97136      multimodal_ner_accuracy3: 0.97466       multimodal_ner_precision-overall: 0.76958       multimodal_ner_recall-overall: 0.80884 multimodal_ner_f1-measure-overall: 0.78872
3257it [00:05, 637.43it/s]
multimodal_ner - ../data/processed/NER/MultiModal/test.conll: 102it [2:04:37,  1.96it/s]
03/08/2019 05:08:31 PM - SocialMediaIE.evaluation.evaluate_model - INFO - multimodal_ner_accuracy: 0.94275      multimodal_ner_accuracy3: 0.94890       multimodal_ner_precision-overall: 0.74394       multimodal_ner_recall-overall: 0.71812 multimodal_ner_f1-measure-overall: 0.73080
2802it [00:00, 18067.67it/s]
broad_ner - ../data/processed/NER/BROAD/test.conll: 88it [00:51,  1.86it/s]
03/08/2019 05:09:23 PM - SocialMediaIE.evaluation.evaluate_model - INFO - broad_ner_accuracy: 0.95861   broad_ner_accuracy3: 0.97597    broad_ner_precision-overall: 0.76559    broad_ner_recall-overall: 0.75688       broad_ner_f1-measure-overall: 0.76121
5369it [00:00, 18886.81it/s]
broad_ner - ../data/processed/NER/Finin/test.conll: 168it [01:39,  1.71it/s]
03/08/2019 05:11:03 PM - SocialMediaIE.evaluation.evaluate_model - INFO - broad_ner_accuracy: 0.96551   broad_ner_accuracy3: 0.97148    broad_ner_precision-overall: 0.50086    broad_ner_recall-overall: 0.60839       broad_ner_f1-measure-overall: 0.54941
1545it [00:00, 18037.33it/s]
broad_ner - ../data/processed/NER/Hege/test.conll: 49it [00:25,  2.11it/s]
03/08/2019 05:11:29 PM - SocialMediaIE.evaluation.evaluate_model - INFO - broad_ner_accuracy: 0.97609   broad_ner_accuracy3: 0.98553    broad_ner_precision-overall: 0.90434    broad_ner_recall-overall: 0.86535       broad_ner_f1-measure-overall: 0.88442
2663it [00:00, 14591.52it/s]
neel_ner - ../data/processed/NER/NEEL2016/test.conll: 84it [00:46,  2.08it/s]
03/08/2019 05:12:16 PM - SocialMediaIE.evaluation.evaluate_model - INFO - neel_ner_accuracy: 0.84501    neel_ner_accuracy3: 0.85127     neel_ner_precision-overall: 0.03765     neel_ner_recall-overall: 0.24429        neel_ner_f1-measure-overall: 0.06524
1287it [00:00, 15135.99it/s]
wnut17_ner - ../data/processed/NER/WNUT2017/test.conll: 41it [00:33,  2.05it/s]
03/08/2019 05:12:49 PM - SocialMediaIE.evaluation.evaluate_model - INFO - wnut17_ner_accuracy: 0.94811  wnut17_ner_accuracy3: 0.95093   wnut17_ner_precision-overall: 0.50339   wnut17_ner_recall-overall: 0.42625      wnut17_ner_f1-measure-overall: 0.46162
254it [00:00, 10943.09it/s]
ritter_ner - ../data/processed/NER/Ritter/test.conll: 8it [00:04,  1.64it/s]
03/08/2019 05:12:54 PM - SocialMediaIE.evaluation.evaluate_model - INFO - ritter_ner_accuracy: 0.99065  ritter_ner_accuracy3: 0.99106   ritter_ner_precision-overall: 0.88976   ritter_ner_recall-overall: 0.82482      ritter_ner_f1-measure-overall: 0.85606
3850it [00:00, 10573.42it/s]
ritter_ner - ../data/processed/NER/WNUT2016/test.conll: 121it [01:02,  2.16it/s]
03/08/2019 05:13:57 PM - SocialMediaIE.evaluation.evaluate_model - INFO - ritter_ner_accuracy: 0.93308  ritter_ner_accuracy3: 0.94009   ritter_ner_precision-overall: 0.54859   ritter_ner_recall-overall: 0.47135      ritter_ner_f1-measure-overall: 0.50705
397it [00:00, 11368.54it/s]
yodie_ner - ../data/processed/NER/YODIE/test.conll: 13it [00:07,  1.86it/s]
03/08/2019 05:14:05 PM - SocialMediaIE.evaluation.evaluate_model - INFO - yodie_ner_accuracy: 0.94908   yodie_ner_accuracy3: 0.95468    yodie_ner_precision-overall: 0.64453    yodie_ner_recall-overall: 0.61338       yodie_ner_f1-measure-overall: 0.62857
119it [00:00, 7434.27it/s]
ritter_chunk - ../data/processed/CHUNKING/Ritter/test.conll: 4it [00:02,  1.88it/s]
03/08/2019 05:14:07 PM - SocialMediaIE.evaluation.evaluate_model - INFO - ritter_chunk_accuracy: 0.91126        ritter_chunk_accuracy3: 0.92597 ritter_chunk_precision-overall: 0.86019 ritter_chunk_recall-overall: 0.87120           ritter_chunk_f1-measure-overall: 0.86566
1201it [00:00, 15046.59it/s]
ud_pos - ../data/processed/POS/Tweetbankv2/test.conll: 38it [00:21,  1.89it/s]
03/08/2019 05:14:29 PM - SocialMediaIE.evaluation.evaluate_model - INFO - ud_pos_accuracy: 0.93187      ud_pos_accuracy3: 0.95020
1000it [00:00, 16502.22it/s]
ud_pos - ../data/processed/POS/DiMSUM2016/test.conll: 32it [00:25,  1.33it/s]
03/08/2019 05:14:54 PM - SocialMediaIE.evaluation.evaluate_model - INFO - ud_pos_accuracy: 0.10758      ud_pos_accuracy3: 0.17539
250it [00:00, 10898.60it/s]
ud_pos - ../data/processed/POS/Foster/test.conll: 8it [00:03,  2.02it/s]
03/08/2019 05:14:58 PM - SocialMediaIE.evaluation.evaluate_model - INFO - ud_pos_accuracy: 0.65012      ud_pos_accuracy3: 0.70679
1318it [00:00, 15546.04it/s]
ud_pos - ../data/processed/POS/lowlands/test.conll: 42it [00:21,  2.32it/s]
03/08/2019 05:15:20 PM - SocialMediaIE.evaluation.evaluate_model - INFO - ud_pos_accuracy: 0.62186      ud_pos_accuracy3: 0.66909
500it [00:00, 16153.93it/s]
ark_pos - ../data/processed/POS/Owoputi/test.conll: 16it [00:08,  1.99it/s]
03/08/2019 05:15:28 PM - SocialMediaIE.evaluation.evaluate_model - INFO - ark_pos_accuracy: 0.90632     ark_pos_accuracy3: 0.91121
250it [00:00, 13189.47it/s]
ptb_pos - ../data/processed/POS/TwitIE/test.conll: 8it [00:03,  2.09it/s]
03/08/2019 05:15:32 PM - SocialMediaIE.evaluation.evaluate_model - INFO - ptb_pos_accuracy: 0.89300     ptb_pos_accuracy3: 0.89898
84it [00:00, 5614.42it/s]
ptb_pos - ../data/processed/POS/Ritter/test.conll: 3it [00:01,  1.74it/s]
03/08/2019 05:15:34 PM - SocialMediaIE.evaluation.evaluate_model - INFO - ptb_pos_accuracy: 0.90289     ptb_pos_accuracy3: 0.90658
118it [00:00, 7869.86it/s]
ritter_ccg - ../data/processed/SUPERSENSE/Ritter/test.conll: 4it [00:02,  1.67it/s]
03/08/2019 05:15:36 PM - SocialMediaIE.evaluation.evaluate_model - INFO - ritter_ccg_accuracy: 0.81711  ritter_ccg_accuracy3: 0.82060   ritter_ccg_precision-overall: 0.57766   ritter_ccg_recall-overall: 0.56011      ritter_ccg_f1-measure-overall: 0.56875
200it [00:00, 7338.54it/s]
ritter_ccg - ../data/processed/SUPERSENSE/Johannsen2014/test.conll: 7it [00:03,  2.21it/s]
03/08/2019 05:15:40 PM - SocialMediaIE.evaluation.evaluate_model - INFO - ritter_ccg_accuracy: 0.64883  ritter_ccg_accuracy3: 0.65437   ritter_ccg_precision-overall: 0.36447   ritter_ccg_recall-overall: 0.36347      ritter_ccg_f1-measure-overall: 0.36397


## Shared SA L2 0 LR 1e-3

1450it [00:00, 12890.29it/s]
multimodal_ner - ../data/processed/NER/MSM2013/test.conll: 46it [00:28,  1.87it/s]
03/08/2019 10:15:58 PM - SocialMediaIE.evaluation.evaluate_model - INFO - multimodal_ner_accuracy: 0.97023      multimodal_ner_accuracy3: 0.97583       multimodal_ner_precision-overall: 0.79368       multimodal_ner_recall-overall: 0.76939 multimodal_ner_f1-measure-overall: 0.78135
3257it [00:04, 753.54it/s]
multimodal_ner - ../data/processed/NER/MultiModal/test.conll: 102it [00:54,  2.16it/s]
03/08/2019 10:16:57 PM - SocialMediaIE.evaluation.evaluate_model - INFO - multimodal_ner_accuracy: 0.93772      multimodal_ner_accuracy3: 0.94574       multimodal_ner_precision-overall: 0.75764       multimodal_ner_recall-overall: 0.66596 multimodal_ner_f1-measure-overall: 0.70885
2802it [00:00, 18854.04it/s]
broad_ner - ../data/processed/NER/BROAD/test.conll: 88it [00:50,  1.73it/s]
03/08/2019 10:17:47 PM - SocialMediaIE.evaluation.evaluate_model - INFO - broad_ner_accuracy: 0.95009   broad_ner_accuracy3: 0.97367    broad_ner_precision-overall: 0.75044    broad_ner_recall-overall: 0.67705       broad_ner_f1-measure-overall: 0.71186
5369it [00:00, 18123.29it/s]
broad_ner - ../data/processed/NER/Finin/test.conll: 168it [01:37,  1.71it/s]
03/08/2019 10:19:26 PM - SocialMediaIE.evaluation.evaluate_model - INFO - broad_ner_accuracy: 0.96322   broad_ner_accuracy3: 0.97125    broad_ner_precision-overall: 0.48459    broad_ner_recall-overall: 0.54417       broad_ner_f1-measure-overall: 0.51266
1545it [00:00, 16700.64it/s]
broad_ner - ../data/processed/NER/Hege/test.conll: 49it [00:25,  2.15it/s]
03/08/2019 10:19:51 PM - SocialMediaIE.evaluation.evaluate_model - INFO - broad_ner_accuracy: 0.97309   broad_ner_accuracy3: 0.98539    broad_ner_precision-overall: 0.92017    broad_ner_recall-overall: 0.82810       broad_ner_f1-measure-overall: 0.87171
2663it [00:00, 2814.70it/s]
neel_ner - ../data/processed/NER/NEEL2016/test.conll: 84it [00:45,  2.10it/s]
03/08/2019 10:20:37 PM - SocialMediaIE.evaluation.evaluate_model - INFO - neel_ner_accuracy: 0.86862    neel_ner_accuracy3: 0.87502     neel_ner_precision-overall: 0.03833     neel_ner_recall-overall: 0.19635        neel_ner_f1-measure-overall: 0.06414
1287it [00:00, 15736.76it/s]
wnut17_ner - ../data/processed/NER/WNUT2017/test.conll: 41it [00:32,  2.04it/s]
03/08/2019 10:21:10 PM - SocialMediaIE.evaluation.evaluate_model - INFO - wnut17_ner_accuracy: 0.94905  wnut17_ner_accuracy3: 0.95259   wnut17_ner_precision-overall: 0.65638   wnut17_ner_recall-overall: 0.30556      wnut17_ner_f1-measure-overall: 0.41699
254it [00:00, 10610.35it/s]
ritter_ner - ../data/processed/NER/Ritter/test.conll: 8it [00:04,  1.73it/s]
03/08/2019 10:21:15 PM - SocialMediaIE.evaluation.evaluate_model - INFO - ritter_ner_accuracy: 0.98374  ritter_ner_accuracy3: 0.98517   ritter_ner_precision-overall: 0.88889   ritter_ner_recall-overall: 0.64234      ritter_ner_f1-measure-overall: 0.74576
3850it [00:00, 11313.37it/s]
ritter_ner - ../data/processed/NER/WNUT2016/test.conll: 121it [01:03,  2.14it/s]
03/08/2019 10:22:18 PM - SocialMediaIE.evaluation.evaluate_model - INFO - ritter_ner_accuracy: 0.93045  ritter_ner_accuracy3: 0.93804   ritter_ner_precision-overall: 0.57291   ritter_ner_recall-overall: 0.41290      ritter_ner_f1-measure-overall: 0.47992
397it [00:00, 11707.45it/s]
yodie_ner - ../data/processed/NER/YODIE/test.conll: 13it [00:07,  1.80it/s]
03/08/2019 10:22:26 PM - SocialMediaIE.evaluation.evaluate_model - INFO - yodie_ner_accuracy: 0.94609   yodie_ner_accuracy3: 0.95182    yodie_ner_precision-overall: 0.69951    yodie_ner_recall-overall: 0.52788       yodie_ner_f1-measure-overall: 0.60169
119it [00:00, 8503.08it/s]
ritter_chunk - ../data/processed/CHUNKING/Ritter/test.conll: 4it [00:02,  1.86it/s]
03/08/2019 10:22:28 PM - SocialMediaIE.evaluation.evaluate_model - INFO - ritter_chunk_accuracy: 0.89784        ritter_chunk_accuracy3: 0.91126 ritter_chunk_precision-overall: 0.83412 ritter_chunk_recall-overall: 0.84480           ritter_chunk_f1-measure-overall: 0.83943
1201it [00:00, 15045.29it/s]
ud_pos - ../data/processed/POS/Tweetbankv2/test.conll: 38it [00:21,  1.77it/s]
03/08/2019 10:22:50 PM - SocialMediaIE.evaluation.evaluate_model - INFO - ud_pos_accuracy: 0.89118      ud_pos_accuracy3: 0.92569
1000it [00:00, 15184.38it/s]
ud_pos - ../data/processed/POS/DiMSUM2016/test.conll: 32it [00:25,  1.30it/s]
03/08/2019 10:23:16 PM - SocialMediaIE.evaluation.evaluate_model - INFO - ud_pos_accuracy: 0.17085      ud_pos_accuracy3: 0.25679
250it [00:00, 9639.51it/s]
ud_pos - ../data/processed/POS/Foster/test.conll: 8it [00:03,  2.19it/s]
03/08/2019 10:23:20 PM - SocialMediaIE.evaluation.evaluate_model - INFO - ud_pos_accuracy: 0.65224      ud_pos_accuracy3: 0.71031
1318it [00:00, 16140.51it/s]
ud_pos - ../data/processed/POS/lowlands/test.conll: 42it [00:21,  2.22it/s]
03/08/2019 10:23:42 PM - SocialMediaIE.evaluation.evaluate_model - INFO - ud_pos_accuracy: 0.61362      ud_pos_accuracy3: 0.66192
500it [00:00, 14310.05it/s]
ark_pos - ../data/processed/POS/Owoputi/test.conll: 16it [00:08,  2.08it/s]
03/08/2019 10:23:50 PM - SocialMediaIE.evaluation.evaluate_model - INFO - ark_pos_accuracy: 0.90562     ark_pos_accuracy3: 0.91107
250it [00:00, 12533.48it/s]
ptb_pos - ../data/processed/POS/TwitIE/test.conll: 8it [00:03,  1.89it/s]
03/08/2019 10:23:54 PM - SocialMediaIE.evaluation.evaluate_model - INFO - ptb_pos_accuracy: 0.90285     ptb_pos_accuracy3: 0.90672
84it [00:00, 6015.70it/s]
ptb_pos - ../data/processed/POS/Ritter/test.conll: 3it [00:01,  1.75it/s]
03/08/2019 10:23:56 PM - SocialMediaIE.evaluation.evaluate_model - INFO - ptb_pos_accuracy: 0.89982     ptb_pos_accuracy3: 0.90227
118it [00:00, 7774.06it/s]
ritter_ccg - ../data/processed/SUPERSENSE/Ritter/test.conll: 4it [00:02,  1.78it/s]
03/08/2019 10:23:58 PM - SocialMediaIE.evaluation.evaluate_model - INFO - ritter_ccg_accuracy: 0.82409  ritter_ccg_accuracy3: 0.82671   ritter_ccg_precision-overall: 0.61810   ritter_ccg_recall-overall: 0.57728      ritter_ccg_f1-measure-overall: 0.59699
200it [00:00, 7528.88it/s]
ritter_ccg - ../data/processed/SUPERSENSE/Johannsen2014/test.conll: 7it [00:03,  2.05it/s]
03/08/2019 10:24:02 PM - SocialMediaIE.evaluation.evaluate_model - INFO - ritter_ccg_accuracy: 0.65078  ritter_ccg_accuracy3: 0.65698   ritter_ccg_precision-overall: 0.37110   ritter_ccg_recall-overall: 0.37991      ritter_ccg_f1-measure-overall: 0.37545


## Multitask Stacked L2=0 LR=1e-3

multimodal_ner - ../data/processed/NER/MSM2013/test.conll: 46it [00:27,  1.92it/s]
03/09/2019 03:38:41 AM - SocialMediaIE.evaluation.evaluate_model - INFO - multimodal_ner_accuracy: 0.97353      multimodal_ner_accuracy3: 0.97803       multimodal_ner_precision-overall: 0.79076       multimodal_ner_recall-overall: 0.81497 multimodal_ner_f1-measure-overall: 0.80268
3257it [00:04, 746.06it/s]
multimodal_ner - ../data/processed/NER/MultiModal/test.conll: 102it [00:50,  2.15it/s]
03/09/2019 03:39:36 AM - SocialMediaIE.evaluation.evaluate_model - INFO - multimodal_ner_accuracy: 0.94082      multimodal_ner_accuracy3: 0.94807       multimodal_ner_precision-overall: 0.75041       multimodal_ner_recall-overall: 0.70393 multimodal_ner_f1-measure-overall: 0.72643
2802it [00:00, 19243.20it/s]
broad_ner - ../data/processed/NER/BROAD/test.conll: 88it [00:48,  1.83it/s]
03/09/2019 03:40:25 AM - SocialMediaIE.evaluation.evaluate_model - INFO - broad_ner_accuracy: 0.95742   broad_ner_accuracy3: 0.97664    broad_ner_precision-overall: 0.75168    broad_ner_recall-overall: 0.73869       broad_ner_f1-measure-overall: 0.74513
5369it [00:00, 18933.80it/s]
broad_ner - ../data/processed/NER/Finin/test.conll: 168it [01:35,  1.75it/s]
03/09/2019 03:42:00 AM - SocialMediaIE.evaluation.evaluate_model - INFO - broad_ner_accuracy: 0.96656   broad_ner_accuracy3: 0.97215    broad_ner_precision-overall: 0.51284    broad_ner_recall-overall: 0.63355       broad_ner_f1-measure-overall: 0.56684
1545it [00:00, 18105.52it/s]
broad_ner - ../data/processed/NER/Hege/test.conll: 49it [00:25,  2.17it/s]
03/09/2019 03:42:26 AM - SocialMediaIE.evaluation.evaluate_model - INFO - broad_ner_accuracy: 0.96908   broad_ner_accuracy3: 0.98340    broad_ner_precision-overall: 0.88638    broad_ner_recall-overall: 0.81373       broad_ner_f1-measure-overall: 0.84850
2663it [00:00, 3163.62it/s]
neel_ner - ../data/processed/NER/NEEL2016/test.conll: 84it [00:45,  2.11it/s]
03/09/2019 03:43:12 AM - SocialMediaIE.evaluation.evaluate_model - INFO - neel_ner_accuracy: 0.89039    neel_ner_accuracy3: 0.89741     neel_ner_precision-overall: 0.03265     neel_ner_recall-overall: 0.13927        neel_ner_f1-measure-overall: 0.05289
1287it [00:00, 16437.63it/s]
wnut17_ner - ../data/processed/NER/WNUT2017/test.conll: 41it [00:32,  1.87it/s]
03/09/2019 03:43:45 AM - SocialMediaIE.evaluation.evaluate_model - INFO - wnut17_ner_accuracy: 0.94866  wnut17_ner_accuracy3: 0.95200   wnut17_ner_precision-overall: 0.54816   wnut17_ner_recall-overall: 0.37069      wnut17_ner_f1-measure-overall: 0.44229
254it [00:00, 5301.50it/s]
ritter_ner - ../data/processed/NER/Ritter/test.conll: 8it [00:04,  1.73it/s]
03/09/2019 03:43:50 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ritter_ner_accuracy: 0.98821  ritter_ner_accuracy3: 0.98862   ritter_ner_precision-overall: 0.83333   ritter_ner_recall-overall: 0.80292      ritter_ner_f1-measure-overall: 0.81784
3850it [00:00, 11556.61it/s]
ritter_ner - ../data/processed/NER/WNUT2016/test.conll: 121it [01:02,  2.01it/s]
03/09/2019 03:44:53 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ritter_ner_accuracy: 0.93256  ritter_ner_accuracy3: 0.93852   ritter_ner_precision-overall: 0.52824   ritter_ner_recall-overall: 0.45235      ritter_ner_f1-measure-overall: 0.48736
397it [00:00, 11495.61it/s]
yodie_ner - ../data/processed/NER/YODIE/test.conll: 13it [00:07,  1.77it/s]
03/09/2019 03:45:00 AM - SocialMediaIE.evaluation.evaluate_model - INFO - yodie_ner_accuracy: 0.94696   yodie_ner_accuracy3: 0.95207    yodie_ner_precision-overall: 0.63855    yodie_ner_recall-overall: 0.59108       yodie_ner_f1-measure-overall: 0.61390
119it [00:00, 8096.59it/s]
ritter_chunk - ../data/processed/CHUNKING/Ritter/test.conll: 4it [00:02,  1.89it/s]
03/09/2019 03:45:03 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ritter_chunk_accuracy: 0.90736        ritter_chunk_accuracy3: 0.92208 ritter_chunk_precision-overall: 0.85402 ritter_chunk_recall-overall: 0.87520           ritter_chunk_f1-measure-overall: 0.86448
1201it [00:00, 15433.75it/s]
ud_pos - ../data/processed/POS/Tweetbankv2/test.conll: 38it [00:20,  1.91it/s]
03/09/2019 03:45:24 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ud_pos_accuracy: 0.93344      ud_pos_accuracy3: 0.95062
1000it [00:00, 16984.98it/s]
ud_pos - ../data/processed/POS/DiMSUM2016/test.conll: 32it [00:24,  1.35it/s]
03/09/2019 03:45:48 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ud_pos_accuracy: 0.17564      ud_pos_accuracy3: 0.26345
250it [00:00, 14022.33it/s]
ud_pos - ../data/processed/POS/Foster/test.conll: 8it [00:03,  2.12it/s]
03/09/2019 03:45:52 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ud_pos_accuracy: 0.65153      ud_pos_accuracy3: 0.70715
1318it [00:00, 13765.93it/s]
ud_pos - ../data/processed/POS/lowlands/test.conll: 42it [00:21,  2.32it/s]
03/09/2019 03:46:13 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ud_pos_accuracy: 0.62504      ud_pos_accuracy3: 0.67137
500it [00:00, 15173.44it/s]
ark_pos - ../data/processed/POS/Owoputi/test.conll: 16it [00:08,  2.04it/s]
03/09/2019 03:46:22 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ark_pos_accuracy: 0.91317     ark_pos_accuracy3: 0.91848
250it [00:00, 10882.09it/s]
ptb_pos - ../data/processed/POS/TwitIE/test.conll: 8it [00:03,  2.14it/s]
03/09/2019 03:46:26 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ptb_pos_accuracy: 0.90496     ptb_pos_accuracy3: 0.90848
84it [00:00, 7481.24it/s]
ptb_pos - ../data/processed/POS/Ritter/test.conll: 3it [00:01,  1.88it/s]
03/09/2019 03:46:27 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ptb_pos_accuracy: 0.90412     ptb_pos_accuracy3: 0.90719
118it [00:00, 7071.51it/s]
ritter_ccg - ../data/processed/SUPERSENSE/Ritter/test.conll: 4it [00:02,  1.74it/s]
03/09/2019 03:46:30 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ritter_ccg_accuracy: 0.78394  ritter_ccg_accuracy3: 0.78743   ritter_ccg_precision-overall: 0.48485   ritter_ccg_recall-overall: 0.46499      ritter_ccg_f1-measure-overall: 0.47471
200it [00:00, 7619.70it/s]
ritter_ccg - ../data/processed/SUPERSENSE/Johannsen2014/test.conll: 7it [00:03,  2.21it/s]
03/09/2019 03:46:33 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ritter_ccg_accuracy: 0.62043  ritter_ccg_accuracy3: 0.62826   ritter_ccg_precision-overall: 0.27562   ritter_ccg_recall-overall: 0.28493      ritter_ccg_f1-measure-overall: 0.28020


## Shared SSA

0it [00:00, ?it/s]03/09/2019 09:14:07 AM - allennlp.data.fields.label_field - WARNING - Your label namespace was 'tag_namespace'. We recommend you use a namespace ending with 'labels' or 'tags', so we don't add UNK and PAD tokens by default to your vocabulary.  See documentation for `non_padded_namespaces` parameter in Vocabulary.
03/09/2019 09:14:07 AM - allennlp.data.fields.sequence_label_field - WARNING - Your label namespace was 'multimodal_ner'. We recommend you use a namespace ending with 'labels' or 'tags', so we don't add UNK and PAD tokens by default to your vocabulary.  See documentation for `non_padded_namespaces` parameter in Vocabulary.
1450it [00:00, 5231.00it/s]
multimodal_ner - ../data/processed/NER/MSM2013/test.conll: 91it [00:45,  2.28it/s]
03/09/2019 09:14:52 AM - SocialMediaIE.evaluation.evaluate_model - INFO - multimodal_ner_accuracy: 0.91072      multimodal_ner_accuracy3: 0.97793       multimodal_ner_precision-overall: 0.00000       multimodal_ner_recall-overall: 0.00000 multimodal_ner_f1-measure-overall: 0.00000
3257it [00:00, 15477.14it/s]
multimodal_ner - ../data/processed/NER/MultiModal/test.conll: 204it [01:20,  2.83it/s]
03/09/2019 09:16:13 AM - SocialMediaIE.evaluation.evaluate_model - INFO - multimodal_ner_accuracy: 0.84597      multimodal_ner_accuracy3: 0.90477       multimodal_ner_precision-overall: 0.00000       multimodal_ner_recall-overall: 0.00000 multimodal_ner_f1-measure-overall: 0.00000
0it [00:00, ?it/s]03/09/2019 09:16:13 AM - allennlp.data.fields.sequence_label_field - WARNING - Your label namespace was 'broad_ner'. We recommend you use a namespace ending with 'labels' or 'tags', so we don't add UNK and PAD tokens by default to your vocabulary.  See documentation for `non_padded_namespaces` parameter in Vocabulary.
2802it [00:00, 15349.66it/s]
broad_ner - ../data/processed/NER/BROAD/test.conll: 176it [01:18,  2.29it/s]
03/09/2019 09:17:32 AM - SocialMediaIE.evaluation.evaluate_model - INFO - broad_ner_accuracy: 0.87945   broad_ner_accuracy3: 0.95875    broad_ner_precision-overall: 0.00000    broad_ner_recall-overall: 0.00000       broad_ner_f1-measure-overall: 0.00000
5369it [00:00, 13318.57it/s]
broad_ner - ../data/processed/NER/Finin/test.conll: 336it [02:34,  2.00it/s]
03/09/2019 09:20:07 AM - SocialMediaIE.evaluation.evaluate_model - INFO - broad_ner_accuracy: 0.95608   broad_ner_accuracy3: 0.98540    broad_ner_precision-overall: 0.00000    broad_ner_recall-overall: 0.00000       broad_ner_f1-measure-overall: 0.00000
1545it [00:00, 12492.99it/s]
broad_ner - ../data/processed/NER/Hege/test.conll: 97it [00:39,  2.49it/s]
03/09/2019 09:20:47 AM - SocialMediaIE.evaluation.evaluate_model - INFO - broad_ner_accuracy: 0.89470   broad_ner_accuracy3: 0.97769    broad_ner_precision-overall: 0.00000    broad_ner_recall-overall: 0.00000       broad_ner_f1-measure-overall: 0.00000
0it [00:00, ?it/s]03/09/2019 09:20:47 AM - allennlp.data.fields.sequence_label_field - WARNING - Your label namespace was 'neel_ner'. We recommend you use a namespace ending with 'labels' or 'tags', so we don't add UNK and PAD tokens by default to your vocabulary.  See documentation for `non_padded_namespaces` parameter in Vocabulary.
2663it [00:00, 9434.43it/s]
neel_ner - ../data/processed/NER/NEEL2016/test.conll: 167it [01:12,  2.47it/s]
03/09/2019 09:22:00 AM - SocialMediaIE.evaluation.evaluate_model - INFO - neel_ner_accuracy: 0.97614    neel_ner_accuracy3: 0.98355     neel_ner_precision-overall: 0.00000     neel_ner_recall-overall: 0.00000        neel_ner_f1-measure-overall: 0.00000
0it [00:00, ?it/s]03/09/2019 09:22:00 AM - allennlp.data.fields.sequence_label_field - WARNING - Your label namespace was 'wnut17_ner'. We recommend you use a namespace ending with 'labels' or 'tags', so we don't add UNK and PAD tokens by default to your vocabulary.  See documentation for `non_padded_namespaces` parameter in Vocabulary.
1287it [00:00, 10160.65it/s]
wnut17_ner - ../data/processed/NER/WNUT2017/test.conll: 81it [00:50,  2.29it/s]
03/09/2019 09:22:50 AM - SocialMediaIE.evaluation.evaluate_model - INFO - wnut17_ner_accuracy: 0.93208  wnut17_ner_accuracy3: 0.93922   wnut17_ner_precision-overall: 0.00000   wnut17_ner_recall-overall: 0.00000      wnut17_ner_f1-measure-overall: 0.00000
0it [00:00, ?it/s]03/09/2019 09:22:50 AM - allennlp.data.fields.sequence_label_field - WARNING - Your label namespace was 'ritter_ner'. We recommend you use a namespace ending with 'labels' or 'tags', so we don't add UNK and PAD tokens by default to your vocabulary.  See documentation for `non_padded_namespaces` parameter in Vocabulary.
254it [00:00, 3745.28it/s]
ritter_ner - ../data/processed/NER/Ritter/test.conll: 16it [00:07,  2.05it/s]
03/09/2019 09:22:58 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ritter_ner_accuracy: 0.95468  ritter_ner_accuracy3: 0.95773   ritter_ner_precision-overall: 0.00000   ritter_ner_recall-overall: 0.00000      ritter_ner_f1-measure-overall: 0.00000
3850it [00:00, 15951.66it/s]
ritter_ner - ../data/processed/NER/WNUT2016/test.conll: 241it [01:42,  2.42it/s]
03/09/2019 09:24:40 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ritter_ner_accuracy: 0.90381  ritter_ner_accuracy3: 0.91187   ritter_ner_precision-overall: 0.00000   ritter_ner_recall-overall: 0.00000      ritter_ner_f1-measure-overall: 0.00000
0it [00:00, ?it/s]03/09/2019 09:24:40 AM - allennlp.data.fields.sequence_label_field - WARNING - Your label namespace was 'yodie_ner'. We recommend you use a namespace ending with 'labels' or 'tags', so we don't add UNK and PAD tokens by default to your vocabulary.  See documentation for `non_padded_namespaces` parameter in Vocabulary.
397it [00:00, 3755.85it/s]
yodie_ner - ../data/processed/NER/YODIE/test.conll: 25it [00:11,  2.28it/s]
03/09/2019 09:24:52 AM - SocialMediaIE.evaluation.evaluate_model - INFO - yodie_ner_accuracy: 0.90152   yodie_ner_accuracy3: 0.90712    yodie_ner_precision-overall: 0.00000    yodie_ner_recall-overall: 0.00000       yodie_ner_f1-measure-overall: 0.00000
0it [00:00, ?it/s]03/09/2019 09:24:52 AM - allennlp.data.fields.sequence_label_field - WARNING - Your label namespace was 'ritter_chunk'. We recommend you use a namespace ending with 'labels' or 'tags', so we don't add UNK and PAD tokens by default to your vocabulary.  See documentation for `non_padded_namespaces` parameter in Vocabulary.
119it [00:00, 11931.30it/s]
ritter_chunk - ../data/processed/CHUNKING/Ritter/test.conll: 8it [00:04,  1.79it/s]
03/09/2019 09:24:56 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ritter_chunk_accuracy: 0.21126        ritter_chunk_accuracy3: 0.27446 ritter_chunk_precision-overall: 0.00000 ritter_chunk_recall-overall: 0.00000           ritter_chunk_f1-measure-overall: 0.00000
0it [00:00, ?it/s]03/09/2019 09:24:56 AM - allennlp.data.fields.sequence_label_field - WARNING - Your label namespace was 'ud_pos'. We recommend you use a namespace ending with 'labels' or 'tags', so we don't add UNK and PAD tokens by default to your vocabulary.  See documentation for `non_padded_namespaces` parameter in Vocabulary.
1201it [00:00, 14171.06it/s]
ud_pos - ../data/processed/POS/Tweetbankv2/test.conll: 76it [00:34,  2.83it/s]
03/09/2019 09:25:31 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ud_pos_accuracy: 0.10704      ud_pos_accuracy3: 0.22959
1000it [00:00, 20448.95it/s]
ud_pos - ../data/processed/POS/DiMSUM2016/test.conll: 63it [00:35,  1.74it/s]
03/09/2019 09:26:07 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ud_pos_accuracy: 0.17485      ud_pos_accuracy3: 0.26339
250it [00:00, 22747.16it/s]
ud_pos - ../data/processed/POS/Foster/test.conll: 16it [00:06,  2.75it/s]
03/09/2019 09:26:13 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ud_pos_accuracy: 0.23407      ud_pos_accuracy3: 0.28441
1318it [00:00, 20634.06it/s]
ud_pos - ../data/processed/POS/lowlands/test.conll: 83it [00:35,  2.62it/s]
03/09/2019 09:26:48 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ud_pos_accuracy: 0.15156      ud_pos_accuracy3: 0.20658
0it [00:00, ?it/s]03/09/2019 09:26:48 AM - allennlp.data.fields.sequence_label_field - WARNING - Your label namespace was 'ark_pos'. We recommend you use a namespace ending with 'labels' or 'tags', so we don't add UNK and PAD tokens by default to your vocabulary.  See documentation for `non_padded_namespaces` parameter in Vocabulary.
500it [00:00, 6867.60it/s]
ark_pos - ../data/processed/POS/Owoputi/test.conll: 32it [00:13,  2.45it/s]
03/09/2019 09:27:02 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ark_pos_accuracy: 0.14989     ark_pos_accuracy3: 0.21686
0it [00:00, ?it/s]03/09/2019 09:27:02 AM - allennlp.data.fields.sequence_label_field - WARNING - Your label namespace was 'ptb_pos'. We recommend you use a namespace ending with 'labels' or 'tags', so we don't add UNK and PAD tokens by default to your vocabulary.  See documentation for `non_padded_namespaces` parameter in Vocabulary.
250it [00:00, 17910.60it/s]
ptb_pos - ../data/processed/POS/TwitIE/test.conll: 16it [00:06,  2.79it/s]
03/09/2019 09:27:08 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ptb_pos_accuracy: 0.21260     ptb_pos_accuracy3: 0.27209
84it [00:00, 14037.27it/s]
ptb_pos - ../data/processed/POS/Ritter/test.conll: 6it [00:02,  2.42it/s]
03/09/2019 09:27:11 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ptb_pos_accuracy: 0.07744     ptb_pos_accuracy3: 0.12293
0it [00:00, ?it/s]03/09/2019 09:27:11 AM - allennlp.data.fields.sequence_label_field - WARNING - Your label namespace was 'ritter_ccg'. We recommend you use a namespace ending with 'labels' or 'tags', so we don't add UNK and PAD tokens by default to your vocabulary.  See documentation for `non_padded_namespaces` parameter in Vocabulary.
118it [00:00, 16821.12it/s]
ritter_ccg - ../data/processed/SUPERSENSE/Ritter/test.conll: 8it [00:03,  2.32it/s]
03/09/2019 09:27:15 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ritter_ccg_accuracy: 0.62724  ritter_ccg_accuracy3: 0.63378   ritter_ccg_precision-overall: 0.00000   ritter_ccg_recall-overall: 0.00000      ritter_ccg_f1-measure-overall: 0.00000
200it [00:00, 12505.38it/s]
ritter_ccg - ../data/processed/SUPERSENSE/Johannsen2014/test.conll: 13it [00:05,  2.62it/s]
03/09/2019 09:27:20 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ritter_ccg_accuracy: 0.55809  ritter_ccg_accuracy3: 0.57311   ritter_ccg_precision-overall: 0.00000   ritter_ccg_recall-overall: 0.00000      ritter_ccg_f1-measure-overall: 0.00000


## Shared SSA L2=0 LR=1e-3

0it [00:00, ?it/s]03/09/2019 08:51:13 AM - allennlp.data.fields.label_field - WARNING - Your label namespace was 'tag_namespace'. We recommend you use a namespace ending with 'labels' or 'tags', so we don't add UNK and PAD tokens by default to your vocabulary.  See documentation for `non_padded_namespaces` parameter in Vocabulary.
03/09/2019 08:51:13 AM - allennlp.data.fields.sequence_label_field - WARNING - Your label namespace was 'multimodal_ner'. We recommend you use a namespace ending with 'labels' or 'tags', so we don't add UNK and PAD tokens by default to your vocabulary.  See documentation for `non_padded_namespaces` parameter in Vocabulary.
1450it [00:00, 7493.85it/s]
multimodal_ner - ../data/processed/NER/MSM2013/test.conll: 91it [00:48,  1.98it/s]
03/09/2019 08:52:02 AM - SocialMediaIE.evaluation.evaluate_model - INFO - multimodal_ner_accuracy: 0.97154      multimodal_ner_accuracy3: 0.97600       multimodal_ner_precision-overall: 0.78340       multimodal_ner_recall-overall: 0.78980 multimodal_ner_f1-measure-overall: 0.78659
3257it [00:00, 15379.44it/s]
multimodal_ner - ../data/processed/NER/MultiModal/test.conll: 204it [01:25,  2.51it/s]
03/09/2019 08:53:27 AM - SocialMediaIE.evaluation.evaluate_model - INFO - multimodal_ner_accuracy: 0.94020      multimodal_ner_accuracy3: 0.94688       multimodal_ner_precision-overall: 0.74839       multimodal_ner_recall-overall: 0.69243 multimodal_ner_f1-measure-overall: 0.71932
0it [00:00, ?it/s]03/09/2019 08:53:27 AM - allennlp.data.fields.sequence_label_field - WARNING - Your label namespace was 'broad_ner'. We recommend you use a namespace ending with 'labels' or 'tags', so we don't add UNK and PAD tokens by default to your vocabulary.  See documentation for `non_padded_namespaces` parameter in Vocabulary.
2802it [00:00, 14259.49it/s]
broad_ner - ../data/processed/NER/BROAD/test.conll: 176it [01:19,  2.24it/s]
03/09/2019 08:54:47 AM - SocialMediaIE.evaluation.evaluate_model - INFO - broad_ner_accuracy: 0.95204   broad_ner_accuracy3: 0.97059    broad_ner_precision-overall: 0.72047    broad_ner_recall-overall: 0.73800       broad_ner_f1-measure-overall: 0.72913
5369it [00:00, 9823.69it/s]
broad_ner - ../data/processed/NER/Finin/test.conll: 336it [02:30,  2.38it/s]
03/09/2019 08:57:19 AM - SocialMediaIE.evaluation.evaluate_model - INFO - broad_ner_accuracy: 0.96365   broad_ner_accuracy3: 0.97029    broad_ner_precision-overall: 0.49525    broad_ner_recall-overall: 0.60131       broad_ner_f1-measure-overall: 0.54315
1545it [00:00, 12489.78it/s]
broad_ner - ../data/processed/NER/Hege/test.conll: 97it [00:38,  2.54it/s]
03/09/2019 08:57:58 AM - SocialMediaIE.evaluation.evaluate_model - INFO - broad_ner_accuracy: 0.96424   broad_ner_accuracy3: 0.98151    broad_ner_precision-overall: 0.86962    broad_ner_recall-overall: 0.75253       broad_ner_f1-measure-overall: 0.80685
0it [00:00, ?it/s]03/09/2019 08:57:58 AM - allennlp.data.fields.sequence_label_field - WARNING - Your label namespace was 'neel_ner'. We recommend you use a namespace ending with 'labels' or 'tags', so we don't add UNK and PAD tokens by default to your vocabulary.  See documentation for `non_padded_namespaces` parameter in Vocabulary.
2663it [00:00, 11219.05it/s]
neel_ner - ../data/processed/NER/NEEL2016/test.conll: 167it [01:10,  2.62it/s]
03/09/2019 08:59:08 AM - SocialMediaIE.evaluation.evaluate_model - INFO - neel_ner_accuracy: 0.86826    neel_ner_accuracy3: 0.87409     neel_ner_precision-overall: 0.03779     neel_ner_recall-overall: 0.17352        neel_ner_f1-measure-overall: 0.06207
0it [00:00, ?it/s]03/09/2019 08:59:08 AM - allennlp.data.fields.sequence_label_field - WARNING - Your label namespace was 'wnut17_ner'. We recommend you use a namespace ending with 'labels' or 'tags', so we don't add UNK and PAD tokens by default to your vocabulary.  See documentation for `non_padded_namespaces` parameter in Vocabulary.
1287it [00:00, 10149.97it/s]
wnut17_ner - ../data/processed/NER/WNUT2017/test.conll: 81it [00:50,  2.35it/s]
03/09/2019 08:59:58 AM - SocialMediaIE.evaluation.evaluate_model - INFO - wnut17_ner_accuracy: 0.94888  wnut17_ner_accuracy3: 0.95238   wnut17_ner_precision-overall: 0.53647   wnut17_ner_recall-overall: 0.43678      wnut17_ner_f1-measure-overall: 0.48152
0it [00:00, ?it/s]03/09/2019 08:59:58 AM - allennlp.data.fields.sequence_label_field - WARNING - Your label namespace was 'ritter_ner'. We recommend you use a namespace ending with 'labels' or 'tags', so we don't add UNK and PAD tokens by default to your vocabulary.  See documentation for `non_padded_namespaces` parameter in Vocabulary.
254it [00:00, 3915.96it/s]
ritter_ner - ../data/processed/NER/Ritter/test.conll: 16it [00:07,  2.14it/s]
03/09/2019 09:00:06 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ritter_ner_accuracy: 0.98354  ritter_ner_accuracy3: 0.98415   ritter_ner_precision-overall: 0.75373   ritter_ner_recall-overall: 0.73723      ritter_ner_f1-measure-overall: 0.74539
3850it [00:00, 16639.14it/s]
ritter_ner - ../data/processed/NER/WNUT2016/test.conll: 241it [01:36,  2.32it/s]
03/09/2019 09:01:44 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ritter_ner_accuracy: 0.92889  ritter_ner_accuracy3: 0.93573   ritter_ner_precision-overall: 0.51257   ritter_ner_recall-overall: 0.48143      ritter_ner_f1-measure-overall: 0.49651
0it [00:00, ?it/s]03/09/2019 09:01:44 AM - allennlp.data.fields.sequence_label_field - WARNING - Your label namespace was 'yodie_ner'. We recommend you use a namespace ending with 'labels' or 'tags', so we don't add UNK and PAD tokens by default to your vocabulary.  See documentation for `non_padded_namespaces` parameter in Vocabulary.
397it [00:00, 4854.50it/s]
yodie_ner - ../data/processed/NER/YODIE/test.conll: 25it [00:11,  2.22it/s]
03/09/2019 09:01:55 AM - SocialMediaIE.evaluation.evaluate_model - INFO - yodie_ner_accuracy: 0.95306   yodie_ner_accuracy3: 0.95792    yodie_ner_precision-overall: 0.65577    yodie_ner_recall-overall: 0.63383       yodie_ner_f1-measure-overall: 0.64461
0it [00:00, ?it/s]03/09/2019 09:01:55 AM - allennlp.data.fields.sequence_label_field - WARNING - Your label namespace was 'ritter_chunk'. We recommend you use a namespace ending with 'labels' or 'tags', so we don't add UNK and PAD tokens by default to your vocabulary.  See documentation for `non_padded_namespaces` parameter in Vocabulary.
119it [00:00, 13256.19it/s]
ritter_chunk - ../data/processed/CHUNKING/Ritter/test.conll: 8it [00:03,  2.32it/s]
03/09/2019 09:01:58 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ritter_chunk_accuracy: 0.89351        ritter_chunk_accuracy3: 0.90649 ritter_chunk_precision-overall: 0.82371 ritter_chunk_recall-overall: 0.84480           ritter_chunk_f1-measure-overall: 0.83412
0it [00:00, ?it/s]03/09/2019 09:01:58 AM - allennlp.data.fields.sequence_label_field - WARNING - Your label namespace was 'ud_pos'. We recommend you use a namespace ending with 'labels' or 'tags', so we don't add UNK and PAD tokens by default to your vocabulary.  See documentation for `non_padded_namespaces` parameter in Vocabulary.
1201it [00:00, 18516.03it/s]
ud_pos - ../data/processed/POS/Tweetbankv2/test.conll: 76it [00:33,  2.85it/s]
03/09/2019 09:02:32 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ud_pos_accuracy: 0.92700      ud_pos_accuracy3: 0.94894
1000it [00:00, 20040.44it/s]
ud_pos - ../data/processed/POS/DiMSUM2016/test.conll: 63it [00:34,  1.82it/s]
03/09/2019 09:03:07 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ud_pos_accuracy: 0.17582      ud_pos_accuracy3: 0.26376
250it [00:00, 20812.51it/s]
ud_pos - ../data/processed/POS/Foster/test.conll: 16it [00:05,  2.92it/s]
03/09/2019 09:03:13 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ud_pos_accuracy: 0.65118      ud_pos_accuracy3: 0.70961
1318it [00:00, 20639.15it/s]
ud_pos - ../data/processed/POS/lowlands/test.conll: 83it [00:33,  2.84it/s]
03/09/2019 09:03:46 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ud_pos_accuracy: 0.62140      ud_pos_accuracy3: 0.66995
0it [00:00, ?it/s]03/09/2019 09:03:46 AM - allennlp.data.fields.sequence_label_field - WARNING - Your label namespace was 'ark_pos'. We recommend you use a namespace ending with 'labels' or 'tags', so we don't add UNK and PAD tokens by default to your vocabulary.  See documentation for `non_padded_namespaces` parameter in Vocabulary.
500it [00:00, 6424.15it/s]
ark_pos - ../data/processed/POS/Owoputi/test.conll: 32it [00:13,  2.24it/s]
03/09/2019 09:04:00 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ark_pos_accuracy: 0.90380     ark_pos_accuracy3: 0.91093
0it [00:00, ?it/s]03/09/2019 09:04:00 AM - allennlp.data.fields.sequence_label_field - WARNING - Your label namespace was 'ptb_pos'. We recommend you use a namespace ending with 'labels' or 'tags', so we don't add UNK and PAD tokens by default to your vocabulary.  See documentation for `non_padded_namespaces` parameter in Vocabulary.
250it [00:00, 19275.29it/s]
ptb_pos - ../data/processed/POS/TwitIE/test.conll: 16it [00:05,  2.79it/s]
03/09/2019 09:04:05 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ptb_pos_accuracy: 0.91024     ptb_pos_accuracy3: 0.91482
84it [00:00, 13948.36it/s]
ptb_pos - ../data/processed/POS/Ritter/test.conll: 6it [00:02,  2.58it/s]
03/09/2019 09:04:08 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ptb_pos_accuracy: 0.91026     ptb_pos_accuracy3: 0.91334
0it [00:00, ?it/s]03/09/2019 09:04:08 AM - allennlp.data.fields.sequence_label_field - WARNING - Your label namespace was 'ritter_ccg'. We recommend you use a namespace ending with 'labels' or 'tags', so we don't add UNK and PAD tokens by default to your vocabulary.  See documentation for `non_padded_namespaces` parameter in Vocabulary.
118it [00:00, 14747.99it/s]
ritter_ccg - ../data/processed/SUPERSENSE/Ritter/test.conll: 8it [00:03,  2.38it/s]
03/09/2019 09:04:12 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ritter_ccg_accuracy: 0.80838  ritter_ccg_accuracy3: 0.81013   ritter_ccg_precision-overall: 0.56038   ritter_ccg_recall-overall: 0.54557      ritter_ccg_f1-measure-overall: 0.55288
200it [00:00, 18168.96it/s]
ritter_ccg - ../data/processed/SUPERSENSE/Johannsen2014/test.conll: 13it [00:04,  2.71it/s]
03/09/2019 09:04:17 AM - SocialMediaIE.evaluation.evaluate_model - INFO - ritter_ccg_accuracy: 0.61162  ritter_ccg_accuracy3: 0.61717   ritter_ccg_precision-overall: 0.33360   ritter_ccg_recall-overall: 0.37443      ritter_ccg_f1-measure-overall: 0.35284