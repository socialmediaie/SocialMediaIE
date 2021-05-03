# TODO for SocialMediaIE

## Naming
  * [ ] If it is extended to scholarly data, rename to DST-IE
  
## Add support for traditional feature models using the same API
 * [ ] Add flashtext support https://github.com/vi3k6i5/flashtext
 * [ ] Upload eLMo and fast text models based on tweets
 * [ ] Use code from NCRFpp - https://github.com/jiesutd/NCRFpp
 * [ ] Use SimpleELMO models (Elmo in various languages to perform SocialMediaIE on multiple languages) - https://github.com/ltgoslo/simple_elmo - https://github.com/HIT-SCIR/ELMoForManyLangs
 

## Support arbitrary features

### Different types of vocabs:
  * [ ] Char vocabs, word vocabs

### Extract multiple features from each column of the data
  * [x] Pass in vocabs for each column
  * [x] Each vocab should have a name denoting type of vocab value. E.g. POS vocab, char vocab, wordshape vocab
  * [ ] Allow arbitrary feature usage. This requires supporting multiple vocabs for each type of feature
  * [ ] Each vocab should be either type text or char, 

## Model

### Decoder improvements
  * [ ] Add CRF_L and CRF_S based on https://github.com/LiyuanLucasLiu/LM-LSTM-CRF/blob/master/model/crf.py

### Loss improvements
  * [ ] Add structured perceptron and structured hinge loss. 
  * [ ] Also include cost sensitive softmax-margin loss

## Multi task learning support
  * [x] Add task identification
  * [x] Add data shuffling

## Training
  * [x] Add logging to file all the training stats, implement a new callback
  * [x] Return the best epoch from the model training. Can be used in hyperopt along with num_epochs to find the hyperparams

## Experiment API
  * [ ] Add hyperopt support for hyper parameter tuning for the experiment
  * [x] Log across experiments
  * [x] Experiment saves the best model, along with vocabs, data loaders

## Frontend

* [ ] Add OpenDistro support for searching data using elastic search on various fields - https://opendistro.github.io/for-elasticsearch-docs/
* [ ] Add ensemble predictions from the multi-task model based on the label similarity
