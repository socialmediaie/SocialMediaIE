# TODO for SocialMediaIE

## Naming
  * If it is extended to scholarly data, rename to DST-IE
  
## Add support for traditional feature models using the same API
 * Add flashtext support https://github.com/vi3k6i5/flashtext
 * Upload eLMo and fast text models based on tweets
 * Use code from NCRFpp - https://github.com/jiesutd/NCRFpp
 

## Support arbitrary features

### Different types of vocabs:
  * Char vocabs, word vocabs

### Extract multiple features from each column of the data
  * Pass in vocabs for each column
  * Each vocab should have a name denoting type of vocab value
  * E.g. POS vocab, char vocab, wordshape vocab
  * Allow arbitrary feature usage
  * This requires supporting multiple vocabs for each type of feature
  * Each vocab should be either type text or char, 

## Model

### Decoder improvements
  * Add CRF_L and CRF_S based on https://github.com/LiyuanLucasLiu/LM-LSTM-CRF/blob/master/model/crf.py

### Loss improvements
  * Add structured perceptron and structured hinge loss. 
  * Also include cost sensitive softmax-margin loss

## Multi task learning support
  * Add task identification
  * Add data shuffling

## Training
  * Add logging to file all the training stats, implement a new callback
  * Return the best epoch from the model training. Can be used in hyperopt along with num_epochs to find the hyperparams

## Experiment API
  * Add hyperopt support for hyper parameter tuning for the experiment
  * Log across experiments
  * Experiment saves the best model, along with vocabs, data loaders
