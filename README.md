# Social Media IE


Social Media information extraction tool. It supports the following tasks: 
* **Sequence Tagging**: Named Entity Recognition, Part of Speech, Chunking, CCG Supersense Tagging (List of datasets at: https://socialmediaie.github.io/datasets.html
* **Classification**: Sentiment classification, Abusive Speech Classification, Uncertainity indicator classification
* **Active Learning**: Classification tasks using active learning

**Tutorial on using SocialMediaIE can be found at our [IC2S2 2020 tutorial website](https://socialmediaie.github.io/tutorials/IC2S2_2020/)**

Please cite the following if using the tool: 

* Shubhanshu Mishra. 2019. Multi-dataset-multi-task Neural Sequence Tagging for Information Extraction from Tweets. In Proceedings of the 30th ACM Conference on Hypertext and Social Media (HT '19). ACM, New York, NY, USA, 283-284. DOI: https://doi.org/10.1145/3342220.3344929
* Shubhanshu Mishra. 2019. Information extraction from digital social trace data with applications to social media and scholarly communication data. PhD Dissertation, University of Illinois at Urbana-Champaign. https://shubhanshu.com/phd_thesis/

## Pretrained multi-task models and experimental models
* Mishra, Shubhanshu (2019): Trained models for multi-task multi-dataset learning for text classification as well as sequence tagging in tweets. University of Illinois at Urbana-Champaign. https://doi.org/10.13012/B2IDB-1094364_V1
* Mishra, Shubhanshu (2019): Trained models for multi-task multi-dataset learning for text classification in tweets. University of Illinois at Urbana-Champaign. https://doi.org/10.13012/B2IDB-1917934_V1
* Mishra, Shubhanshu (2019): Trained models for multi-task multi-dataset learning for sequence prediction in tweets. University of Illinois at Urbana-Champaign. https://doi.org/10.13012/B2IDB-0934773_V1






## SociaMediaIE

Main library for doing the analyis

## Notebooks

Example applications of the library and additional experiments
* Tutorial on using SocialMediaIE can be found at our [IC2S2 2020 tutorial website](https://socialmediaie.github.io/tutorials/IC2S2_2020/)


## Experiments

Run experiments based on dataset


## Usage

### For developers

Install in editable mode:
```
pip install -e .
```

### For users
Install as pip package:
```
pip install .
```



## Create documents

```
https://samnicholls.net/2016/06/15/how-to-sphinx-readthedocs/

cd docs/
sphinx-apidoc -o source/ ../SocialMediaIE
```



## Installing python kernel to jupyter

```
python -m ipykernel install --user --name ${CONDA_DEFAULT_ENV} --display-name "Python (${CONDA_DEFAULT_ENV})"
```


## Acknowledgements

This library builds upon AllenNLP and Pytorch. Some of the mutli-task learning code is based on the multi-task learning examples in allennlp. 
