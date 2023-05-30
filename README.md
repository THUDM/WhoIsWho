# WhoIsWho Toolkit
**TL;DR**: By automating data loading, feature creation, model construction, and evaluation processes, the WhoIsWho toolkit is easy for researchers to use and let them develop new name disambiguation approaches.

The toolkit is fully compatible with PyTorch and its associated deep-learning libraries, such as Hugging face. Additionally, the toolkit offers library-agnostic dataset objects that can be used by any other Python deep-learning framework such as TensorFlow.  

## Overview

<img src="whoiswho_pipeline.png" alt="shot" style="zoom:50%;" />

## Install

```
pip install whoiswho
```


## Pipeline
The WhoIsWho toolkit aims at providing lightweight APIs to facilitate researchers to build SOTA name disambiguation algorithms with several lines of code. The abstraction has 4 parts:
* **WhoIsWho Data Loader**: Automating dataset processing and splitting. 

* **Feature Creation**: Providing flexible modules for extracting and creating features.

* **Model Construction**: Adopting pre-defined models in the toolkit library for training and prediction.

* **WhoIsWho Evaluator**: Evaluating models in a task-dependent manner and output the model performance on the validation set.

  

### preliminary document
Download:  https://pan.baidu.com/s/1MFtbsk-qG7KO_8FEHouMFA 

password: p052
* **whoiswho/oagbert-v2-sim**

* **whoiswho/paper-tf-idf**

* **whoiswho/snd-embs**

  


### Commands
choose your name disambiguation task type: **rnd / snd**
```shell
python demo.py --task_type your_task_type
```

The result is stored in ./whoiswho/training/{**your_task**}_result , 

you can upload your results to the benchmark for evaluation!

### Examples of Build Basic RND Algotithms.

```python
# Module-1: Data Loading
train, version = LoadData(name="v3", type="train", task='RND')
valid, version = LoadData(name="v3", type="valid", task='RND')
test, version = LoadData(name="v3", type="test", task='RND')

# Split data into unassigned papers and candidate authors
# Combine unassigned papers and candidate authors into train pairs.
train, version = LoadData(name="v3", type="train", task='RND')
processdata_RND(train, version)
logger.info("Finish pre-process")

# Modules-2: Feature Creation
version = LoadData(name="v3", type="train", task='RND', just_version=True)
adhoc_features = AdhocFeatures(version)
adhoc_features.get_hand_feature()
oagbert_features = OagbertFeatures(version)
oagbert_features.get_oagbert_feature()
logger.info("Finish Train data")

version = LoadData(name="v3", type="valid", task='RND', just_version=True)
adhoc_features = AdhocFeatures(version)
adhoc_features.get_hand_feature()
oagbert_features = OagbertFeatures(version)
oagbert_features.get_oagbert_feature()
logger.info("Finish Valid data")

version = LoadData(name="v3", type="test", task='RND', just_version=True)
adhoc_features = AdhocFeatures(version)
adhoc_features.get_hand_feature()
oagbert_features = OagbertFeatures(version)
oagbert_features.get_oagbert_feature()
logger.info("Finish Test data")

# Module-3: Model Construction
version = LoadData(name="v3", type="train", task='RND', just_version=True)
trainer = RNDTrainer(version)
cell_model_list = trainer.fit()
logger.info("Finish Training")

version = LoadData(name="v3", type="valid", task='RND', just_version=True)
trainer = RNDTrainer(version)
trainer.predict(cell_model_list=cell_model_list) #Use trained model or model path.
logger.info("Finish Predict Valid data")

version = LoadData(name="v3", type="test", task='RND', just_version=True)
trainer = RNDTrainer(version)
trainer.predict(cell_model_list=cell_model_list) #Use trained model or model path.
logger.info("Finish Predict Test data")

# Modules-4: Evaluation
# Upload the results to the whoiswho competition
```



### Examples of Build Basic SND Algotithms.

```python
# Module-1: Data Loading
train, version = LoadData(name="v3", type="train", task='SND')
valid, version = LoadData(name="v3", type="valid", task='SND')
test, version = LoadData(name="v3", type="test", task='SND')
processdata_SND(train, version)
logger.info("Finish pre-process")

# Modules-2: Feature Creation & Module-3: Model Construction
version = LoadData(name="v3", type="train", task='SND',just_version=True)
trainer = SNDTrainer(version)
trainer.fit()
logger.info("Finish Predict Train data")

version = LoadData(name="v3", type="valid", task='SND',just_version=True)
trainer = SNDTrainer(version)
trainer.fit()
logger.info("Finish Predict Valid data")

version = LoadData(name="v3", type="test", task='SND',just_version=True)
trainer = SNDTrainer(version)
trainer.fit()
logger.info("Finish Predict Test data")

#Modules-4: Evaluation
#Upload the results to the whoiswho competition
```

