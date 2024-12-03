# MIND: Effective Incorrect Assignment Detection through a Multi-Modal Structural-Enhanced Language Model

![Model Diagram](assets/model.png)

This project leverages a multi-modal structural-enhanced language model for effective incorrect assignment detection. We supply detailed instructions below for setting up the environment, downloading the necessary models and datasets, running training scripts, and evaluating the models.

<p align="center">
📃 <a href="需要补充arxiv地址" target="_blank"> Paper </a> 
🤖 <a href="https://www.modelscope.cn/models/canalpang/MIND-lora" target="_blank"> Model Parameters </a> 
💻 <a href="https://github.com/pangaass/M-IND" target="_blank"> GitHub </a>
</p>

## 🚀 Quick Start

### Dependencies
Build a conda environment using:

```bash

conda create -n mind python==3.10
conda activate mind

# Navigate to the current directory
cd ./M-IND

# Install required packages
pip install -r requirements.txt
```

### Datasets

```bash
mkdir data
wget --directory-prefix data https://open-data-set.oss-cn-beijing.aliyuncs.com/oag-benchmark/kddcup-2024/IND-WhoIsWho/IND-WhoIsWho.zip
wget --directory-prefix data https://open-data-set.oss-cn-beijing.aliyuncs.com/oag-benchmark/kddcup-2024/IND-WhoIsWho/IND-test-public.zip
wget --directory-prefix data https://open-data-set.oss-cn-beijing.aliyuncs.com/oag-benchmark/kddcup-2024/IND-WhoIsWho/IND-WhoIsWho-valid.zip

unzip data/IND-WhoIsWho.zip -d data
unzip data/IND-test-public.zip -d data
unzip data/IND-WhoIsWho-valid.zip -d data
```

### Required Models and Parameters

download Meta-Llama-3-8B and roberta model from [huggingface](https://huggingface.co/models) or [modelscope](https://www.modelscope.cn/models) 

download and trained checkpoints to params/ from [modelscope](https://www.modelscope.cn/models/canalpang/MIND-lora/summary)

download node embeddings of GCCAD from [here](https://pan.baidu.com/s/1T9fR1dWUdMmf81RHc38dlA?pwd=uqy6)

### Reproduce from checkpoint

```bash 
# Firstly edit model_name_or_path and ptm_model_path from configs/reproduce.json 
bash script/reproduce.sh 

python data/IND-WhoIsWho-valid/eval_valid_ind.py  -hp output/predict/predict_res.json -rf data/IND-WhoIsWho-valid/ind_valid_author_ground_truth.json -l result.log
```

### Train & Evaluate

Config  
change the path of all the config file( in ./configs/llama3/* ):
- ptm_model_path : path to roberta model
- model_name_or_path : path to Meta-Llama-3-8B
- stage1.sh : add your wandb API key


```bash
bash script/stage1.sh

# get best eval epoch and edit "lora_ckpt_path" configs/llama3/stage2.json

bash script/stage2.sh

# get best eval step and edit "lora_ckpt _path" and "text_proj_ckpt_path" configs/llama3/stage3.json

bash script/stage3.sh

# get best eval step and edit "lora_ckpt_path" , "text_proj_ckpt_path" and "graph_proj_ckpt_path" configs/llama3/eval.json

#eval 
bash script/eval.sh

```

## Citation
```
@artical{pang2024mind,
      title={MIND: Effective Incorrect Assignment Detection through a Multi-Modal Structural-Enhanced Language Model}, 
      author={},
      journal={arXiv preprint arXiv:},
      year={2024},
}