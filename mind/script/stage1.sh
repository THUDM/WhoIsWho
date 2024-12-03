set -ex

pip install -r requirements.txt
wandb login YOUR_WANDB_API_KEY
wandb online   
wandb enabled

NUM_GPUS=8

torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS  pipeline_for_multimodal.py \
    configs/llama3/stage3.json 2>&1 | tee output/llama3/stage3.log


# torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS  pipeline_for_multimodal.py \
#     configs/base/llama3/author.json 2>&1 | tee output/base/llama3/author.log
