set -ex
pip install -r requirements.txt


# NUM_GPUS=8

# torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS  pipeline.py \
# deepspeed --include localhost:3  pipeline.py \  #deepspeed is not supported

deepspeed  predict_for_multimodal.py \
    configs/llama3/eval.json