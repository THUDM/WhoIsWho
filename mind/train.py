#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""

# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.
# Adapted from


import pickle
import logging
import os
import sys
import json
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
    TrainerCallback,
    Seq2SeqTrainingArguments
)
from argparse import ArgumentParser
from trainer import GLMTrainer
from arguments import ModelArguments, DataTrainingArguments, GLMTrainingArguments
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from utils import *
from model import PackingModelForIND
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.
    From https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

cfgs = [i for i in sys.argv if i.endswith('.json')]
parser = HfArgumentParser((ModelArguments, DataTrainingArguments, GLMTrainingArguments))
model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(cfgs[0]))
logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
if training_args.should_log:
    transformers.utils.logging.set_verbosity_info()
log_level = training_args.get_process_log_level()
logger.setLevel(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

# Log on each process the small summary:
logger.warning(
    f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
)
logger.info(f"Training/evaluation parameters {training_args}")
set_seed(training_args.seed)

config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
config.use_cache = False
# config._attn_implementation = "flash_attention_2" #use flash attention
config.model_args = model_args
tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

if training_args.bf16:
    dtype = torch.bfloat16
elif training_args.fp16:
    dtype = torch.float16
else:
    dtype = torch.float32
model = PackingModelForIND.from_pretrained(model_args.model_name_or_path, torch_dtype=dtype ,config=config, trust_remote_code=True,attn_implementation="flash_attention_2").cuda()

if tokenizer.pad_token is None:
    special_token_dict["pad_token"] = DEFAULT_PAD_TOKEN
if tokenizer.eos_token is None:
    special_token_dict["eos_token"] = DEFAULT_EOS_TOKEN
if tokenizer.bos_token is None:
    special_token_dict["bos_token"] = DEFAULT_BOS_TOKEN
if tokenizer.unk_token is None:
    special_token_dict["unk_token"] = DEFAULT_UNK_TOKEN
smart_tokenizer_and_embedding_resize(
    special_tokens_dict=special_token_dict,
    tokenizer=tokenizer,
    model=model,
)
model.add_special_tokens(tokenizer)
ptm_tokenizer = None
if model_args.use_emb:
    if config.model_args.ptm_model_path == config.model_args.model_name_or_path:
        ptm_tokenizer = tokenizer
    else:
        ptm_tokenizer = AutoTokenizer.from_pretrained(model_args.ptm_model_path)


# if model_args.quantization_bit is not None:
#     print(f"Quantized to {model_args.quantization_bit} bit")
#     model = model.quantize(model_args.quantization_bit)

with open(data_args.author_data, "r", encoding="utf-8") as f:
    author_data = json.load(f)
with open(data_args.eval_data, "r", encoding="utf-8") as f:
    eval_data = json.load(f)
with open(data_args.pub_data, "r" , encoding = "utf-8") as f:
    pub_data = json.load(f)

train_dataset = INDPacking(
    (author_data,pub_data),
    tokenizer,
    model_args = model_args,
    data_args = data_args,
    mode ="train",
    ptm_tokenizer = ptm_tokenizer,
    use_graph = model_args.use_graph,
    use_emb = model_args.use_emb 
)

eval_dataset = INDPacking(
    (eval_data,pub_data),
    tokenizer,
    model_args = model_args,
    data_args = data_args,
    mode ="eval",
    ptm_tokenizer = ptm_tokenizer,
    use_graph = model_args.use_graph,
    use_emb = model_args.use_emb 
)

data_collator = DataCollatorForPacking()

if model_args.lora_checkpoint:
    model = PeftModel.from_pretrained(
        model,
        model_args.lora_checkpoint,
        torch_dtype=torch.float32,
    )
    model = model.cuda()

elif model_args.lora_rank and model_args.lora_alpha:
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=model_args.lora_rank,
        target_modules=['q_proj','k_proj','v_proj','o_proj'], # different among different fundation model 
        modules_to_save = ['text_proj','graph_proj'],
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
    )
    model = get_peft_model(model, peft_config).cuda()

if model_args.ckpts:
    state_dict = torch.load(model_args.ckpts, map_location="cpu")
    model.load_state_dict(state_dict,strict=False)
     
if model_args.enable_llm_requires_grad:
    model.unfreeze_lora()    
else:
    model.freeze_lora()
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

logger.info(f"enable_llm_requires_grad:{model_args.enable_llm_requires_grad} , printing trainable params:")
logger.info(str([k for k,v in model.named_parameters() if v.requires_grad]))

trainer = GLMTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)
trainer.train()