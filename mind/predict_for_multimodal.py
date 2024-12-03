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
    Seq2SeqTrainingArguments,
    LlamaTokenizer
)
from argparse import ArgumentParser
from trainer import GLMTrainer
from arguments import ModelArguments, DataTrainingArguments, GLMTrainingArguments
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from utils import *
from model import GLMModelforIND,LlamaModelForIND,Qwen2ModelForIND
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

if 'glm' in model_args.model_name_or_path:

    # from bin.modeling_chatglm import ChatGLMForConditionalGeneration
    # model = ChatGLMForConditionalGeneration.from_pretrained(model_args.model_name_or_path, torch_dtype=dtype ,config=config, trust_remote_code=True,empty_init=False,attn_implementation="flash_attention_2").cuda()
    model = GLMModelforIND.from_pretrained(model_args.model_name_or_path, torch_dtype=dtype ,config=config, trust_remote_code=True,empty_init=False,attn_implementation="flash_attention_2").cuda()


elif "Llama" in model_args.model_name_or_path:
    model = LlamaModelForIND.from_pretrained(model_args.model_name_or_path, torch_dtype=dtype ,config=config, trust_remote_code=True,attn_implementation="flash_attention_2").cuda()
elif "Qwen2" in model_args.model_name_or_path:
    model = Qwen2ModelForIND.from_pretrained(model_args.model_name_or_path, torch_dtype=dtype ,config=config, trust_remote_code=True,attn_implementation="flash_attention_2").cuda()


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
    elif config.model_args.use_oagbert:
        ptm_tokenizer = model.get_oagbert_tokenizer()
    else:
        ptm_tokenizer = AutoTokenizer.from_pretrained(model_args.ptm_model_path)

# if model_args.quantization_bit is not None:
#     print(f"Quantized to {model_args.quantization_bit} bit")
#     model = model.quantize(model_args.quantization_bit)

with open(data_args.author_data, "r", encoding="utf-8") as f:
    author_data = json.load(f)
with open(data_args.pub_data, "r" , encoding = "utf-8") as f:
    pub_data = json.load(f)
if data_args.test_data is not None:
    with open(data_args.test_data,'r') as f:
        test_data= json.load(f)
else:
    with open(data_args.eval_data,'r') as f:
        test_data= json.load(f)



datasetclass  = INDPacking

train_dataset = datasetclass(
    (author_data,pub_data),
    tokenizer,
    model_args = model_args,
    data_args = data_args,
    mode ="train",
    ptm_tokenizer = ptm_tokenizer,
    use_graph = model_args.use_graph,
    use_emb = model_args.use_emb 
)

if test_data is not None :
    test_dataset = datasetclass(
        (test_data,pub_data),
        tokenizer,
        model_args = model_args,
        data_args = data_args,
        mode ="eval",
        ptm_tokenizer = ptm_tokenizer,
        use_graph = model_args.use_graph,
        use_emb = model_args.use_emb 
    )
data_collator = DataCollatorForPacking()

modules_to_save = []
if model_args.use_graph and model_args.enable_graph_proj_requires_grad:
    modules_to_save += ['graph_proj']
if model_args.use_emb and model_args.enable_text_proj_requires_grad:
    modules_to_save += ['text_proj']
if model_args.enable_embedddings_requires_grad:
    modules_to_save += ['embed_tokens']
if model_args.enable_lmhead_requires_grad:
    modules_to_save += ['lm_head']
if model_args.enable_layernorm_requires_grad:
    modules_to_save += ['input_layernorm','post_attention_layernorm']
if model_args.enable_llm_requires_grad:
    modules_to_save += ['lora']

if 'Llama' in model_args.model_name_or_path or 'Qwen2' in model_args.model_name_or_path:
    target_modules = ['q_proj','k_proj','v_proj','o_proj']
elif 'glm' in model_args.model_name_or_path:
    target_modules = ['query_key_value']
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=model_args.lora_rank,
    target_modules=target_modules, # different among different fundation model
    lora_alpha=model_args.lora_alpha,
    lora_dropout=model_args.lora_dropout,
)
model = get_peft_model(model, peft_config).cuda() 
if model_args.lora_ckpt_path:  # load lora checkpoint, maybe modified
    if os.path.exists(os.path.join(model_args.lora_ckpt_path, "pytorch_model.bin")):
        paras_path = os.path.join(model_args.lora_ckpt_path, "pytorch_model.bin")
    elif os.path.exists(os.path.join(model_args.lora_ckpt_path, "adapter_model.bin")):
        paras_path = os.path.join(model_args.lora_ckpt_path, "adapter_model.bin")
    else:
        raise ValueError("pytorch_model.bin or adapter_model.bin not found in the lora checkpoint")
    ckpt = torch.load(paras_path)

    for k, v in model.named_parameters():
        if "lora" in k:
            if "default" in k:  # if using torch.save to save peft model, the key will contains "default", such as "base_model.model.model.layers.31.mlp.up_proj.default.weight"
                modify_paras_for_lora = True
            else: #save using peftmodel.save_pretrained 
                modify_paras_for_lora = False
    if modify_paras_for_lora: # add "default" to the key of the parameters
        modified_ckpt = {}
        for k, v in ckpt.items():
            if "lora" in k and "default" not in k:
                n_list = k.split(".")
                n_list.insert(-1, "default")
                n = ".".join(n_list)
                modified_ckpt[n] = v
            else:
                modified_ckpt[k] = v
        loading_res = model.load_state_dict(modified_ckpt,strict = False)
    else:
        loading_res = model.load_state_dict(ckpt,strict = False)
    assert loading_res.unexpected_keys == [], f"missing keys: {loading_res.missing_keys}"
    model = model.cuda()
if model_args.text_proj_ckpt_path:
    text_proj_ckpt = torch.load(model_args.text_proj_ckpt_path)
    text_state_dict = {}
    for k,v in text_proj_ckpt.items():
        if "text_proj" in k:
            text_state_dict[k] = v
    loading_res = model.load_state_dict(text_state_dict, strict=False)
    assert loading_res.unexpected_keys == [], f"missing keys: {loading_res.missing_keys}"

if model_args.graph_proj_ckpt_path:
    graph_proj_ckpt = torch.load(model_args.graph_proj_ckpt_path)
    graph_state_dict = {}
    for k,v in graph_proj_ckpt.items():
        if "graph_proj" in k:
            graph_state_dict[k] = v
    loading_res = model.load_state_dict(graph_state_dict, strict=False)
    assert loading_res.unexpected_keys == [], f"missing keys: {loading_res.missing_keys}"

if model_args.other_ckpt_path:
    other_dict = torch.load(model_args.other_ckpt_path)

    loading_res = model.load_state_dict(other_dict, strict=False)
    assert loading_res.unexpected_keys == [], f"missing keys: {loading_res.missing_keys}"

#[k for k,v in model.named_parameters() if 'text_proj' in k and v.requires_grad]
trainer = GLMTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)
trainer.modules_to_save = modules_to_save #customized variable to be used in trainer.save_models

# with open(trainer.args.eval_ground_truth, "r") as f:
#     ground_truth = json.load(f)
trainer.predict_without_train(test_dataset = test_dataset)
