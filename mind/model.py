from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import math
import sys
sys.path.append('.')
from transformers import AutoModel,AutoTokenizer,AutoConfig
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
)
from bin.modeling_chatglm4 import ChatGLMPreTrainedModel as ChatGLM4PretrainedModel
from bin.modeling_chatglm4 import  ChatGLMModel as ChatGLM4Model
from bin.modeling_chatglm3 import ChatGLMPreTrainedModel as ChatGLM3PretrainedModel
from bin.modeling_chatglm3 import ChatGLMModel as ChatGLM3Model

import logging
logger = logging.getLogger(__name__)
from transformers import LlamaPreTrainedModel,LlamaModel
from transformers import Qwen2Model,Qwen2PreTrainedModel
# from transformers.models.llama.modeling_llama import _get_unpad_data,repeat_kv
# from transformers.utils import is_flash_attn_2_available
# if is_flash_attn_2_available():
#     from flash_attn import flash_attn_func, flash_attn_varlen_func
#     from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

# from Qformer import BertConfig, BertPreTrainedModel, BertModel
from utils import LABEL_TOKEN,EMBED_TOKEN,GRAPH_TOKEN,TRAINABLE_SPECIAL_TOKENS
import torch.nn.functional as F
from dataclasses import dataclass
from transformers.utils import ModelOutput

class LlamaMLP(nn.Module):
    def __init__(self, in_size, intermediate_size, out_size ):
        super().__init__()
        self.gate = nn.Linear(in_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, out_size)
        self.up_proj = nn.Linear(in_size, intermediate_size)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate(x)) * self.up_proj(x))

class LlamaModelForIND(LlamaPreTrainedModel):
    def __init__(self,config):
        super().__init__(config)
        # self.trainable_embeddings = nn.Embedding(len(TRAINABLE_SPECIAL_TOKENS),config.hidden_size)
        
        if config.model_args.use_chunkllama:
            import sys
            sys.path.append("./")
            from ChunkLlama.chunkllama_attn_replace import replace_with_chunkllama
            replace_with_chunkllama(pretraining_length=8192)

        self.model = LlamaModel(config)
        self.config = config
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if self.config.model_args.use_emb:
            self.init_text_proj(config)
        if self.config.model_args.use_graph:
            self.init_graph_proj(config)
            # self.graph_proj = nn.Linear(1538,config.hidden_size)
        # if  self.config.model_args.enable_label_token_requires_grad:
        #     self.label_learner = nn.Linear(config.hidden_size, config.hidden_size)
        self.init_weights()

    def init_text_proj(self,config):
        # self.config = config
        if not config.model_args.use_oagbert:
            ptm_config = AutoConfig.from_pretrained(config.model_args.ptm_model_path)
            if config.model_args.ptm_model_path == config.model_args.model_name_or_path:
                self.ptm = self.model
            else:
                self.ptm = AutoModel.from_pretrained(config.model_args.ptm_model_path)
                self.ptm_tokenizer = AutoTokenizer.from_pretrained(config.model_args.ptm_model_path)
                for _, param in self.ptm.named_parameters():
                    param.requires_grad = False
            # if self.config.model_args.text_proj =='qformer':
            #     self.text_proj = self._init_qformer(config,ptm_config.hidden_size,config.hidden_size)
            # else:
            #     self.text_proj = LlamaMLP(768,config.hidden_size*2,config.hidden_size)
        elif config.model_args.use_oagbert:
            from cogdl.oag import oagbert
            _, self.ptm = oagbert("oagbert-v2-sim")
            # self.text_proj = LlamaMLP(768,config.hidden_size*2,config.hidden_size)
        if config.model_args.text_proj =='qformer':
            self.text_proj = self._init_qformer(config,768,config.hidden_size)
        elif config.model_args.text_proj == 'crossattn':
            self.text_proj = CrossAttention(in_dim = 768, out_dim = 4096, query_length = 1, attention_heads= 2)
        elif config.model_args.text_proj == 'linear':
            self.text_proj = LlamaMLP(768,config.hidden_size*2,config.hidden_size)
        elif config.model_args.text_proj == 'naive_linear':
            self.text_proj = nn.Linear(768,config.hidden_size)
        
    def init_graph_proj(self,config):
        # self.config = config
        if self.config.model_args.use_graph:
            if self.config.model_args.graph_proj == 'naive_linear':
                self.graph_proj = nn.Linear(1538,config.hidden_size)
            else:
                self.graph_proj = LlamaMLP(1538,config.hidden_size*2,config.hidden_size)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model
    
    def get_oagbert_tokenizer(self):
        return self.ptm
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        text_inputs:Optional[List[str]] = None,
        graph_emb:Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        *args,
        **kwargs
    
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # ptm_inputs = self.ptm_tokenizer(texts, return_tensors='pt', add_special_tokens=False, truncation=True,padding="max_length",max_length=512)
        # ptm_inputs = {k: v.to(self.device) for k, v in ptm_inputs.items()}
        # if self.config.model_args.lora_rank and self.config.model_args.input_type != 'text':
        #     inputs_embeds = self.model.embed_tokens(input_ids)
        # else:
        #     inputs_embeds = self.model.embed_tokens(input_ids)
        
        inputs_embeds = self.model.embed_tokens(input_ids)
        # if self.config.model_args.enable_text_proj_requires_grad or self.config.model_args.enable_graph_proj_requires_grad:
        #     inputs_embeds = inputs_embeds.detach()
        inputs_embeds = inputs_embeds.clone()

        if self.config.model_args.use_emb:
            if self.config.model_args.use_oagbert:
                with torch.no_grad():
                    ptm_last_hidden_states,ptm_outputs = self.ptm.bert.forward(
                        **text_inputs,
                        output_all_encoded_layers=False,
                        checkpoint_activations=False,
                    )
            else:
                with torch.no_grad(): 
                    ptm_outputs = self.ptm(**text_inputs)
                    ptm_last_hidden_states = ptm_outputs.last_hidden_state

            if self.config.model_args.text_proj == "qformer":
                text_embeds = self.text_proj(
                    encoder_hidden_states=ptm_last_hidden_states,
                    encoder_attention_mask=text_inputs['attention_mask'],
                    use_cache=True,
                    return_dict=True,
                )
                text_embeds = text_embeds.view(-1, text_embeds.shape[-1]).contiguous()
            elif self.config.model_args.text_proj == "crossattn":
                text_embeds = self.text_proj(
                    encoder_hidden_states=ptm_last_hidden_states,
                    encoder_attention_mask=text_inputs['attention_mask'].unsqueeze(1).unsqueeze(1),
                    output_attentions=False,
                )[0].transpose(0,1)
            elif self.config.model_args.text_proj == "linear":
                text_embeds = (ptm_last_hidden_states*text_inputs['attention_mask'].unsqueeze(-1)).sum(dim=1)/text_inputs['attention_mask'].sum(dim=1).unsqueeze(-1)
                text_embeds = self.text_proj(text_embeds)                

            embedding_ids = torch.masked_select(torch.arange(input_ids.shape[-1], device = self.device).unsqueeze(0), input_ids == self.EMBED_TOKEN_IDS) # need define add special token
            inputs_embeds[:,embedding_ids] = text_embeds

        if self.config.model_args.use_graph:
            if graph_emb.shape[0] !=0:
                graph_ids = torch.masked_select(torch.arange(input_ids.shape[-1], device = self.device).unsqueeze(0), input_ids == self.GRAPH_TOKEN_IDS)
                inputs_embeds[:,graph_ids] = self.graph_proj(graph_emb.to(self.dtype))
        
        labels_pos = torch.masked_select(torch.arange(input_ids.shape[-1], device = self.device), input_ids == self.LABEL_TOKEN_IDS)

        # if self.config.model_args.enable_label_token_requires_grad:
        #     label_token_hidden_states = inputs_embeds[:,labels_pos]
        #     label_token_hidden_states = self.label_learner(label_token_hidden_states)
        #     inputs_embeds[:,labels_pos] = label_token_hidden_states


        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )



        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            # input_ids=input_ids if inputs_embeds is None else None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        lm_logits = self.lm_head(hidden_states)
        
        loss = None
        score = None


        if labels is not None:
            lm_logits = lm_logits.float()

            masked_labels = torch.ones_like(input_ids,device= self.device,dtype = torch.long)*-100
            masked_labels[:,labels_pos] = torch.tensor([self.YES_TOKEN_IDS if l == 1 else self.NO_TOKEN_IDS for l in labels],device = self.device).unsqueeze(0)

            lm_logits = lm_logits.to(torch.float32)
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = masked_labels[:,1:].contiguous() 
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.squeeze(0),shift_labels.to(self.device).squeeze(0))
            # if not self.config.model_args.use_weighted_loss:
            #     loss_fct = CrossEntropyLoss(ignore_index=-100)
            #     loss = loss_fct(shift_logits.squeeze(0),shift_labels.to(self.device).squeeze(0))
            # else:
            #     loss_fct = CrossEntropyLoss(ignore_index=-100,reduction='none')
            #     loss = loss_fct(shift_logits.squeeze(0),shift_labels.to(self.device).squeeze(0))
            #     #将loss的非0值挑出来
            #     loss = loss[loss != 0]

            #     weight_pos = 1/max(sum(labels),1) # max to prevent division by zero
            #     weight_neg = 1/max((len(labels) - sum(labels)),1)
            #     weight = torch.tensor([weight_pos if l == 1 else weight_neg for l in labels],device = self.device)
            #     #根据label计算weight,二分类任务的每个样本weight为batch中该类别数量的倒数
            #     loss = torch.dot(loss,weight/2)
                

            lm_logits = lm_logits.to(hidden_states.dtype)
            loss = loss.to(hidden_states.dtype)
        
        
        # else:
        #     #use token n-1 to predict token n 
        #     logits = lm_logits[:,labels_pos-1].detach()
        #     # calculate the logit by softmax
        #     yes_logit,no_logit= logits[:,:,self.YES_TOKEN_IDS],logits[:,:,self.NO_TOKEN_IDS]
        #     score = F.softmax(torch.concat([yes_logit,no_logit],dim=0),dim=0)[0]
        #     return INDModelWithPast(
        #         loss=None,
        #         logits=None,
        #         past_key_values=outputs.past_key_values,
        #         hidden_states=outputs.hidden_states,
        #         attentions=outputs.attentions,
        #         score=score
        #     )

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return INDModelWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            score=score
        )
    def _init_qformer(self,config,in_size,out_size):
            qformer_config = BertConfig.from_pretrained(config.model_args.qformer_model_path)
            qformer_config.encoder_width = in_size
            # insert cross-attention layer every other block
            qformer_config.add_cross_attention = True
            qformer_config.num_hidden_layers = config.model_args.qformer_layer_num #1
            qformer_config.cross_attention_freq = config.model_args.qformer_cross_attention_freq #1
            qformer_config.query_length = config.model_args.qformer_query_length
            qformer_config.out_size = out_size
            return QformerModel.from_pretrained(
                config.model_args.qformer_model_path, config=qformer_config
            )
    
    def add_special_tokens(self, tokenizer):
        
        self.tokenizer = tokenizer
        if self.config.model_args.ptm_model_path == self.config.model_args.model_name_or_path:
            self.ptm_tokenizer = self.tokenizer
        self.LABEL_TOKEN_IDS = torch.tensor(tokenizer.convert_tokens_to_ids(LABEL_TOKEN))
        self.EMBED_TOKEN_IDS = torch.tensor(tokenizer.convert_tokens_to_ids(EMBED_TOKEN))
        self.GRAPH_TOKEN_IDS = torch.tensor(tokenizer.convert_tokens_to_ids(GRAPH_TOKEN))
        YES_TOKEN_IDS, NO_TOKEN_IDS = tokenizer.convert_tokens_to_ids(['Yes','No'])
        self.YES_TOKEN_IDS, self.NO_TOKEN_IDS= torch.tensor(YES_TOKEN_IDS), torch.tensor(NO_TOKEN_IDS)

        # self.trainable_token_ids = tokenizer.convert_tokens_to_ids(TRAINABLE_SPECIAL_TOKENS)
        # self.sorted_new_vocab_ids, self.indices = torch.sort(self.trainable_token_ids)
        # self.old_to_new_indices = torch.searchsorted(self.sorted_new_vocab_ids, old_vocab_ids)


    def freeze_lora(self):
        for name, param in self.model.named_parameters():
            if 'lora' in name:
                param.requires_grad = False
                
    def unfreeze_lora(self): # unfreeze llm lora parameters
        for name, param in self.model.named_parameters(): #匹配并unfreeze所有'lora'参数
            if 'lora' in name:
                param.requires_grad = True
     


@dataclass
class INDModelWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    score: torch.FloatTensor = None

class GLMModelforIND(ChatGLM4PretrainedModel):

    def __init__(self, config, empty_init=True, device=None):
        super().__init__(config)
        # self.trainable_embeddings = nn.Embedding(len(TRAINABLE_SPECIAL_TOKENS),config.hidden_size)

        self.transformer = ChatGLM4Model(config, empty_init=empty_init, device=device)
        self.config = config
        self.vocab_size = config.vocab_size
        # self.lm_head = self.transformer.output_layer
        
        if self.config.model_args.use_emb:
            self.init_text_proj(config)
        if self.config.model_args.use_graph:
            self.init_graph_proj(config)
            # self.graph_proj = nn.Linear(1538,config.hidden_size)
            
        # self.init_weights()

    def init_text_proj(self,config):
        # self.config = config
        if not config.model_args.use_oagbert:
            ptm_config = AutoConfig.from_pretrained(config.model_args.ptm_model_path)
            if config.model_args.ptm_model_path == config.model_args.model_name_or_path:
                self.ptm = self.transformer
            else:
                self.ptm = AutoModel.from_pretrained(config.model_args.ptm_model_path)
                self.ptm_tokenizer = AutoTokenizer.from_pretrained(config.model_args.ptm_model_path)
                for _, param in self.ptm.named_parameters():
                    param.requires_grad = False
            # if self.config.model_args.text_proj =='qformer':
            #     self.text_proj = self._init_qformer(config,ptm_config.hidden_size,config.hidden_size)
            # else:
            #     self.text_proj = LlamaMLP(768,config.hidden_size*2,config.hidden_size)
        elif config.model_args.use_oagbert:
            from cogdl.oag import oagbert
            _, self.ptm = oagbert("oagbert-v2-sim")
            # self.text_proj = LlamaMLP(768,config.hidden_size*2,config.hidden_size)
        if config.model_args.text_proj =='qformer':
            self.text_proj = self._init_qformer(config,768,config.hidden_size)
        elif config.model_args.text_proj == 'crossattn':
            self.text_proj = CrossAttention(in_dim = 768, out_dim = 4096, query_length = 1, attention_heads= 2)
        elif config.model_args.text_proj == 'linear':
            self.text_proj = LlamaMLP(768,config.hidden_size*2,config.hidden_size)

        
    def init_graph_proj(self,config):
        # self.config = config
        if self.config.model_args.use_graph:
            self.graph_proj = LlamaMLP(1538,config.hidden_size*2,config.hidden_size)

    def get_input_embeddings(self):
        return self.transformer.get_input_embeddings()

    def set_input_embeddings(self, value):
        # self.transformer.embeddings = value
        return self.transformer.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.transformer.output_layer

    def set_output_embeddings(self, new_embeddings):
        self.transformer.output_layer = new_embeddings

    def set_decoder(self, decoder):
        self.transformer = decoder

    def get_decoder(self):
        return self.transformer
    
    def get_oagbert_tokenizer(self):
        return self.ptm
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        text_inputs:Optional[List[str]] = None,
        graph_emb:Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        *args,
        **kwargs
    
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # ptm_inputs = self.ptm_tokenizer(texts, return_tensors='pt', add_special_tokens=False, truncation=True,padding="max_length",max_length=512)
        # ptm_inputs = {k: v.to(self.device) for k, v in ptm_inputs.items()}
        # if self.config.model_args.lora_rank and self.config.model_args.input_type != 'text':
        #     inputs_embeds = self.transformer.embed_tokens(input_ids)
        # else:
        #     inputs_embeds = self.transformer.embed_tokens(input_ids)
        inputs_embeds = self.transformer.embedding(input_ids)
        # if self.config.model_args.enable_text_proj_requires_grad or self.config.model_args.enable_graph_proj_requires_grad:
        #     inputs_embeds = inputs_embeds.detach()
        
        # inputs_embeds = inputs_embeds.clone()

        if self.config.model_args.use_emb:
            if self.config.model_args.use_oagbert:
                with torch.no_grad():
                    ptm_last_hidden_states,ptm_outputs = self.ptm.bert.forward(
                        **text_inputs,
                        output_all_encoded_layers=False,
                        checkpoint_activations=False,
                    )
            else:
                with torch.no_grad(): 
                    ptm_outputs = self.ptm(**text_inputs)
                    ptm_last_hidden_states = ptm_outputs.last_hidden_state

            if self.config.model_args.text_proj == "qformer":
                text_embeds = self.text_proj(
                    encoder_hidden_states=ptm_last_hidden_states,
                    encoder_attention_mask=text_inputs['attention_mask'],
                    use_cache=True,
                    return_dict=True,
                )
                text_embeds = text_embeds.view(-1, text_embeds.shape[-1]).contiguous()
            elif self.config.model_args.text_proj == "crossattn":
                text_embeds = self.text_proj(
                    encoder_hidden_states=ptm_last_hidden_states,
                    encoder_attention_mask=text_inputs['attention_mask'].unsqueeze(1).unsqueeze(1),
                    output_attentions=False,
                )[0].transpose(0,1)
            elif self.config.model_args.text_proj == "linear":
                text_embeds = (ptm_last_hidden_states*text_inputs['attention_mask'].unsqueeze(-1)).sum(dim=1)/text_inputs['attention_mask'].sum(dim=1).unsqueeze(-1)
                text_embeds = self.text_proj(text_embeds)                

            embedding_ids = torch.masked_select(torch.arange(input_ids.shape[-1], device = self.device).unsqueeze(0), input_ids == self.EMBED_TOKEN_IDS) # need define add special token
            inputs_embeds[:,embedding_ids] = text_embeds

        if self.config.model_args.use_graph:
            if graph_emb.shape[0] !=0:
                graph_ids = torch.masked_select(torch.arange(input_ids.shape[-1], device = self.device).unsqueeze(0), input_ids == self.GRAPH_TOKEN_IDS)
                inputs_embeds[:,graph_ids] = self.graph_proj(graph_emb.to(self.dtype))
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.transformer(
            # input_ids=input_ids if inputs_embeds is None else None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        lm_logits = self.transformer.output_layer(hidden_states)
        
        loss = None
        score = None
        labels_pos = torch.masked_select(torch.arange(input_ids.shape[-1], device = self.device), input_ids == self.LABEL_TOKEN_IDS)

        if labels is not None:
            lm_logits = lm_logits.float()

            masked_labels = torch.ones_like(input_ids,device= self.device,dtype = torch.long)*-100
            masked_labels[:,labels_pos] = torch.tensor([self.YES_TOKEN_IDS if l == 1 else self.NO_TOKEN_IDS for l in labels],device = self.device).unsqueeze(0)

            lm_logits = lm_logits.to(torch.float32)
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = masked_labels[:,1:].contiguous()

            if not self.config.model_args.use_weighted_loss:
                loss_fct = CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(shift_logits.squeeze(0),shift_labels.to(self.device).squeeze(0))
            else:
                loss_fct = CrossEntropyLoss(ignore_index=-100,reduction = None)
                loss = loss_fct(shift_logits.squeeze(0),shift_labels.to(self.device).squeeze(0))


            lm_logits = lm_logits.to(hidden_states.dtype)
            loss = loss.to(hidden_states.dtype)
        
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return INDModelWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            score=score
        )
    def _init_qformer(self,config,in_size,out_size):
            qformer_config = BertConfig.from_pretrained(config.model_args.qformer_model_path)
            qformer_config.encoder_width = in_size
            # insert cross-attention layer every other block
            qformer_config.add_cross_attention = True
            qformer_config.num_hidden_layers = config.model_args.qformer_layer_num #1
            qformer_config.cross_attention_freq = config.model_args.qformer_cross_attention_freq #1
            qformer_config.query_length = config.model_args.qformer_query_length
            qformer_config.out_size = out_size
            return QformerModel.from_pretrained(
                config.model_args.qformer_model_path, config=qformer_config
            )
    
    def add_special_tokens(self, tokenizer):
        
        self.tokenizer = tokenizer
        if self.config.model_args.ptm_model_path == self.config.model_args.model_name_or_path:
            self.ptm_tokenizer = self.tokenizer
        self.LABEL_TOKEN_IDS = torch.tensor(tokenizer.convert_tokens_to_ids(LABEL_TOKEN))
        self.EMBED_TOKEN_IDS = torch.tensor(tokenizer.convert_tokens_to_ids(EMBED_TOKEN))
        self.GRAPH_TOKEN_IDS = torch.tensor(tokenizer.convert_tokens_to_ids(GRAPH_TOKEN))
        if "glm4" in tokenizer.name_or_path:# for glm4 tokenizer
            YES_TOKEN_IDS, NO_TOKEN_IDS = tokenizer.encode('Yes',add_special_tokens = False)[0],tokenizer.encode('No',add_special_tokens = False)[0]
        else:# for llama tokenizer
            YES_TOKEN_IDS, NO_TOKEN_IDS = tokenizer.convert_tokens_to_ids(['Yes','No'])
        self.YES_TOKEN_IDS, self.NO_TOKEN_IDS= torch.tensor(YES_TOKEN_IDS), torch.tensor(NO_TOKEN_IDS)

        # self.trainable_token_ids = tokenizer.convert_tokens_to_ids(TRAINABLE_SPECIAL_TOKENS)
        # self.sorted_new_vocab_ids, self.indices = torch.sort(self.trainable_token_ids)
        # self.old_to_new_indices = torch.searchsorted(self.sorted_new_vocab_ids, old_vocab_ids)


    # def freeze_lora(self):
    #     for name, param in self.transformer.named_parameters():
    #         if 'lora' in name:
    #             param.requires_grad = False
                
    # def unfreeze_lora(self): # unfreeze llm lora parameters
    #     for name, param in self.transformer.named_parameters(): #匹配并unfreeze所有'lora'参数
    #         if 'lora' in name:
    #             param.requires_grad = True


class Qwen2ModelForIND(Qwen2PreTrainedModel):
    def __init__(self,config):
        super().__init__(config)
        # self.trainable_embeddings = nn.Embedding(len(TRAINABLE_SPECIAL_TOKENS),config.hidden_size)

        self.model = Qwen2Model(config)
        self.config = config
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if self.config.model_args.use_emb:
            self.init_text_proj(config)
        if self.config.model_args.use_graph:
            self.init_graph_proj(config)
            # self.graph_proj = nn.Linear(1538,config.hidden_size)
            
        self.init_weights()

    def init_text_proj(self,config):
        # self.config = config
        if not config.model_args.use_oagbert:
            ptm_config = AutoConfig.from_pretrained(config.model_args.ptm_model_path)
            if config.model_args.ptm_model_path == config.model_args.model_name_or_path:
                self.ptm = self.model
            else:
                self.ptm = AutoModel.from_pretrained(config.model_args.ptm_model_path)
                self.ptm_tokenizer = AutoTokenizer.from_pretrained(config.model_args.ptm_model_path)
                for _, param in self.ptm.named_parameters():
                    param.requires_grad = False
            # if self.config.model_args.text_proj =='qformer':
            #     self.text_proj = self._init_qformer(config,ptm_config.hidden_size,config.hidden_size)
            # else:
            #     self.text_proj = LlamaMLP(768,config.hidden_size*2,config.hidden_size)
        elif config.model_args.use_oagbert:
            from cogdl.oag import oagbert
            _, self.ptm = oagbert("oagbert-v2-sim")
            # self.text_proj = LlamaMLP(768,config.hidden_size*2,config.hidden_size)
        if config.model_args.text_proj =='qformer':
            self.text_proj = self._init_qformer(config,768,config.hidden_size)
        elif config.model_args.text_proj == 'crossattn':
            self.text_proj = CrossAttention(in_dim = 768, out_dim = 4096, query_length = 1, attention_heads= 2)
        elif config.model_args.text_proj == 'linear':
            self.text_proj = LlamaMLP(768,config.hidden_size*2,config.hidden_size)

        
    def init_graph_proj(self,config):
        # self.config = config
        if self.config.model_args.use_graph:
            self.graph_proj = LlamaMLP(1538,config.hidden_size*2,config.hidden_size)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model
    
    def get_oagbert_tokenizer(self):
        return self.ptm
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        text_inputs:Optional[List[str]] = None,
        graph_emb:Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        *args,
        **kwargs
    
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # ptm_inputs = self.ptm_tokenizer(texts, return_tensors='pt', add_special_tokens=False, truncation=True,padding="max_length",max_length=512)
        # ptm_inputs = {k: v.to(self.device) for k, v in ptm_inputs.items()}
        # if self.config.model_args.lora_rank and self.config.model_args.input_type != 'text':
        #     inputs_embeds = self.model.embed_tokens(input_ids)
        # else:
        #     inputs_embeds = self.model.embed_tokens(input_ids)
        
        inputs_embeds = self.model.embed_tokens(input_ids)
        # if self.config.model_args.enable_text_proj_requires_grad or self.config.model_args.enable_graph_proj_requires_grad:
        #     inputs_embeds = inputs_embeds.detach()
        inputs_embeds = inputs_embeds.clone()

        if self.config.model_args.use_emb:
            if self.config.model_args.use_oagbert:
                with torch.no_grad():
                    ptm_last_hidden_states,ptm_outputs = self.ptm.bert.forward(
                        **text_inputs,
                        output_all_encoded_layers=False,
                        checkpoint_activations=False,
                    )
            else:
                with torch.no_grad(): 
                    ptm_outputs = self.ptm(**text_inputs)
                    ptm_last_hidden_states = ptm_outputs.last_hidden_state

            if self.config.model_args.text_proj == "qformer":
                text_embeds = self.text_proj(
                    encoder_hidden_states=ptm_last_hidden_states,
                    encoder_attention_mask=text_inputs['attention_mask'],
                    use_cache=True,
                    return_dict=True,
                )
                text_embeds = text_embeds.view(-1, text_embeds.shape[-1]).contiguous()
            elif self.config.model_args.text_proj == "crossattn":
                text_embeds = self.text_proj(
                    encoder_hidden_states=ptm_last_hidden_states,
                    encoder_attention_mask=text_inputs['attention_mask'].unsqueeze(1).unsqueeze(1),
                    output_attentions=False,
                )[0].transpose(0,1)
            elif self.config.model_args.text_proj == "linear":
                text_embeds = (ptm_last_hidden_states*text_inputs['attention_mask'].unsqueeze(-1)).sum(dim=1)/text_inputs['attention_mask'].sum(dim=1).unsqueeze(-1)
                text_embeds = self.text_proj(text_embeds)                

            embedding_ids = torch.masked_select(torch.arange(input_ids.shape[-1], device = self.device).unsqueeze(0), input_ids == self.EMBED_TOKEN_IDS) # need define add special token
            inputs_embeds[:,embedding_ids] = text_embeds

        if self.config.model_args.use_graph:
            if graph_emb.shape[0] !=0:
                graph_ids = torch.masked_select(torch.arange(input_ids.shape[-1], device = self.device).unsqueeze(0), input_ids == self.GRAPH_TOKEN_IDS)
                inputs_embeds[:,graph_ids] = self.graph_proj(graph_emb.to(self.dtype))
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            # input_ids=input_ids if inputs_embeds is None else None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        lm_logits = self.lm_head(hidden_states)
        
        loss = None
        score = None
        labels_pos = torch.masked_select(torch.arange(input_ids.shape[-1], device = self.device), input_ids == self.LABEL_TOKEN_IDS)

        if labels is not None:
            lm_logits = lm_logits.float()

            masked_labels = torch.ones_like(input_ids,device= self.device,dtype = torch.long)*-100
            masked_labels[:,labels_pos] = torch.tensor([self.YES_TOKEN_IDS if l == 1 else self.NO_TOKEN_IDS for l in labels],device = self.device).unsqueeze(0)

            lm_logits = lm_logits.to(torch.float32)
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = masked_labels[:,1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=-100)

            loss = loss_fct(shift_logits.squeeze(0),shift_labels.to(self.device).squeeze(0))

            lm_logits = lm_logits.to(hidden_states.dtype)
            loss = loss.to(hidden_states.dtype)
        
        
        # else:
        #     #use token n-1 to predict token n 
        #     logits = lm_logits[:,labels_pos-1].detach()
        #     # calculate the logit by softmax
        #     yes_logit,no_logit= logits[:,:,self.YES_TOKEN_IDS],logits[:,:,self.NO_TOKEN_IDS]
        #     score = F.softmax(torch.concat([yes_logit,no_logit],dim=0),dim=0)[0]
        #     return INDModelWithPast(
        #         loss=None,
        #         logits=None,
        #         past_key_values=outputs.past_key_values,
        #         hidden_states=outputs.hidden_states,
        #         attentions=outputs.attentions,
        #         score=score
        #     )

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return INDModelWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            score=score
        )

    
    def add_special_tokens(self, tokenizer):
        
        self.tokenizer = tokenizer
        if self.config.model_args.ptm_model_path == self.config.model_args.model_name_or_path:
            self.ptm_tokenizer = self.tokenizer
        self.LABEL_TOKEN_IDS = torch.tensor(tokenizer.convert_tokens_to_ids(LABEL_TOKEN))
        self.EMBED_TOKEN_IDS = torch.tensor(tokenizer.convert_tokens_to_ids(EMBED_TOKEN))
        self.GRAPH_TOKEN_IDS = torch.tensor(tokenizer.convert_tokens_to_ids(GRAPH_TOKEN))
        YES_TOKEN_IDS, NO_TOKEN_IDS = tokenizer.convert_tokens_to_ids(['Yes','No'])
        self.YES_TOKEN_IDS, self.NO_TOKEN_IDS= torch.tensor(YES_TOKEN_IDS), torch.tensor(NO_TOKEN_IDS)

        # self.trainable_token_ids = tokenizer.convert_tokens_to_ids(TRAINABLE_SPECIAL_TOKENS)
        # self.sorted_new_vocab_ids, self.indices = torch.sort(self.trainable_token_ids)
        # self.old_to_new_indices = torch.searchsorted(self.sorted_new_vocab_ids, old_vocab_ids)


    def freeze_lora(self):
        for name, param in self.model.named_parameters():
            if 'lora' in name:
                param.requires_grad = False
                
    def unfreeze_lora(self): # unfreeze llm lora parameters
        for name, param in self.model.named_parameters(): #匹配并unfreeze所有'lora'参数
            if 'lora' in name:
                param.requires_grad = True


class ChatGLMModelforIND(ChatGLM3PretrainedModel):

    def __init__(self, config, empty_init=True, device=None):
        super().__init__(config)
        # self.trainable_embeddings = nn.Embedding(len(TRAINABLE_SPECIAL_TOKENS),config.hidden_size)

        self.transformer = ChatGLM3Model(config)
        self.config = config
        self.vocab_size = config.vocab_size
        # self.lm_head = self.transformer.output_layer
        
        if self.config.model_args.use_emb:
            self.init_text_proj(config)
        if self.config.model_args.use_graph:
            self.init_graph_proj(config)
            # self.graph_proj = nn.Linear(1538,config.hidden_size)
            
        self.init_weights()

    def init_text_proj(self,config):
        # self.config = config
        if not config.model_args.use_oagbert:
            ptm_config = AutoConfig.from_pretrained(config.model_args.ptm_model_path)
            if config.model_args.ptm_model_path == config.model_args.model_name_or_path:
                self.ptm = self.transformer
            else:
                self.ptm = AutoModel.from_pretrained(config.model_args.ptm_model_path)
                self.ptm_tokenizer = AutoTokenizer.from_pretrained(config.model_args.ptm_model_path)
                for _, param in self.ptm.named_parameters():
                    param.requires_grad = False
            # if self.config.model_args.text_proj =='qformer':
            #     self.text_proj = self._init_qformer(config,ptm_config.hidden_size,config.hidden_size)
            # else:
            #     self.text_proj = LlamaMLP(768,config.hidden_size*2,config.hidden_size)
        elif config.model_args.use_oagbert:
            from cogdl.oag import oagbert
            _, self.ptm = oagbert("oagbert-v2-sim")
            # self.text_proj = LlamaMLP(768,config.hidden_size*2,config.hidden_size)
        if config.model_args.text_proj =='qformer':
            self.text_proj = self._init_qformer(config,768,config.hidden_size)
        elif config.model_args.text_proj == 'crossattn':
            self.text_proj = CrossAttention(in_dim = 768, out_dim = 4096, query_length = 1, attention_heads= 2)
        elif config.model_args.text_proj == 'linear':
            self.text_proj = LlamaMLP(768,config.hidden_size*2,config.hidden_size)

        
    def init_graph_proj(self,config):
        # self.config = config
        if self.config.model_args.use_graph:
            self.graph_proj = LlamaMLP(1538,config.hidden_size*2,config.hidden_size)

    def get_input_embeddings(self):
        return self.transformer.get_input_embeddings()

    def set_input_embeddings(self, value):
        # self.transformer.embeddings = value
        self.transformer.embedding.word_embedding = value

    def get_output_embeddings(self):
        return self.transformer.output_layer

    def set_output_embeddings(self, new_embeddings):
        self.transformer.output_layer = new_embeddings

    def set_decoder(self, decoder):
        self.transformer = decoder

    def get_decoder(self):
        return self.transformer
    
    def get_oagbert_tokenizer(self):
        return self.ptm
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        text_inputs:Optional[List[str]] = None,
        graph_emb:Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        *args,
        **kwargs
    
    ) -> Union[Tuple, BaseModelOutputWithPast]:


        # for chatglm model, embedding output shape is [sequence_len, batch_size, hidden_size]
        # so its needed to transpose before replace text embedding or graph embeddings 
        # and then transpose back after replace

        inputs_embeds = self.transformer.embedding(input_ids).transpose(0,1)

        inputs_embeds = inputs_embeds.clone()

        if self.config.model_args.use_emb:
            if self.config.model_args.use_oagbert:
                with torch.no_grad():
                    ptm_last_hidden_states,ptm_outputs = self.ptm.bert.forward(
                        **text_inputs,
                        output_all_encoded_layers=False,
                        checkpoint_activations=False,
                    )
            else:
                with torch.no_grad(): 
                    ptm_outputs = self.ptm(**text_inputs)
                    ptm_last_hidden_states = ptm_outputs.last_hidden_state

            if self.config.model_args.text_proj == "qformer":
                text_embeds = self.text_proj(
                    encoder_hidden_states=ptm_last_hidden_states,
                    encoder_attention_mask=text_inputs['attention_mask'],
                    use_cache=True,
                    return_dict=True,
                )
                text_embeds = text_embeds.view(-1, text_embeds.shape[-1]).contiguous()
            elif self.config.model_args.text_proj == "crossattn":
                text_embeds = self.text_proj(
                    encoder_hidden_states=ptm_last_hidden_states,
                    encoder_attention_mask=text_inputs['attention_mask'].unsqueeze(1).unsqueeze(1),
                    output_attentions=False,
                )[0].transpose(0,1)
            elif self.config.model_args.text_proj == "linear":
                text_embeds = (ptm_last_hidden_states*text_inputs['attention_mask'].unsqueeze(-1)).sum(dim=1)/text_inputs['attention_mask'].sum(dim=1).unsqueeze(-1)
                text_embeds = self.text_proj(text_embeds)                
            embedding_ids = torch.masked_select(torch.arange(input_ids.shape[-1], device = self.device).unsqueeze(0), input_ids == self.EMBED_TOKEN_IDS) # need define add special token
            inputs_embeds[:,embedding_ids] = text_embeds

        if self.config.model_args.use_graph:
            if graph_emb.shape[0] !=0:
                graph_ids = torch.masked_select(torch.arange(input_ids.shape[-1], device = self.device).unsqueeze(0), input_ids == self.GRAPH_TOKEN_IDS)
                inputs_embeds[:,graph_ids] = self.graph_proj(graph_emb.to(self.dtype))
        
        # and then transpose back after replace
        inputs_embeds = inputs_embeds.transpose(0,1)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.transformer(
            input_ids=input_ids,# if inputs_embeds is None else None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            # output_attentions=output_attentions, # ChatGLM3 implement has no output_attentions
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        lm_logits = self.transformer.output_layer(hidden_states)
        lm_logits = lm_logits.transpose(0, 1).contiguous()
        
        loss = None
        score = None

        if labels is not None:
            labels_pos = torch.masked_select(torch.arange(input_ids.shape[-1], device = self.device), input_ids == self.LABEL_TOKEN_IDS)            

            masked_labels = torch.ones_like(input_ids,device= self.device,dtype = torch.long)*-100
            masked_labels[:,labels_pos] = torch.tensor([self.YES_TOKEN_IDS if l == 1 else self.NO_TOKEN_IDS for l in labels],device = self.device).unsqueeze(0)
            lm_logits = lm_logits.to(torch.float32)
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = masked_labels[:,1:].contiguous()

            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.squeeze(0),shift_labels.to(self.device).squeeze(0))

            lm_logits = lm_logits.to(hidden_states.dtype)
            loss = loss.to(hidden_states.dtype)
        
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return INDModelWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            score=score
        )
    
    def add_special_tokens(self, tokenizer):
        
        self.tokenizer = tokenizer
        self.LABEL_TOKEN_IDS = torch.tensor(tokenizer.convert_tokens_to_ids(LABEL_TOKEN))
        self.EMBED_TOKEN_IDS = torch.tensor(tokenizer.convert_tokens_to_ids(EMBED_TOKEN))
        self.GRAPH_TOKEN_IDS = torch.tensor(tokenizer.convert_tokens_to_ids(GRAPH_TOKEN))
        
        YES_TOKEN_IDS, NO_TOKEN_IDS = tokenizer.convert_tokens_to_ids(['Yes','No'])
        self.YES_TOKEN_IDS, self.NO_TOKEN_IDS= torch.tensor(YES_TOKEN_IDS), torch.tensor(NO_TOKEN_IDS)
