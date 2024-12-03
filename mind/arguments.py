from dataclasses import dataclass, field
from typing import Optional,Union
from transformers import TrainingArguments
@dataclass
class ModelArguments:
    """ 
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default = "LLM-Research/Meta-Llama-3-8B",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    #str or list
    lora_checkpoint: Optional[str] = field(
        default=None, metadata={"help": "Path to lora checkpoints"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )
    quantization_bit: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "An optional parameter specifying the number of bits used for quantization. "
                "Quantization is a process that reduces the model size by limiting the number of "
                "bits that represent each weight in the model. A lower number of bits can reduce "
                "the model size and speed up inference, but might also decrease model accuracy. "
                "If not set (None), quantization is not applied."
            )
        },
    )

    lora_rank: Optional[int] = field(
        default=0,
        metadata={
            "help": (
                "balancing between complexity and model flexibility. A higher rank allows more "
                "complex adaptations but increases the number of parameters and computational cost."
            )
        },
    )
    lora_alpha: Optional[float] = field(
        default=0,
        metadata={
            "help": (
                "A higher value results in more significant adjustments, potentially improving adaptation to new tasks or data, "
                "but might also risk overfitting. A lower value makes smaller adjustments, possibly maintaining better generalization."
            )
        }, )

    lora_dropout: Optional[float] = field(
        default=0.1,
        metadata={
            "help": (
                "during training to prevent the model from overly relying on specific patterns in the training data. "
                "Higher dropout rates can improve model generalization but may reduce learning efficiency."
            )
        },
    )
    qformer_query_length : Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "token length of qformer"
            )
        }
    )
    qformer_layer_num : Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "layer number of qformer"
            )
        }
    )
    qformer_cross_attention_freq: Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "cross attention frequency of qformer, which is add into qformer encoders"
            )
        }
    )
    qformer_model_path:Optional[str] = field(
        default = "models/bert"
    )
    ptm_model_path:Optional[str] = field(
        default = "models/roberta"
    )
    input_type:Optional[str] = field(
        default = "text",
        metadata={
            "help": (
                "text or embedding"
            )
        }
    )
    feature: Optional[str] = field(
        default="title",
        metadata={
            "help": (
                "title or authors(format as [organization:author for _ in orgs])"
            )
        }
    )
    text_feature:Optional[str] = field(
        default="title",
        metadata={
            "help": (
                "title only or all"
            )
        }
    )
    text_proj:Optional[str] = field(
        default='linear',
        metadata={
            "help": (
                "if use qformer or a single linear layer"
            )
        }
    )
    packing_size:Optional[int] = field(
        default=10,
        metadata={
            "help": (
                "packing num of papers in a batch"
            )
        }
    )
    graph_path:Optional[str] = field(
        default ='./node_embeddings_for_test.pkl'
    )
    
    #可以是bool或者是str,默认是False
    use_graph:Optional[str] = field(
        default = False,
    )

    use_emb:Optional[str] = field(
        default = False,
    )
    use_text:Optional[str] = field(
        default = True,
    )

    enable_llm_requires_grad: Optional[bool] = field(
        default = False,
        metadata={
            "help": (
                "if use llm requires grad(just lora model)"
            )
        }
    )

    papers_drop_ratio:Optional[float] = field(
        default = 0,
        metadata={
            "help": (
                "if drop papers, if drop , input drop rate"
            )
        }
    )
    enable_embedddings_requires_grad: Optional[bool] = field(
        default = False,
        metadata={
            "help": (
                "if use embeddings requires grad"
            )
        }
    )
    enable_lmhead_requires_grad: Optional[bool] = field(
        default = False,
        metadata={
            "help": (
                "if use lm_head requires grad"
            )
        }
    )
    enable_layernorm_requires_grad: Optional[bool] = field(
        default = False,
        metadata={
            "help": (
                "if use layernorm requires grad"
            )
        }
    )
    use_oagbert:Optional[bool] = field(
        default = False,
    )
    
    lora_ckpt_path:Optional[str] = field(
        default = None,
    )
    text_proj_ckpt_path:Optional[str] = field(
        default = None,
    )
    graph_proj_ckpt_path:Optional[str] = field(
        default = None,
    )
    other_ckpt_path: Optional[str] = field(
        default = None,
    )
    enable_graph_proj_requires_grad:Optional[bool] = field(
        default = False,
    )
    enable_text_proj_requires_grad:Optional[bool] = field(
        default = False,
    )
    enable_label_token_requires_grad:Optional[bool] = field(
        default = False,
    )
    feature_drop_prob:Optional[str]=field(
        default = None,
        metadata={
            "help": (
                "set to {text_embed_drop_rate}_{graph_embed_drop_rate}_{text_drop_rate} for random dropping part of inputs, to reduce overfitting"
            )
        }
    )
    apply_chat_template:Optional[bool] = field(
        default = False,
    )
    use_chunkllama: Optional[bool] = field(
        default=False,
    )
    use_weighted_loss: Optional[bool] = field(
        default=False,
    )    

    use_chat_template: Optional[bool] = field(
        default=False,
    )
    graph_proj: Optional[bool] = field(
        default=None,
    )
#     modules_to_save:list[str] = field(
#         default=['text_proj','graph_proj'],
#         metadata={
#             "help": (
#                 "which modules to save, lora is auto saved, the modules are also trained in training process"
#         )
#     }
# )
@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    author_data: Optional[str] = field(
        default=None, metadata={"help": "The input total author_data (a jsonlines or csv file)."}
    )
    pub_data: Optional[str] = field(
        default=None, metadata={"help": "The input pub_data (a jsonlines or csv file)."}
    )
    embedding_path: Optional[str] = field(
        default=None, metadata={"help": "the path of all paper embeddings."}
    )
    eval_data: Optional[str] = field(
        default=None, metadata={"help": "The input validate author data file (a jsonlines or csv file)."}
    )
    test_data: Optional[str] = field(
        default=None,
    )
    max_seq_length: Optional[int] = field(
        default=2048,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )

    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )

    train_format: str = field(
        default="input-output", metadata={"help": "The format of the training data file (multi-turn or input-output)"},
    )

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    max_seq_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )

    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )

    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    sample: Optional[bool] = field(
        default=True,
        metadata={"help": "sample dataset to balance dataset"},
    )
    sorted_file: Optional[bool] = field(
        default=None,
    )

    normalize_profile: Optional[bool] = field(
        default=False,
    )

    profile_truncation: Optional[int] = field(
        default=300,
    )
    shuffle_profile: Optional[bool] = field(
        default=False,
        metadata={
            "help": "if shuffle profile or not during evaluating or testing"
        }
    )



    def __post_init__(self):
        # extension = self.train_data.split(".")[-1]
        # assert extension in {"jsonl", "json"}, "`train_file` should be a jsonl or a json file."

        assert self.train_format in {"multi-turn", "input-output"}
    



@dataclass
class GLMTrainingArguments(TrainingArguments):
    """
    Arguments pertaining to training.
    """


    lr_scheduler_type: str = field(
        default="cosine",
        metadata={"help": "The scheduler type to use."},
    )
    # lr_scheduler_kwargs: dict = field(
    #     default_factory=dict({
    #         "num_cycles":0.5
    #     }),
    #     metadata={"help": "Args of scheduler."},
    # )
    warmup_ratio: float = field(
        default=0.1,
        metadata={"help": "ratio of training steps to perform linear learning rate warmup for (not applicable to all schedulers)"},
    )

    num_train_epochs: float = field(
        default=3.0,
        metadata={
            "help": "Total number of training epochs to perform"},)

    remove_unused_columns:bool=field(
        default = False,
        metadata={
            "help": "if added other input rows"},)
            
    #参数二选一
    resume_from_checkpoint: Optional[str]=field(
        default = None,
        metadata={
            "help": "if continue finetuning and load from checkpoint"
        }
    )

    report_to:str=field(
        default = 'wandb',
        metadata={"help": "wandb or tensorboard"}
    )    

    eval_steps:int=field(
        default = 1000,
        metadata={"help": "evaluate every n steps"}
        )
    
    evaluation_strategy:str=field(
        default = "steps",
    )

    save_only_model:bool=field(
        default = True,
    )
    ddp_find_unused_parameters:bool=field(
        default=False
    )
    
    save_safetensors:bool=field(
        default= False
    )
    
    logging_steps:bool=field(
        default=1
    )

    greater_is_better:bool=field(
        default=True
    )

    load_best_model_at_end:bool=field(
        default=False
    )
    
    metric_for_best_model:str=field(
        default="AUC"
    )
    
    eval_ground_truth:str=field(
        default = "data/IND-WhoIsWho/ind_valid_author_ground_truth.json"
    )
    
    eval_delay: Optional[float] = field(
        default=0,
        metadata={
            "help": (
                "Number of epochs or steps to wait for before the first evaluation can be performed, depending on the"
                " eval_strategy."
            )
        },
    )

    include_inputs_for_metrics: bool = field(
        default=False, metadata={"help": "Whether or not the inputs will be passed to the `compute_metrics` function."}
    )
    predict_saved_path: Optional[str] = field(
        default=None, metadata={"help": "The path to save the predict result."}
    )