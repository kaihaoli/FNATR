import os
import torch
from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments


from logger_config import logger


@dataclass
class Arguments(TrainingArguments):
    model_name_or_path: str = field(
        default='bert-base-uncased',
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    data_name: str = field(default = 'esci', metadata={"help":'esci / msmarco'})

    data_dir: str = field(
        default=None, metadata={"help": "Path to train directory"}
    )
    task_type: str = field(
        default='ir', metadata={"help": "task type: ir / qa"}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics on (a jsonlines file)."
        },
    )

    train_n_passages: int = field(
        default=8,
        metadata={"help": "number of passages for each example (including both positive and negative passages)"}
    )
    positive_size: int = field(
        default=1,
        metadata={"help": "number of positive passages for each example"}
    )
    additional_positives: int = field(
        default=0,
        metadata={"help": "number of positive passages for in-batch false negative removal"}
    )
    share_encoder: bool = field(
        default=True,
        metadata={"help": "no weight sharing between qry passage encoders"}
    )
    use_first_positive: bool = field(
        default=False,
        metadata={"help": "Always use the first positive passage"}
    )
    fill_neg_with_random_ids: bool = field(
        default=False,
        metadata={"help": "Fill the negative size by random docs when short of negative docs"}
    )
    use_scaled_loss: bool = field(
        default=True,
        metadata={"help": "Use scaled loss or not"}
    )
    loss_scale: float = field(
        default=-1.,
        metadata={"help": "loss scale, -1 will use world_size"}
    )
    add_pooler: bool = field(default=False)
    out_dimension: int = field(
        default=768,
        metadata={"help": "output dimension for pooler"}
    )
    t: float = field(default=0.05, metadata={"help": "temperature of biencoder training"})
    l2_normalize: bool = field(default=True, metadata={"help": "L2 normalize embeddings or not"})
    t_warmup: bool = field(default=False, metadata={"help": "warmup temperature"})
    full_contrastive_loss: bool = field(default=True, metadata={"help": "use full contrastive loss or not"})
    n_hard_neg: int = field(default=None, metadata={"help": "the number of hard negatives"})
    n_rand_neg: int = field(default=None, metadata={"help": "the number of random negatives"})
    n_other_neg: int = field(default=None, metadata={"help": "the number of negatives from kq, qq, and kk"})
    other_neg_sampling_strategy: str = field(default=None, metadata={"help": "negative sampling strategy for kq, qq, and kk ('hard', 'random', or None)"})
    use_gradient_checkpointing: bool = field(default=False, metadata={"help": "use gradient checkpointing"})
    add_qq_regularization: bool = field(default=False, metadata={"help": "add query-query regularization for negative generation (https://arxiv.org/pdf/2311.06444)"})
    fn_kk_threshold: float = field(default=False, metadata={"help": "the threshold to remove false negatives based on passage-passage similarity"})
    embedding_dedup_threshold: float = field(default=False, metadata={"help": "the threshold to remove duplicates based on passage-passage similarity"})
    use_softmax_loss: bool = field(default=True, metadata={"help": "use softmax loss"})
    aux_loss_additional_positives_weight: float = field(default=0, metadata={"help": "weight for auxiliary loss between the query and the additional positives"})
    aux_loss_consistency_positives_weight: float = field(default=0, metadata={"help": "weight for auxiliary loss between the primary positive and the additional positives"})


    # following arguments are used for encoding documents
    do_encode: bool = field(default=False, metadata={"help": "run the encoding loop"})
    encode_in_path: str = field(default=None, metadata={"help": "Path to data to encode"})
    encode_save_dir: str = field(default=None, metadata={"help": "where to save the encode"})
    encode_shard_size: int = field(default=int(2 * 10**6))
    encode_batch_size: int = field(default=256)

    # used for index search
    do_search: bool = field(default=False, metadata={"help": "run the index search loop"})
    search_split: str = field(default='dev', metadata={"help": "which split to search"})
    search_batch_size: int = field(default=128, metadata={"help": "query batch size for index search"})
    search_topk: int = field(default=200, metadata={"help": "return topk search results"})
    search_out_dir: str = field(default='', metadata={"help": "output directory for writing search results"})

    # used for knowledge distillation
    do_kd_gen_score: bool = field(default=False, metadata={"help": "run the score generation for distillation"})
    kd_gen_score_split: str = field(default='dev', metadata={
        "help": "Which split to use for generation of teacher score"
    })
    kd_gen_score_batch_size: int = field(default=128, metadata={"help": "batch size for teacher score generation"})
    kd_gen_score_n_neg: int = field(default=30, metadata={"help": "number of negatives to compute teacher scores"})

    do_kd_biencoder: bool = field(default=False, metadata={"help": "knowledge distillation to biencoder"})
    kd_mask_hn: bool = field(default=True, metadata={"help": "mask out hard negatives for distillation"})
    kd_cont_loss_weight: float = field(default=1.0, metadata={"help": "weight for contrastive loss"})

    q_max_len: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for query."
        },
    )
    p_max_len: int = field(
        default=144,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    dry_run: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set dry_run to True for debugging purpose'}
    )

    def __post_init__(self):
        assert os.path.exists(self.data_dir)
        assert torch.cuda.is_available(), 'Only support running on GPUs'
        assert self.task_type in ['ir', 'qa']

        if self.dry_run:
            self.logging_steps = 1
            self.max_train_samples = self.max_train_samples or 128
            self.num_train_epochs = 1
            self.per_device_train_batch_size = min(2, self.per_device_train_batch_size)
            self.train_n_passages = min(4, self.train_n_passages)
            self.rerank_forward_factor = 1
            self.gradient_accumulation_steps = 1
            self.rlm_num_eval_samples = min(256, self.rlm_num_eval_samples)
            self.max_steps = 30
            self.save_steps = self.eval_steps = 30
            logger.warning('Dry run: set logging_steps=1')

        if self.do_encode:
            assert self.encode_save_dir
            os.makedirs(self.encode_save_dir, exist_ok=True)
            assert os.path.exists(self.encode_in_path)

        if self.do_search:
            assert os.path.exists(self.encode_save_dir)
            assert self.search_out_dir
            os.makedirs(self.search_out_dir, exist_ok=True)

        if self.do_kd_biencoder:
            if self.use_scaled_loss:
                assert not self.kd_mask_hn, 'Use scaled loss only works with not masking out hard negatives'

        if torch.cuda.device_count() <= 1:
            self.logging_steps = min(10, self.logging_steps)

        super(Arguments, self).__post_init__()

        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

        self.label_names = ['labels']
