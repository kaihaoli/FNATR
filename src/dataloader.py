import os
import random

from typing import Tuple, Dict, List, Optional
from datasets import load_dataset, DatasetDict, Dataset
from transformers.file_utils import PaddingStrategy
from transformers import PreTrainedTokenizerFast, Trainer

from config import Arguments
from logger_config import logger

def _slice_with_mod(elements: List, offset: int, cnt: int) -> List:
    return [elements[(offset + idx) % len(elements)] for idx in range(cnt)]


def group_doc_ids(examples: Dict[str, List],
                  positive_size: int,
                  negative_size: int,
                  offset: int,
                  available_doc_ids: List[int] = None,
                  use_first_positive: bool = False,
                  fill_neg_with_random_ids: bool = False) -> List[int]:
    pos_doc_ids: List[List[int]] = []
    positives: List[Dict[str, List]] = examples['positives']
    for idx, ex_pos in enumerate(positives):
        all_pos_doc_ids = ex_pos['doc_id']

        if use_first_positive:
            # keep positives that has higher score than all negatives
            all_pos_doc_ids = [doc_id for p_idx, doc_id in enumerate(all_pos_doc_ids)
                               if p_idx == 0 or ex_pos['score'][p_idx] >= ex_pos['score'][0]
                               or ex_pos['score'][p_idx] > max(examples['negatives'][idx]['score'])]

        cur_pos_doc_ids = _slice_with_mod(all_pos_doc_ids, offset=offset*positive_size, cnt=positive_size)
        cur_pos_doc_ids = [int(doc_id) for doc_id in cur_pos_doc_ids]
        pos_doc_ids.append(cur_pos_doc_ids)

    neg_doc_ids: List[List[int]] = []
    negatives: List[Dict[str, List]] = examples['negatives']
    for ex_neg in negatives:
        if not fill_neg_with_random_ids:
            cur_neg_doc_ids = _slice_with_mod(ex_neg['doc_id'],
                                              offset=offset * negative_size,
                                              cnt=negative_size)
            cur_neg_doc_ids = [int(doc_id) for doc_id in cur_neg_doc_ids]
            neg_doc_ids.append(cur_neg_doc_ids)
        else:
            final_negs: List[int] = []
            initial_neg_docs = [int(doc_id) for doc_id in _slice_with_mod(
                ex_neg['doc_id'],
                offset=offset * negative_size,
                cnt=negative_size
            )]
            # remove duplicates
            unique_neg_docs = list(set(initial_neg_docs))
            final_negs.extend(unique_neg_docs)
            if negative_size > len(final_negs):
                final_negs.extend(random.sample(available_doc_ids, negative_size - len(final_negs)))
            neg_doc_ids.append(final_negs)

    assert len(pos_doc_ids) == len(neg_doc_ids), '{} != {}'.format(len(pos_doc_ids), len(neg_doc_ids))
    assert all(len(doc_ids) == positive_size for doc_ids in pos_doc_ids)
    assert all(len(doc_ids) == negative_size for doc_ids in neg_doc_ids)

    input_doc_ids: List[int] = []
    for pos_ids, neg_ids in zip(pos_doc_ids, neg_doc_ids):
        input_doc_ids.extend(pos_ids)
        input_doc_ids.extend(neg_ids)

    return input_doc_ids

class RetrievalDataLoader:

    def __init__(self, args: Arguments, tokenizer: PreTrainedTokenizerFast):
        self.args = args
        self.positive_size = args.positive_size
        self.negative_size = args.train_n_passages - self.positive_size
        assert self.negative_size > 0
        self.tokenizer = tokenizer
        if self.args.data_name == 'esci':
            passage_file_name = 'passages.jsonl'
        elif self.args.data_name == 'msmarco':
            passage_file_name = 'passages.jsonl.gz'
        corpus_path = os.path.join(args.data_dir, passage_file_name)
        self.corpus: Dataset = load_dataset('json', data_files=corpus_path)['train']
        self.train_dataset, self.eval_dataset = self._get_transformed_datasets()

        # use its state to decide which positives/negatives to sample
        self.trainer: Optional[Trainer] = None

    def _transform_func(self, examples: Dict[str, List]) -> Dict[str, List]:
        current_epoch = int(self.trainer.state.epoch or 0)

        input_doc_ids: List[int] = group_doc_ids(
            examples=examples,
            positive_size=self.positive_size,
            negative_size=self.negative_size,
            offset=current_epoch + self.args.seed,
            available_doc_ids=list(range(len(self.corpus))),
            use_first_positive=self.args.use_first_positive,
            fill_neg_with_random_ids=self.args.fill_neg_with_random_ids
        )

        assert len(input_doc_ids) == len(examples['query']) * self.args.train_n_passages

        if self.args.data_name == 'esci':
            content = 'product_description'
            title = 'product_title'
        elif self.args.data_name == 'msmarco':
            content = 'contents'
            title = 'title'

        input_docs: List[str] = [self.corpus[doc_id][content] for doc_id in input_doc_ids]
        input_titles: List[str] = [self.corpus[doc_id][title] for doc_id in input_doc_ids]

        query_batch_dict = self.tokenizer(examples['query'],
                                          max_length=self.args.q_max_len,
                                          padding=PaddingStrategy.DO_NOT_PAD,
                                          truncation=True)
        doc_batch_dict = self.tokenizer(input_titles,
                                        text_pair=input_docs,
                                        max_length=self.args.p_max_len,
                                        padding=PaddingStrategy.DO_NOT_PAD,
                                        truncation=True)

        merged_dict = {'q_{}'.format(k): v for k, v in query_batch_dict.items()}
        step_size = self.args.train_n_passages
        for k, v in doc_batch_dict.items():
            k = 'd_{}'.format(k)
            merged_dict[k] = []
            for idx in range(0, len(v), step_size):
                merged_dict[k].append(v[idx:(idx + step_size)])

        if self.args.additional_positives > 0 :
            additional_positive_doc_ids: List[int] = group_doc_ids(
                examples=examples,
                positive_size=self.args.additional_positives,  # Load additional k positives
                negative_size=0,  # No negatives
                offset=current_epoch + self.args.seed,
                available_doc_ids=list(range(len(self.corpus))),
                use_first_positive=True,
            )
            assert len(additional_positive_doc_ids) == len(examples['query']) * self.args.additional_positives

            additional_docs: List[str] = [self.corpus[doc_id][content] for doc_id in additional_positive_doc_ids]
            additional_titles: List[str] = [self.corpus[doc_id][title] for doc_id in additional_positive_doc_ids]
            additional_doc_batch_dict = self.tokenizer(additional_titles,
                text_pair=additional_docs,
                max_length=self.args.p_max_len,
                padding=PaddingStrategy.DO_NOT_PAD,
                truncation=True)
            step_size = self.args.additional_positives
            for k, v in additional_doc_batch_dict.items():
                k = 'additional_d_{}'.format(k)
                merged_dict[k] = []
                for idx in range(0, len(v), step_size):
                    merged_dict[k].append(v[idx:(idx + step_size)])

        if self.args.do_kd_biencoder:
            qid_to_doc_id_to_score = {}

            def _update_qid_pid_score(q_id: str, ex: Dict):
                assert len(ex['doc_id']) == len(ex['score'])
                if q_id not in qid_to_doc_id_to_score:
                    qid_to_doc_id_to_score[q_id] = {}
                for doc_id, score in zip(ex['doc_id'], ex['score']):
                    qid_to_doc_id_to_score[q_id][int(doc_id)] = score

            for idx, query_id in enumerate(examples['query_id']):
                _update_qid_pid_score(query_id, examples['positives'][idx])
                _update_qid_pid_score(query_id, examples['negatives'][idx])

            merged_dict['kd_labels'] = []
            for idx in range(0, len(input_doc_ids), step_size):
                qid = examples['query_id'][idx // step_size]
                cur_kd_labels = [qid_to_doc_id_to_score[qid].get(doc_id, 0) for doc_id in input_doc_ids[idx:idx + step_size]]
                merged_dict['kd_labels'].append(cur_kd_labels)
            assert len(merged_dict['kd_labels']) == len(examples['query_id']), \
                '{} != {}'.format(len(merged_dict['kd_labels']), len(examples['query_id']))

        return merged_dict

    def _get_transformed_datasets(self) -> Tuple:
        data_files = {}
        if self.args.train_file is not None:
            data_files["train"] = self.args.train_file.split(',')
        if self.args.validation_file is not None:
            data_files["validation"] = self.args.validation_file
        raw_datasets: DatasetDict = load_dataset('json', data_files=data_files)

        train_dataset, eval_dataset = None, None

        if self.args.do_train:
            if "train" not in raw_datasets:
                raise ValueError("--do_train requires a train dataset")
            train_dataset = raw_datasets["train"]
            if self.args.max_train_samples is not None:
                train_dataset = train_dataset.select(range(self.args.max_train_samples))
            # Log a few random samples from the training set:
            for index in random.sample(range(len(train_dataset)), 3):
                logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
            train_dataset.set_transform(self._transform_func)

        if self.args.do_eval:
            if "validation" not in raw_datasets:
                raise ValueError("--do_eval requires a validation dataset")
            eval_dataset = raw_datasets["validation"]
            eval_dataset.set_transform(self._transform_func)

        return train_dataset, eval_dataset
