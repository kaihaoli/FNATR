import os
import random
import tqdm
import json

from typing import Dict, List, Any
from datasets import load_dataset, Dataset
from dataclasses import dataclass, field

from logger_config import logger
from config import Arguments
from utils import save_json_to_file


@dataclass
class ScoredDoc:
    qid: str
    pid: str
    rank: int
    score: float = field(default=-1)


def load_qrels(path: str) -> Dict[str, Dict[str, int]]:
    if path.endswith('.txt'):
        # qid -> pid -> score
        qrels = {}
        for line in open(path, 'r', encoding='utf-8'):
            qid, _, pid, score = line.strip().split('\t')
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][pid] = int(score)
    elif path.endswith('.jsonl'):
        qrels = {}
        for line in open(path, 'r', encoding='utf-8'):
            d = json.loads(line)
            qid = str(d['query_id'])
            qrels[qid] = {}
            for pid, score in zip(d['positives']['doc_id'], d['positives']['score']):
                pid = str(pid)
                qrels[qid][pid] = int(score)
            for pid, score in zip(d['negatives']['doc_id'], d['negatives']['score']):
                pid = str(pid)
                qrels[qid][pid] = int(score)

    logger.info('Load {} queries {} qrels from {}'.format(len(qrels), sum(len(v) for v in qrels.values()), path))
    return qrels


def load_queries(path: str, task_type: str = 'ir') -> Dict[str, str]:

    if path.endswith('.tsv'):
        if task_type == 'qa':
            qid_to_query = load_query_answers(path)
            qid_to_query = {k: v['query'] for k, v in qid_to_query.items()}
        elif task_type == 'ir':
            qid_to_query = {}
            for line in open(path, 'r', encoding='utf-8'):
                qid, query = line.strip().split('\t')
                qid_to_query[qid] = query
        else:
            raise ValueError('Unknown task type: {}'.format(task_type))
    elif path.endswith('.jsonl'):
        qid_to_query = {}
        for line in open(path, 'r', encoding='utf-8'):
            d = json.loads(line)
            qid_to_query[d['query_id']] = d['query']

    logger.info('Load {} queries from {}'.format(len(qid_to_query), path))
    return qid_to_query


def normalize_qa_text(text: str) -> str:
    # TriviaQA has some weird formats
    # For example: """What breakfast food gets its name from the German word for """"stirrup""""?"""
    while text.startswith('"') and text.endswith('"'):
        text = text[1:-1].replace('""', '"')
    return text


def get_question_key(question: str) -> str:
    # For QA dataset, we'll use normalized question strings as dict key
    return question


def load_query_answers(path: str) -> Dict[str, Dict[str, Any]]:
    assert path.endswith('.tsv')

    qid_to_query = {}
    for line in open(path, 'r', encoding='utf-8'):
        query, answers = line.strip().split('\t')
        query = normalize_qa_text(query)
        answers = normalize_qa_text(answers)
        qid = get_question_key(query)
        if qid in qid_to_query:
            logger.warning('Duplicate question: {} vs {}'.format(query, qid_to_query[qid]['query']))
            continue

        qid_to_query[qid] = {}
        qid_to_query[qid]['query'] = query
        qid_to_query[qid]['answers'] = list(eval(answers))

    logger.info('Load {} queries from {}'.format(len(qid_to_query), path))
    return qid_to_query


def load_corpus(path: str) -> Dataset:
    assert path.endswith('.jsonl') or path.endswith('.jsonl.gz')

    # two fields: id, contents
    corpus = load_dataset('json', data_files=path)['train']
    logger.info('Load {} documents from {} with columns {}'.format(len(corpus), path, corpus.column_names))
    logger.info('A random document: {}'.format(random.choice(corpus)))
    return corpus


def load_predictions(path: str) -> Dict[str, List[ScoredDoc]]:
    assert path.endswith('.txt')

    qid_to_scored_doc = {}
    for line in tqdm.tqdm(open(path, 'r', encoding='utf-8'), desc='load prediction', mininterval=3):
        fs = line.strip().split('\t')
        qid, pid, rank = fs[:3]
        rank = int(rank)
        score = round(1 / rank, 4) if len(fs) == 3 else float(fs[3])

        if qid not in qid_to_scored_doc:
            qid_to_scored_doc[qid] = []
        scored_doc = ScoredDoc(qid=qid, pid=pid, rank=rank, score=score)
        qid_to_scored_doc[qid].append(scored_doc)

    qid_to_scored_doc = {qid: sorted(scored_docs, key=lambda sd: sd.rank)
                         for qid, scored_docs in qid_to_scored_doc.items()}

    logger.info('Load {} query predictions from {}'.format(len(qid_to_scored_doc), path))
    return qid_to_scored_doc


def save_preds_to_output_format(preds: Dict[str, List[ScoredDoc]], out_path: str):
    with open(out_path, 'w', encoding='utf-8') as writer:
        for qid in preds:
            for idx, scored_doc in enumerate(preds[qid]):
                writer.write('{}\t{}\t{}\t{}\n'.format(qid, scored_doc.pid, idx + 1, round(scored_doc.score, 3)))
    logger.info('Successfully saved to {}'.format(out_path))


def save_to_readable_format(in_path: str, corpus: Dataset):
    out_path = '{}/readable_{}'.format(os.path.dirname(in_path), os.path.basename(in_path))
    dataset: Dataset = load_dataset('json', data_files=in_path)['train']

    max_to_keep = 5

    def _create_readable_field(samples: Dict[str, List]) -> List:
        readable_ex = []
        for idx in range(min(len(samples['doc_id']), max_to_keep)):
            doc_id = samples['doc_id'][idx]
            readable_ex.append({'doc_id': doc_id,
                                'title': corpus[int(doc_id)].get('title', ''),
                                'contents': corpus[int(doc_id)]['contents'],
                                'score': samples['score'][idx]})
        return readable_ex

    def _mp_func(ex: Dict) -> Dict:
        ex['positives'] = _create_readable_field(ex['positives'])
        ex['negatives'] = _create_readable_field(ex['negatives'])
        return ex
    dataset = dataset.map(_mp_func, num_proc=8)

    dataset.to_json(out_path, force_ascii=False, lines=False, indent=4)
    logger.info('Done convert {} to readable format in {}'.format(in_path, out_path))


if __name__ == '__main__':
    load_qrels('./data/msmarco/dev_qrels.txt')
    load_queries('./data/msmarco/dev_queries.tsv')
    corpus = load_corpus('./data/msmarco/passages.jsonl.gz')
    preds = load_msmarco_predictions('./data/bm25.msmarco.txt')
