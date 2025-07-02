import json
import argparse
import os
import glob
import torch
import torch.nn.functional as F
from tqdm import tqdm

import numpy as np

def calc_stats(lst):
    if not lst:
        return 0, 0.0, 0
    arr = np.array(lst)
    return int(arr.min()), float(arr.mean()), int(arr.max())

def print_stats_train_lookup(train_lookup):
    pos_counts = []
    neg_counts = []
    for entry in train_lookup.values():
        pos_counts.append(len(entry['positives']['doc_id']))
        neg_counts.append(len(entry['negatives']['doc_id']))
    pos_min, pos_avg, pos_max = calc_stats(pos_counts)
    neg_min, neg_avg, neg_max = calc_stats(neg_counts)
    print(f"Positives per query: min={pos_min}, avg={pos_avg:.2f}, max={pos_max}")
    print(f"Negatives per query: min={neg_min}, avg={neg_avg:.2f}, max={neg_max}")

def print_stats_filtered(filtered_out_per_query):
    filtered_counts = [len(entry["filtered_neg_doc_ids"]) for entry in filtered_out_per_query]
    filt_min, filt_avg, filt_max = calc_stats(filtered_counts)
    print(f"Filtered (skipped) negatives per query: min={filt_min}, avg={filt_avg:.2f}, max={filt_max}")

def get_filtered_out_path(output_file):
    base, ext = os.path.splitext(output_file)
    return f"{base}_filtered{ext}"

def get_all_shards_path(directory):
    path_list = glob.glob('{}/shard_*_*'.format(directory))
    assert len(path_list) > 0

    def _parse_worker_idx_shard_idx(p):
        worker_idx, shard_idx = [int(f) for f in os.path.basename(p).split('_')[-2:]]
        return worker_idx, shard_idx

    path_list = sorted(path_list, key=lambda path: _parse_worker_idx_shard_idx(path))
    return path_list

def load_all_embeddings(directory):
    embeds_path_list = get_all_shards_path(directory)
    embeddings = {}
    doc_id_offset = 0
    for shard_path in embeds_path_list:
        shard = torch.load(shard_path, map_location="cpu")
        for i in range(shard.shape[0]):
            embeddings[doc_id_offset + i] = shard[i]
        doc_id_offset += shard.shape[0]
    return embeddings

def filter_neg_by_similarity(candidate_neg_ids, pos_doc_ids, embeddings, threshold=0.85):
    if threshold == 1:
        return [{"neg_id": x } for x in candidate_neg_ids], []

    pos_embeds = []
    for pid in pos_doc_ids:
        emb = embeddings.get(pid)
        if emb is not None:
            pos_embeds.append(emb.unsqueeze(0))
    if not pos_embeds:
        return candidate_neg_ids, []
    pos_embeds = torch.cat(pos_embeds, dim=0)

    filtered_neg_ids = []
    filtered_out_info = []
    for neg_id in candidate_neg_ids:
        neg_emb = embeddings.get(neg_id)
        if neg_emb is None:
            continue
        sims = F.cosine_similarity(pos_embeds, neg_emb.unsqueeze(0))
        max_sim = sims.max().item()
        if max_sim <= threshold:
            filtered_neg_ids.append({"neg_id": neg_id, "max_similarity": max_sim})
        else:
            filtered_out_info.append({"neg_id": neg_id, "max_similarity": max_sim})
    return filtered_neg_ids, filtered_out_info

def main(
    train_file_path, nn_path, output_path, top_k, embeds_dir, threshold=0.85
):

    embeddings = load_all_embeddings(embeds_dir)
    print(f"Loaded {len(embeddings)} embeddings.")

    train_lookup = {}
    with open(train_file_path, 'r') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                train_lookup[entry['query_id']] = entry

    nn_dict = {}
    with open(nn_path, 'r') as f:
        for line in f:
            if line.strip():
                qid, doc_id, rank, score = line.strip().split('\t')
                qid = int(qid)
                doc_id = int(doc_id)
                rank = int(rank)
                score = float(score)
                nn_dict.setdefault(qid, []).append((doc_id, rank, score))

    filtered_out_per_query = []
    for qid, neighbors in tqdm(nn_dict.items(), desc="Augmenting queries"):
        entry = train_lookup[qid]
        positive_set = set(entry['positives']['doc_id'])
        negative_set = set(entry['negatives']['doc_id'])
        neg_docs, neg_scores = entry['negatives']['doc_id'], entry['negatives']['score']

        candidates = []
        for doc_id, rank, score in sorted(neighbors, key=lambda x: x[1]):
            if doc_id not in positive_set and doc_id not in negative_set:
                candidates.append(doc_id)
                if len(candidates) >= top_k * 2:
                    break

        filtered_candidates, filtered_out_info = filter_neg_by_similarity(
            candidates, entry['positives']['doc_id'], embeddings, threshold
        )
        # for RCA
        if filtered_out_info:
            filtered_out_per_query.append({
                "query_id": qid,
                "filtered_neg_doc_ids": [item["neg_id"] for item in filtered_out_info],
                "filtered_neg_similarities": [item["max_similarity"] for item in filtered_out_info]
            })
        count = 0
        for item in filtered_candidates:
            doc_id = item['neg_id']
            neg_docs.append(doc_id)
            neg_scores.append(-1)
            negative_set.add(doc_id)
            count += 1
            if count >= top_k:
                break

    with open(output_path, 'w') as fout:
        for entry in train_lookup.values():
            fout.write(json.dumps(entry) + '\n')

    filtered_out_path = get_filtered_out_path(output_path)
    with open(filtered_out_path, 'w') as fout:
        for entry in filtered_out_per_query:
            fout.write(json.dumps(entry) + '\n')
    print(f"Filtered (skipped) negatives written to: {filtered_out_path}")

    print_stats_train_lookup(train_lookup)
    print_stats_filtered(filtered_out_per_query)



if __name__ == "__main__":
    """
    python augment_negative.py \
            --train_file /data/train.jsonl \
            --nn_file ../../results/biencoder_1pos_gc_epoch40_ls8_bs416_qkonly_fillrand/train.esci.txt \
            --output_file /data/train_w_negatives_fn085.jsonl \
            --embeds_dir ../../results/biencoder_1pos_gc_epoch40_ls8_bs416_qkonly_fillrand/ \
            --threshold 0.85
    """
    parser = argparse.ArgumentParser(description="Augment negatives with NN and filter false negative by high cosine similarity between candidate and positives.")
    parser.add_argument("--train_file", required=True, help="Path to input training data (JSONL).")
    parser.add_argument("--nn_file", required=True, help="Path to nearest neighbor file (tab-separated).")
    parser.add_argument("--output_file", required=True, help="Path to output augmented JSONL file.")
    parser.add_argument("--top_k", type=int, default=50, help="Number of negatives per query")
    parser.add_argument("--embeds_dir", type=str, help="Directory of embedding shard files")
    parser.add_argument("--threshold", type=float, default=0.85, help="Cosine similarity threshold for filtering negatives.")
    args = parser.parse_args()

    main(
        args.train_file,
        args.nn_file,
        args.output_file,
        args.top_k,
        args.embeds_dir,
        args.threshold
    )
