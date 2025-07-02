import pandas as pd
import json
import jsonlines
from tqdm import tqdm
import numpy as np
import os
import argparse

def make_corpus(output_dir, df_task_en):
    from clean import clean
    seen_product = set()
    with jsonlines.open(os.path.join(output_dir, "passages_raw.jsonl"), 'w') as w:
        for index, row in tqdm(df_task_en.iterrows()):
            product_id = row['product_id'].strip()
            doc_id = row['doc_id']
            if product_id not in seen_product:
                seen_product.add(product_id)
            else:
                continue
            
            product_title = "" if row["product_title"] is None or row["product_title"].lower(
            ) == 'none' else row["product_title"].strip()
            product_title = clean(product_title)

            product_description = "" if row["product_description"] is None or row["product_description"].lower(
            ) == 'none' else row["product_description"].strip()
            product_description = clean(product_description)
            
            product_bullet_point = "" if row["product_bullet_point"] is None or row["product_bullet_point"].lower(
            ) == 'none' else row["product_bullet_point"].strip()
            product_bullet_point = clean(product_bullet_point)

            product_brand = "" if row["product_brand"] is None or row["product_brand"].lower(
            ) == 'none' else row["product_brand"].strip()
            product_brand = clean(product_brand)

            product_color = "" if row["product_color"] is None or row["product_color"].lower(
            ) == 'none' else row["product_color"].strip()
            product_color = clean(product_color)
            
            instance = {
                "doc_id": doc_id,
                "product_id": product_id,
                "product_title": product_title,
                "product_description": product_description,
                "product_bullet_point": product_bullet_point,
                "product_brand": product_brand,
                "product_color": product_color
            }
            """
            instance = {
                'id': doc_id,
                'contents': product_title + '\n' + product_description,
            }
            """
            w.write(instance)
    print("corpus size: ", len(seen_product))

def make_data(output_dir, df_task_en):
    LABEL_MAP = {
        'E': 1.0,  # Exact
        'S': 0.1,  # Substitute
        'C': 0.01, # Complement
        'I': 0.0,  # Irrelevant
    }
    #f_train = open(
    #    os.path.join(output_dir, 'train.jsonl'), 'w', encoding='utf-8')
    #f_test = open(
    #    os.path.join(output_dir, 'test.jsonl'), 'w', encoding='utf-8')
    
    train_queries = {}
    test_queries = {}

    for index, row in tqdm(df_task_en.iterrows()):
        doc_id = row['doc_id']
        product_id = row["product_id"].strip()

        query_id = row["query_id"]
        query = row["query"].strip()
        
        tag = row['split']
        seen_queries = train_queries if tag == 'train' else test_queries

        if query_id not in seen_queries:
            seen_queries[query_id] = {
                'query': query, 
                'query_id': query_id, 
                'positives': {'doc_id': [], 'score': [] }, 
                'negatives': {'doc_id': [], 'score': [] },
            }

        if row["esci_label"] == 'E':
            if doc_id not in seen_queries[query_id]['positives']['doc_id']:
                seen_queries[query_id]['positives']['doc_id'].append(doc_id)
                seen_queries[query_id]['positives']['score'].append(1)
        else:
            if doc_id not in seen_queries[query_id]['negatives']['doc_id']:
                seen_queries[query_id]['negatives']['doc_id'].append(doc_id)
                seen_queries[query_id]['negatives']['score'].append(LABEL_MAP[row["esci_label"]])

    num_docs = {'train': {'positives' : [], 'negatives': []}, 'test': {'positives' : [], 'negatives': []}, }

    for query_id, qrel_detail in train_queries.items():
        if len(qrel_detail['positives']['doc_id']) == 0:  # no positives
            continue
        if len(qrel_detail['negatives']['doc_id']) == 0:  # no negatives
            continue
        num_docs['train']['positives'].append(len(qrel_detail['positives']['doc_id']))
        num_docs['train']['negatives'].append(len(qrel_detail['negatives']['doc_id']))
        f_train.write(json.dumps(qrel_detail, ensure_ascii=False))
        f_train.write("\n")

    for query_id, qrel_detail in test_queries.items():
        if len(qrel_detail['positives']['doc_id']) == 0:  # no positives
            continue
        if len(qrel_detail['negatives']['doc_id']) == 0:  # no negatives
            continue
        num_docs['test']['positives'].append(len(qrel_detail['positives']['doc_id']))
        num_docs['test']['negatives'].append(len(qrel_detail['negatives']['doc_id']))
        f_test.write(json.dumps(qrel_detail, ensure_ascii=False))
        f_test.write("\n")

    f_train.close()
    f_test.close()
    for k in num_docs['train']:
        print('train -- avg for {}: '.format(k), np.mean(num_docs['train'][k]))
        print('train -- min for {}: '.format(k), np.min(num_docs['train'][k]))
        print('train -- max for {}: '.format(k), np.max(num_docs['train'][k]))
    for k in num_docs['test']:
        print('test -- avg for {}: '.format(k), np.mean(num_docs['test'][k]))
        print('test -- min for {}: '.format(k), np.min(num_docs['test'][k]))
        print('test -- max for {}: '.format(k), np.max(num_docs['test'][k]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess ESCI data")
    parser.add_argument("--data_dir", type=str, default='/home/j0l07il/esci-data/shopping_queries_dataset', help="Path to the amazon directory.")
    parser.add_argument("--output_dir", type=str, default='/home/j0l07il/esci-train-data', help="Path to the amazon directory.")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df_examples = pd.read_parquet(os.path.join(args.data_dir, 'shopping_queries_dataset_examples.parquet'))
    df_products = pd.read_parquet(os.path.join(args.data_dir, 'shopping_queries_dataset_products.parquet'))
    df_examples_products = pd.merge(
         df_examples,
         df_products,
         how='left',
         left_on=['product_locale', 'product_id'],
         right_on=['product_locale', 'product_id']
    )
    
    df_task_1 = df_examples_products[df_examples_products["small_version"] == 1]
    df_task_en = df_task_1[df_task_1["product_locale"] == 'us']
    df_task_en['doc_id'] = pd.factorize(df_task_en['product_id'])[0]
    df_task_1_train = df_task_en[df_task_en["split"] == "train"]
    df_task_1_test = df_task_en[df_task_en["split"] == "test"]
    print('train data:', len(df_task_1_train))
    print('test data:', len(df_task_1_test))
    print('all data:', len(df_task_en))

    # step1: preprocess corpus
    make_corpus(args.output_dir, df_task_en)
    # step2: construct qrels
    make_data(args.output_dir, df_task_en)

