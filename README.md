# Contrastive Learning for Dense Retrieval: Batch Composition, Sampling, and Experiments

This README documents the setup, data statistics, negative sampling strategies, and experimental results for contrastive learning in dense retrieval tasks. The experiments explore various negative sampling methods, batch compositions, and loss functions.

---

## Table of Contents

- [Batch Composition](#batch-composition)
- [Negative Sampling Strategies](#negative-sampling-strategies)
- [Data Statistics](#data-statistics)
- [Experimental Setup](#experimental-setup)
- [Results](#results)
- [Batch Size Analysis](#batch-size-analysis)
- [References](#references)

---

## Batch Composition

| Type         | qp (query-positive)         | pq (positive-query)         | qq (query-query)       | pp (positive-positive) | Total   |
|--------------|----------------------------|-----------------------------|------------------------|-----------------------|---------|
| n_negatives  | batch_size × gpus × train_n_passages | batch_size × gpus           | batch_size × gpus      | batch_size × gpus     | qp+pq+qq+pp |
| Example (batch_size=16, gpus=8, train_n_passages=16) | 2048 (84%)                  | 128 (5%)               | 128 (5%)              | 128 (5%)             | 2432    |

---

## Data Statistics

| Split   | # Positive Items per Query | # Negative Items per Query |
|---------|---------------------------|---------------------------|
| Train   | 8.70                      | 11.39                     |
| Test    | 8.90                      | 11.39                     |

### Item Uniqueness

| Setting         | Unique Items (%) | Duplicated Items (%) |
|-----------------|-----------------|----------------------|
| bs=128, ls=16   | 66.72           | 33.28                |
| bs=128, ls=8    | 91.58           | 8.42                 |
| bs=248, ls=8    | 89.56           | 10.44                |
| bs=248, ls=8, random negatives | 91.92 | 8.08           |

---

## Negative Sampling Strategies

- **Hard negatives**: Difficult negatives identified by the model.
- **Random negatives**: Randomly sampled negatives.
- **Other negatives**: Additional negatives not in `qp`.

**Parameters:**
- `n_hard_negatives`: Number of hard negatives per query.
- `n_other_negatives`: Number of other negatives per query.
- `embedding_dedup_threshold`: Threshold for deduplicating by embedding similarity.
- `fn_kk_threshold`: Threshold for filtering negatives by nearest neighbor similarity.

---

## Experimental Setup

- **Epochs**: 40
- **num_positives**: 1
- **Learning Rate**: 2e-5
- **Batch Size (bs)**: 416 (Gradient Checkpointing enabled, A100 40G)
- **Initial Model**: `simlm-base-msmarco-finetuned`
- **Loss Functions**: Cross-entropy, Softmax

---

## Results

| Model/Setup | Loss | NDCG@50 | R@100 | R@500 | NDCG@50 (qp only) | R@100 (qp only) | R@500 (qp only) |
|-------------|------|---------|-------|-------|-------------------|-----------------|-----------------|
| simlm-base-msmarco-finetuned | - | 0.34433 | 0.52389 | 0.68171 | - | - | - |
| BIBERT (init: simlm-base-msmarco-finetuned) | Cross-entropy | - | - | - | - | - | - |
| BIBERT (init: simlm-base-msmarco-finetuned) | Softmax | 0.43003 | 0.632 | 0.79 | 0.43528 | 0.63732 | 0.79256 |
| M1: BIBERT (fill negatives by random docs) | Softmax | 0.42953 | 0.63168 | 0.78995 | 0.43589 | 0.63796 | 0.79313 |
| M1 + n_hard_negatives=30, n_other_negatives=5 | - | - | - | - | - | - | - |
| M1 + n_hard_negatives=30, n_other_negatives=5, embedding_dedup_threshold=0.999 | - | - | - | - | - | - | - |
| M1 + n_hard_negatives=30, n_other_negatives=5, fn_kk_threshold=0.85 | - | - | - | - | - | - | - |
| M1 + n_hard_negatives=30, n_other_negatives=5, embedding_dedup_threshold=0.999, fn_kk_threshold=0.85 | - | - | - | - | - | - | - |
| Offline negatives from ANCE (based on M1) | Softmax | - | - | - | - | - | - |
| ... with n_hard_negatives=30, n_other_negatives=5, fn_kk_threshold=0.85 | - | - | - | - | - | - | - |
| ANCE* (fn_kk_threshold=0.85 based on M1) | Softmax | - | - | - | - | - | - |
| ... with n_hard_negatives=30, n_other_negatives=5, fn_kk_threshold=0.85 | - | - | - | - | - | - | - |

---

## Batch Size Analysis

- **Batch sizes studied:** 16, 32, 64 (GC), 256 (GC), 416 (GC)
- **Observation:** Shall we compute % False Negatives (FN) from batch? Metrics increase, then decrease with batch size.

| Batch Size | NDCG@50 | R@100 | R@500 |
|------------|---------|-------|-------|
| 16         | -       | -     | -     |
| 32         | -       | -     | -     |
| 64 (gc)    | -       | -     | -     |
| 256 (gc)   | -       | -     | -     |
| 416 (gc)   | -       | -     | -     |

---

## References

- simlm-base-msmarco-finetuned
- BIBERT
- ANCE



