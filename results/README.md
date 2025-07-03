# Data Statistics and Experimental Results for Dense Retrieval

This document summarizes key data statistics, analysis of item uniqueness, and experimental results for dense retrieval tasks using contrastive learning approaches. The experiments focus on the effect of negative sampling strategies, deduplication thresholds, and batch size on retrieval metrics.

---

## Data Statistics

### Items per Query

| Split | # Positive Items | # Negative Items |
|-------|------------------|------------------|
| Train | 8.70             | 11.39            |
| Test  | 8.90             | 11.39            |

---

## Item Uniqueness Analysis

The following table shows the percentage of unique and duplicated items per batch, under different batch size (`bs`) and list size (`ls`) configurations, and when sampling negatives randomly.

| Setting                | Unique Items (%) | Duplicated Items (%) |
|------------------------|-----------------|----------------------|
| bs=128, ls=16          | 66.72           | 33.28                |
| bs=128, ls=8           | 91.58           | 8.42                 |
| bs=248, ls=8           | 89.56           | 10.44                |
| bs=248, ls=8 (random negatives) | 91.92   | 8.08                 |
| bs=416, ls=8           | 91.35           | 8.65                 |

---

## Experimental Setup

- **Epochs:** 40
- **Number of positives per query:** 1
- **Learning Rate:** 2e-5
- **Batch Size:** 416 (GC enabled, A100 40G)

---

## Results

### Metrics Reported

- NDCG@50
- R@100
- R@500
- MRR
- R@50
- R@20
- R@100 (MS MARCO / NQ)

### Model Performance

| Model / Setting                          | NDCG@50 | R@100 | R@500 | MRR     | R@50    | R@500   | R@20    | R@100   |
|------------------------------------------|---------|-------|-------|---------|---------|---------|---------|---------|
| **Initial model: simlm-base-msmarco-finetuned** | 0.34433 | 0.52389| 0.68171|         |         |         |         |         |
| **BIBERT (initial simlm-base-msmarco-finetuned)** |         |        |        |         |         |         |         |         |
| all qips                                 | 0.43528 | 0.63732|0.79256|         |         |         |         |         |
| **M1: BIBERT, fill negative by random docs** |         |        |        |         |         |         |         |         |
| all qips                                 | 0.43589 | 0.63796|0.79313| 37.3042 | 0.85614 | 0.97375 |         |         |
| top 100 qips                             | 0.44181 | 0.64693|0.79645| 37.3516 | 0.85714 | 0.97346 |         |         |
| dedup threshold = 0.999                  | 0.43810 | 0.64305|0.79471|         |         |         |         |         |
| top 100 qips, ii-sim=0.85                | 0.44436 | 0.64849|0.79648| 37.2479 | 0.85986 | 0.97381 |         |         |
| top 100 qips, dedup=0.999, ii-sim=0.85   | 0.44081 | 0.64528|0.79630|         |         |         |         |         |
| **Offline negatives from ANCE (top50) based on M1** |         |        |        |         |         |         |         |         |
| all qips                                 | 0.44287 | 0.64649|0.79649|         |         |         |         |         |
| top 100 qips                             | 0.44368 | 0.64965|0.79762|         |         |         |         |         |
| ii-sim=0.85                              | 0.44518 | 0.64913|0.79743|         |         |         |         |         |
| **ANCE* (fn_kk_threshold=0.85) based on M1** |         |        |        |         |         |         |         |         |
| all qips                                 | 0.44388 | 0.64772|0.79852|         |         |         |         |         |
| top 100 qips                             | 0.44325 | 0.64826|0.79664|         |         |         |         |         |
| top 100 qips, ii-sim=0.85                | 0.44644 | 0.65132|0.79818|         |         |         |         |         |
| **ANCE* (fn_kk_threshold=0.85) based on M1, n_hard_neg** |         |        |        |         |         |         |         |         |
| n_hard_neg=30                            | 0.44272 | 0.64610|0.79022|         |         |         |         |         |
| n_hard_neg=50                            | 0.44503 | 0.64885|0.79419|         |         |         |         |         |
| n_hard_neg=100                           | 0.44644 | 0.65132|0.79818|         |         |         |         |         |
| n_hard_neg=150                           | 0.44461 | 0.64943|0.79746|         |         |         |         |         |
| **BIBERT, fill negative by random docs, n_hard_neg=100** |         |        |        |         |         |         |         |         |
| fn_kk_threshold=0.80                     | 0.44024 | 0.64425|0.79664|         |         |         |         |         |
| fn_kk_threshold=0.83                     | 0.44085 | 0.64534|0.79627|         |         |         |         |         |
| fn_kk_threshold=0.84                     | 0.44070 | 0.64516|0.79637|         |         |         |         |         |
| fn_kk_threshold=0.85                     | 0.44436 | 0.64849|0.79648|         |         |         |         |         |
| fn_kk_threshold=0.86                     | 0.44053 | 0.64547|0.79542|         |         |         |         |         |
| fn_kk_threshold=0.87                     | 0.44087 | 0.64533|0.79652|         |         |         |         |         |

---

## Effect of Batch Size

- Batch sizes studied: 16, 64, 256, (max)
- Observation: Metrics generally increase with batch size, then decrease. Should % of false negatives (FN) from batch be computed?

---

## Notes

- "qips" refers to query-item pairs.
- "ii-sim" is the item-item similarity threshold used for deduplication or negative filtering.
- "fn_kk_threshold" is used for filtering negatives based on nearest neighbor similarity.
- Experiments use MS MARCO and NQ benchmarks.

---

For more details on the experimental setup, code, and full logs, please refer to the repository files and scripts.
