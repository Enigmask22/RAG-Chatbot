# Retrieval eval — `bgem3-ctx-rr-c50-w025`

- Thời điểm chạy: `2026-09-03T15:38:43+00:00`
- Số truy vấn: **242** (chấm điểm 209, bỏ qua 33 câu unanswerable)
- Config: `{"retriever": "reranked[qdrant-hybrid:rag_bgem3_ctx:rrf1-c20-w1:0.25]:BAAI/bge-reranker-v2-m3@cuda:L512:float16:n50", "top_k": 20, "index_config": "configs\\indexing\\bgem3-contextual.yaml", "index_fingerprint": "ff0828fecae7998ad2bf04a389fbe2194ee62506468a6ceb6e9660a981eac52f", "collection": "rag_bgem3_ctx", "embedding_model": "BAAI/bge-m3", "retrieval_mode": "reranked", "branch_options": {"k": 1, "candidate_k": 20, "weights": [1.0, 0.25], "base": "hybrid"}, "chunking": {"strategy": "hybrid", "size_unit": "chars", "chunk_size": 1000, "chunk_overlap": 100, "separators": ["\n\n", "\n", ". ", " ", ""], "min_chunk_size": 200, "max_chunk_size": 1500, "semantic_buffer_size": 1, "semantic_threshold_percentile": 85.0, "semantic_min_sentences": 3, "hybrid_max_docs_for_semantic": 5, "parent_size_multiple": 4, "structure_merge_short_sections": true, "neighbor_context_chars": 100}, "span_resolution": {"resolved": 209, "kept_chunk_ids": 33, "unmatched_queries": [], "min_overlap_ratio": 0.5, "label_changed": 9}}`
- Môi trường: platform=Windows-11-10.0.26200-SP0, python=3.13.11

## Tổng thể

| Metric | Giá trị |
|---|---:|
| hit_rate@1 | 0.6220 |
| hit_rate@10 | 0.8325 |
| hit_rate@20 | 0.8469 |
| hit_rate@5 | 0.8230 |
| map@20 | 0.6636 |
| mrr | 0.7047 |
| ndcg@10 | 0.7079 |
| precision@1 | 0.6220 |
| precision@10 | 0.1105 |
| precision@20 | 0.0577 |
| precision@5 | 0.2153 |
| recall@1 | 0.5000 |
| recall@10 | 0.8022 |
| recall@20 | 0.8246 |
| recall@5 | 0.7847 |

## Độ trễ truy hồi (ms)

| Phân vị | ms |
|---|---:|
| mean | 686.8 |
| p50 | 691.6 |
| p95 | 759.0 |
| max | 873.7 |
| stdev | 52.6 |

## Theo nhóm truy vấn

| Nhóm | n | hit_rate@1 | hit_rate@10 | hit_rate@20 | hit_rate@5 | map@20 | mrr | ndcg@10 | precision@1 | precision@10 | precision@20 | precision@5 | recall@1 | recall@10 | recall@20 | recall@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adversarial | 34 | 0.5294 | 0.7647 | 0.7647 | 0.7647 | 0.6127 | 0.6201 | 0.6530 | 0.5294 | 0.0824 | 0.0412 | 0.1647 | 0.5000 | 0.7647 | 0.7647 | 0.7647 |
| aggregation | 26 | 0.6923 | 0.8846 | 0.9231 | 0.8846 | 0.6254 | 0.7855 | 0.6988 | 0.6923 | 0.1808 | 0.1019 | 0.3462 | 0.3141 | 0.7692 | 0.8590 | 0.7436 |
| cross_lingual | 43 | 0.4884 | 0.6512 | 0.6744 | 0.6279 | 0.5411 | 0.5401 | 0.5663 | 0.4884 | 0.0744 | 0.0384 | 0.1395 | 0.4535 | 0.6512 | 0.6744 | 0.6163 |
| factoid | 68 | 0.6618 | 0.9265 | 0.9265 | 0.9118 | 0.7654 | 0.7642 | 0.8057 | 0.6618 | 0.0941 | 0.0471 | 0.1853 | 0.6618 | 0.9265 | 0.9265 | 0.9118 |
| multi_hop | 34 | 0.7647 | 0.9412 | 0.9706 | 0.9412 | 0.7140 | 0.8407 | 0.7776 | 0.7647 | 0.1706 | 0.0897 | 0.3353 | 0.3775 | 0.8431 | 0.8824 | 0.8284 |
| table_lookup | 4 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.0500 | 0.0250 | 0.1000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |

## Theo ngôn ngữ

| Nhóm | n | hit_rate@1 | hit_rate@10 | hit_rate@20 | hit_rate@5 | map@20 | mrr | ndcg@10 | precision@1 | precision@10 | precision@20 | precision@5 | recall@1 | recall@10 | recall@20 | recall@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| en | 82 | 0.6341 | 0.8780 | 0.8902 | 0.8659 | 0.6754 | 0.7310 | 0.7312 | 0.6341 | 0.1268 | 0.0659 | 0.2439 | 0.4675 | 0.8415 | 0.8577 | 0.8130 |
| vi | 127 | 0.6142 | 0.8031 | 0.8189 | 0.7953 | 0.6559 | 0.6877 | 0.6928 | 0.6142 | 0.1000 | 0.0524 | 0.1969 | 0.5210 | 0.7769 | 0.8031 | 0.7664 |

> Câu thuộc nhóm `unanswerable` không có tài liệu liên quan nên bị loại khỏi
> mọi metric xếp hạng. Chúng được đo riêng bằng refusal correctness (W5-02).
