# Retrieval eval — `e1-rrf-bgem3-hybrid-k2`

- Thời điểm chạy: `2026-08-21T09:48:43+00:00`
- Số truy vấn: **242** (chấm điểm 209, bỏ qua 33 câu unanswerable)
- Config: `{"retriever": "qdrant-hybrid:rag_bgem3:rrf2-c20", "top_k": 20, "index_config": "configs\\indexing\\bgem3.yaml", "index_fingerprint": "0eaaf9265487eabb25eade5ecb6a85a74ebdbee194b1b5e215befe4bec474932", "collection": "rag_bgem3", "embedding_model": "BAAI/bge-m3", "retrieval_mode": "hybrid", "branch_options": {"candidate_k": 20, "k": 2}, "chunking": {"strategy": "hybrid", "chunk_size": 1000, "chunk_overlap": 100, "separators": ["\n\n", "\n", ". ", " ", ""], "min_chunk_size": 200, "max_chunk_size": 1500, "semantic_buffer_size": 1, "semantic_threshold_percentile": 85.0, "semantic_min_sentences": 3, "hybrid_max_docs_for_semantic": 5, "neighbor_context_chars": 100}, "span_resolution": {"resolved": 209, "kept_chunk_ids": 33, "unmatched_queries": [], "min_overlap_ratio": 0.5, "label_changed": 9}}`
- Môi trường: platform=Windows-11-10.0.26200-SP0, python=3.13.11

## Tổng thể

| Metric | Giá trị |
|---|---:|
| hit_rate@1 | 0.3301 |
| hit_rate@10 | 0.6555 |
| hit_rate@20 | 0.7177 |
| hit_rate@5 | 0.5789 |
| map@20 | 0.3959 |
| mrr | 0.4362 |
| ndcg@10 | 0.4521 |
| precision@1 | 0.3301 |
| precision@10 | 0.0842 |
| precision@20 | 0.0474 |
| precision@5 | 0.1455 |
| recall@1 | 0.2544 |
| recall@10 | 0.5989 |
| recall@20 | 0.6770 |
| recall@5 | 0.5136 |

## Độ trễ truy hồi (ms)

| Phân vị | ms |
|---|---:|
| mean | 30.3 |
| p50 | 27.3 |
| p95 | 46.7 |
| max | 54.4 |
| stdev | 8.0 |

## Theo nhóm truy vấn

| Nhóm | n | hit_rate@1 | hit_rate@10 | hit_rate@20 | hit_rate@5 | map@20 | mrr | ndcg@10 | precision@1 | precision@10 | precision@20 | precision@5 | recall@1 | recall@10 | recall@20 | recall@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adversarial | 34 | 0.2647 | 0.5000 | 0.5882 | 0.4706 | 0.3420 | 0.3492 | 0.3790 | 0.2647 | 0.0559 | 0.0324 | 0.1059 | 0.2500 | 0.5000 | 0.5735 | 0.4706 |
| aggregation | 26 | 0.3846 | 0.8846 | 0.8846 | 0.8462 | 0.4147 | 0.5844 | 0.5105 | 0.3846 | 0.1423 | 0.0769 | 0.2538 | 0.1859 | 0.6026 | 0.6538 | 0.5449 |
| cross_lingual | 43 | 0.0465 | 0.3488 | 0.4651 | 0.2326 | 0.1243 | 0.1280 | 0.1615 | 0.0465 | 0.0372 | 0.0279 | 0.0512 | 0.0465 | 0.3140 | 0.4651 | 0.2093 |
| factoid | 68 | 0.4265 | 0.7941 | 0.8529 | 0.6765 | 0.5373 | 0.5397 | 0.5950 | 0.4265 | 0.0809 | 0.0434 | 0.1382 | 0.4191 | 0.7941 | 0.8529 | 0.6765 |
| multi_hop | 34 | 0.5588 | 0.7647 | 0.7941 | 0.7353 | 0.5208 | 0.6216 | 0.5840 | 0.5588 | 0.1382 | 0.0750 | 0.2412 | 0.2745 | 0.6765 | 0.7353 | 0.5931 |
| table_lookup | 4 | 0.0000 | 0.5000 | 0.5000 | 0.5000 | 0.1875 | 0.1875 | 0.2654 | 0.0000 | 0.0500 | 0.0250 | 0.1000 | 0.0000 | 0.5000 | 0.5000 | 0.5000 |

## Theo ngôn ngữ

| Nhóm | n | hit_rate@1 | hit_rate@10 | hit_rate@20 | hit_rate@5 | map@20 | mrr | ndcg@10 | precision@1 | precision@10 | precision@20 | precision@5 | recall@1 | recall@10 | recall@20 | recall@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| en | 82 | 0.3415 | 0.6829 | 0.7195 | 0.5732 | 0.4105 | 0.4478 | 0.4716 | 0.3415 | 0.0988 | 0.0530 | 0.1634 | 0.2337 | 0.6260 | 0.6789 | 0.5102 |
| vi | 127 | 0.3228 | 0.6378 | 0.7165 | 0.5827 | 0.3865 | 0.4286 | 0.4394 | 0.3228 | 0.0748 | 0.0437 | 0.1339 | 0.2677 | 0.5814 | 0.6759 | 0.5157 |

> Câu thuộc nhóm `unanswerable` không có tài liệu liên quan nên bị loại khỏi
> mọi metric xếp hạng. Chúng được đo riêng bằng refusal correctness (W5-02).
