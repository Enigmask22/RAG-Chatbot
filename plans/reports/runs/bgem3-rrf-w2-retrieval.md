# Retrieval eval — `bgem3-rrf-w2`

- Thời điểm chạy: `2026-08-20T15:38:48+00:00`
- Số truy vấn: **242** (chấm điểm 209, bỏ qua 33 câu unanswerable)
- Config: `{"retriever": "qdrant-hybrid:rag_bgem3:rrf60-c50-w2:1", "top_k": 20, "index_config": "configs\\indexing\\bgem3.yaml", "index_fingerprint": "0eaaf9265487eabb25eade5ecb6a85a74ebdbee194b1b5e215befe4bec474932", "collection": "rag_bgem3", "embedding_model": "BAAI/bge-m3", "retrieval_mode": "hybrid", "branch_options": {"weights": [2.0, 1.0]}, "chunking": {"strategy": "hybrid", "chunk_size": 1000, "chunk_overlap": 100, "separators": ["\n\n", "\n", ". ", " ", ""], "min_chunk_size": 200, "max_chunk_size": 1500, "semantic_buffer_size": 1, "semantic_threshold_percentile": 85.0, "semantic_min_sentences": 3, "hybrid_max_docs_for_semantic": 5, "neighbor_context_chars": 100}, "span_resolution": {"resolved": 209, "kept_chunk_ids": 33, "unmatched_queries": [], "min_overlap_ratio": 0.5, "label_changed": 9}}`
- Môi trường: platform=Windows-11-10.0.26200-SP0, python=3.13.11

## Tổng thể

| Metric | Giá trị |
|---|---:|
| hit_rate@1 | 0.2967 |
| hit_rate@10 | 0.6029 |
| hit_rate@20 | 0.6794 |
| hit_rate@5 | 0.4880 |
| map@20 | 0.3522 |
| mrr | 0.3848 |
| ndcg@10 | 0.4053 |
| precision@1 | 0.2967 |
| precision@10 | 0.0775 |
| precision@20 | 0.0447 |
| precision@5 | 0.1234 |
| recall@1 | 0.2257 |
| recall@10 | 0.5566 |
| recall@20 | 0.6427 |
| recall@5 | 0.4410 |

## Độ trễ truy hồi (ms)

| Phân vị | ms |
|---|---:|
| mean | 36.5 |
| p50 | 36.1 |
| p95 | 49.9 |
| max | 55.9 |
| stdev | 9.9 |

## Theo nhóm truy vấn

| Nhóm | n | hit_rate@1 | hit_rate@10 | hit_rate@20 | hit_rate@5 | map@20 | mrr | ndcg@10 | precision@1 | precision@10 | precision@20 | precision@5 | recall@1 | recall@10 | recall@20 | recall@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adversarial | 34 | 0.2353 | 0.4706 | 0.6471 | 0.3529 | 0.2998 | 0.3002 | 0.3312 | 0.2353 | 0.0529 | 0.0353 | 0.0765 | 0.2353 | 0.4706 | 0.6324 | 0.3382 |
| aggregation | 26 | 0.4231 | 0.8077 | 0.8462 | 0.6538 | 0.4039 | 0.5612 | 0.4843 | 0.4231 | 0.1308 | 0.0731 | 0.2077 | 0.2051 | 0.5641 | 0.6282 | 0.4615 |
| cross_lingual | 43 | 0.0000 | 0.3023 | 0.4419 | 0.1395 | 0.0639 | 0.0625 | 0.1081 | 0.0000 | 0.0326 | 0.0244 | 0.0279 | 0.0000 | 0.2907 | 0.4302 | 0.1279 |
| factoid | 68 | 0.3676 | 0.7500 | 0.7794 | 0.6471 | 0.4843 | 0.4868 | 0.5469 | 0.3676 | 0.0765 | 0.0397 | 0.1324 | 0.3603 | 0.7500 | 0.7794 | 0.6471 |
| multi_hop | 34 | 0.5000 | 0.7059 | 0.7353 | 0.6471 | 0.4777 | 0.5541 | 0.5298 | 0.5000 | 0.1265 | 0.0721 | 0.2176 | 0.2451 | 0.6225 | 0.7059 | 0.5343 |
| table_lookup | 4 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.0250 | 0.0125 | 0.0500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 |

## Theo ngôn ngữ

| Nhóm | n | hit_rate@1 | hit_rate@10 | hit_rate@20 | hit_rate@5 | map@20 | mrr | ndcg@10 | precision@1 | precision@10 | precision@20 | precision@5 | recall@1 | recall@10 | recall@20 | recall@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| en | 82 | 0.3537 | 0.5976 | 0.6585 | 0.5366 | 0.3990 | 0.4420 | 0.4459 | 0.3537 | 0.0878 | 0.0482 | 0.1537 | 0.2398 | 0.5589 | 0.6220 | 0.4837 |
| vi | 127 | 0.2598 | 0.6063 | 0.6929 | 0.4567 | 0.3220 | 0.3479 | 0.3790 | 0.2598 | 0.0709 | 0.0425 | 0.1039 | 0.2165 | 0.5551 | 0.6562 | 0.4134 |

> Câu thuộc nhóm `unanswerable` không có tài liệu liên quan nên bị loại khỏi
> mọi metric xếp hạng. Chúng được đo riêng bằng refusal correctness (W5-02).
