# Retrieval eval — `bgem3-rrf-k5`

- Thời điểm chạy: `2026-08-20T15:42:18+00:00`
- Số truy vấn: **242** (chấm điểm 209, bỏ qua 33 câu unanswerable)
- Config: `{"retriever": "qdrant-hybrid:rag_bgem3:rrf5-c50", "top_k": 20, "index_config": "configs\\indexing\\bgem3.yaml", "index_fingerprint": "0eaaf9265487eabb25eade5ecb6a85a74ebdbee194b1b5e215befe4bec474932", "collection": "rag_bgem3", "embedding_model": "BAAI/bge-m3", "retrieval_mode": "hybrid", "branch_options": {"k": 5}, "chunking": {"strategy": "hybrid", "chunk_size": 1000, "chunk_overlap": 100, "separators": ["\n\n", "\n", ". ", " ", ""], "min_chunk_size": 200, "max_chunk_size": 1500, "semantic_buffer_size": 1, "semantic_threshold_percentile": 85.0, "semantic_min_sentences": 3, "hybrid_max_docs_for_semantic": 5, "neighbor_context_chars": 100}, "span_resolution": {"resolved": 209, "kept_chunk_ids": 33, "unmatched_queries": [], "min_overlap_ratio": 0.5, "label_changed": 9}}`
- Môi trường: platform=Windows-11-10.0.26200-SP0, python=3.13.11

## Tổng thể

| Metric | Giá trị |
|---|---:|
| hit_rate@1 | 0.3206 |
| hit_rate@10 | 0.6603 |
| hit_rate@20 | 0.7129 |
| hit_rate@5 | 0.5502 |
| map@20 | 0.3842 |
| mrr | 0.4236 |
| ndcg@10 | 0.4443 |
| precision@1 | 0.3206 |
| precision@10 | 0.0842 |
| precision@20 | 0.0471 |
| precision@5 | 0.1378 |
| recall@1 | 0.2448 |
| recall@10 | 0.6053 |
| recall@20 | 0.6754 |
| recall@5 | 0.4848 |

## Độ trễ truy hồi (ms)

| Phân vị | ms |
|---|---:|
| mean | 39.4 |
| p50 | 43.5 |
| p95 | 52.8 |
| max | 63.8 |
| stdev | 9.9 |

## Theo nhóm truy vấn

| Nhóm | n | hit_rate@1 | hit_rate@10 | hit_rate@20 | hit_rate@5 | map@20 | mrr | ndcg@10 | precision@1 | precision@10 | precision@20 | precision@5 | recall@1 | recall@10 | recall@20 | recall@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adversarial | 34 | 0.2647 | 0.5000 | 0.5882 | 0.4412 | 0.3249 | 0.3320 | 0.3653 | 0.2647 | 0.0559 | 0.0324 | 0.1000 | 0.2500 | 0.5000 | 0.5735 | 0.4412 |
| aggregation | 26 | 0.3846 | 0.8846 | 0.8846 | 0.8462 | 0.4141 | 0.5843 | 0.5111 | 0.3846 | 0.1423 | 0.0769 | 0.2538 | 0.1859 | 0.6090 | 0.6603 | 0.5449 |
| cross_lingual | 43 | 0.0233 | 0.3488 | 0.4651 | 0.1860 | 0.1018 | 0.1036 | 0.1442 | 0.0233 | 0.0372 | 0.0279 | 0.0372 | 0.0233 | 0.3140 | 0.4651 | 0.1628 |
| factoid | 68 | 0.4118 | 0.8235 | 0.8529 | 0.6471 | 0.5255 | 0.5280 | 0.5940 | 0.4118 | 0.0838 | 0.0434 | 0.1324 | 0.4044 | 0.8235 | 0.8529 | 0.6471 |
| multi_hop | 34 | 0.5588 | 0.7353 | 0.7647 | 0.7059 | 0.5184 | 0.6162 | 0.5733 | 0.5588 | 0.1324 | 0.0735 | 0.2294 | 0.2745 | 0.6520 | 0.7206 | 0.5637 |
| table_lookup | 4 | 0.0000 | 0.5000 | 0.5000 | 0.5000 | 0.1875 | 0.1875 | 0.2654 | 0.0000 | 0.0500 | 0.0250 | 0.1000 | 0.0000 | 0.5000 | 0.5000 | 0.5000 |

## Theo ngôn ngữ

| Nhóm | n | hit_rate@1 | hit_rate@10 | hit_rate@20 | hit_rate@5 | map@20 | mrr | ndcg@10 | precision@1 | precision@10 | precision@20 | precision@5 | recall@1 | recall@10 | recall@20 | recall@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| en | 82 | 0.3415 | 0.6707 | 0.7195 | 0.5732 | 0.4060 | 0.4448 | 0.4635 | 0.3415 | 0.0951 | 0.0524 | 0.1634 | 0.2337 | 0.6118 | 0.6748 | 0.5102 |
| vi | 127 | 0.3071 | 0.6535 | 0.7087 | 0.5354 | 0.3702 | 0.4099 | 0.4319 | 0.3071 | 0.0772 | 0.0437 | 0.1213 | 0.2520 | 0.6010 | 0.6759 | 0.4685 |

> Câu thuộc nhóm `unanswerable` không có tài liệu liên quan nên bị loại khỏi
> mọi metric xếp hạng. Chúng được đo riêng bằng refusal correctness (W5-02).
