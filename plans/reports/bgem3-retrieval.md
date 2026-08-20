# Retrieval eval — `bgem3`

- Thời điểm chạy: `2026-08-20T13:04:30+00:00`
- Số truy vấn: **242** (chấm điểm 209, bỏ qua 33 câu unanswerable)
- Config: `{"retriever": "qdrant-dense:rag_bgem3", "top_k": 20, "index_config": "configs\\indexing\\bgem3.yaml", "index_fingerprint": "0eaaf9265487eabb25eade5ecb6a85a74ebdbee194b1b5e215befe4bec474932", "collection": "rag_bgem3", "embedding_model": "BAAI/bge-m3", "chunking": {"strategy": "hybrid", "chunk_size": 1000, "chunk_overlap": 100, "separators": ["\n\n", "\n", ". ", " ", ""], "min_chunk_size": 200, "max_chunk_size": 1500, "semantic_buffer_size": 1, "semantic_threshold_percentile": 85.0, "semantic_min_sentences": 3, "hybrid_max_docs_for_semantic": 5, "neighbor_context_chars": 100}, "span_resolution": {"resolved": 209, "kept_chunk_ids": 33, "unmatched_queries": [], "min_overlap_ratio": 0.5, "label_changed": 9}}`
- Môi trường: platform=Windows-11-10.0.26200-SP0, python=3.13.11

## Tổng thể

| Metric | Giá trị |
|---|---:|
| hit_rate@1 | 0.3397 |
| hit_rate@10 | 0.6268 |
| hit_rate@20 | 0.6746 |
| hit_rate@5 | 0.5455 |
| map@20 | 0.3853 |
| mrr | 0.4394 |
| ndcg@10 | 0.4442 |
| precision@1 | 0.3397 |
| precision@10 | 0.0818 |
| precision@20 | 0.0445 |
| precision@5 | 0.1340 |
| recall@1 | 0.2512 |
| recall@10 | 0.5813 |
| recall@20 | 0.6324 |
| recall@5 | 0.4769 |

## Độ trễ truy hồi (ms)

| Phân vị | ms |
|---|---:|
| mean | 31.0 |
| p50 | 30.4 |
| p95 | 46.5 |
| max | 49.5 |
| stdev | 9.6 |

## Theo nhóm truy vấn

| Nhóm | n | hit_rate@1 | hit_rate@10 | hit_rate@20 | hit_rate@5 | map@20 | mrr | ndcg@10 | precision@1 | precision@10 | precision@20 | precision@5 | recall@1 | recall@10 | recall@20 | recall@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adversarial | 34 | 0.2941 | 0.5000 | 0.5588 | 0.4412 | 0.3665 | 0.3703 | 0.3974 | 0.2941 | 0.0559 | 0.0309 | 0.0941 | 0.2794 | 0.5000 | 0.5441 | 0.4265 |
| aggregation | 26 | 0.5385 | 0.8462 | 0.8462 | 0.7692 | 0.4264 | 0.6620 | 0.5235 | 0.5385 | 0.1346 | 0.0731 | 0.2308 | 0.2564 | 0.5769 | 0.6218 | 0.5000 |
| cross_lingual | 43 | 0.0930 | 0.4419 | 0.5116 | 0.3256 | 0.1963 | 0.2028 | 0.2538 | 0.0930 | 0.0535 | 0.0302 | 0.0744 | 0.0814 | 0.4419 | 0.5116 | 0.3023 |
| factoid | 68 | 0.3529 | 0.7059 | 0.7500 | 0.6029 | 0.4681 | 0.4718 | 0.5241 | 0.3529 | 0.0721 | 0.0382 | 0.1235 | 0.3456 | 0.7059 | 0.7500 | 0.6029 |
| multi_hop | 34 | 0.5588 | 0.7059 | 0.7647 | 0.6765 | 0.4855 | 0.6184 | 0.5522 | 0.5588 | 0.1294 | 0.0706 | 0.2059 | 0.2745 | 0.6324 | 0.6912 | 0.5049 |
| table_lookup | 4 | 0.0000 | 0.2500 | 0.2500 | 0.2500 | 0.0500 | 0.0500 | 0.0967 | 0.0000 | 0.0250 | 0.0125 | 0.0500 | 0.0000 | 0.2500 | 0.2500 | 0.2500 |

## Theo ngôn ngữ

| Nhóm | n | hit_rate@1 | hit_rate@10 | hit_rate@20 | hit_rate@5 | map@20 | mrr | ndcg@10 | precision@1 | precision@10 | precision@20 | precision@5 | recall@1 | recall@10 | recall@20 | recall@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| en | 82 | 0.3537 | 0.5976 | 0.6220 | 0.5122 | 0.3759 | 0.4306 | 0.4301 | 0.3537 | 0.0878 | 0.0470 | 0.1439 | 0.2398 | 0.5447 | 0.5854 | 0.4492 |
| vi | 127 | 0.3307 | 0.6457 | 0.7087 | 0.5669 | 0.3914 | 0.4451 | 0.4532 | 0.3307 | 0.0780 | 0.0429 | 0.1276 | 0.2585 | 0.6050 | 0.6627 | 0.4948 |

> Câu thuộc nhóm `unanswerable` không có tài liệu liên quan nên bị loại khỏi
> mọi metric xếp hạng. Chúng được đo riêng bằng refusal correctness (W5-02).
