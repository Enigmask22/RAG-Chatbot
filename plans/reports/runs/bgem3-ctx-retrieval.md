# Retrieval eval — `bgem3-ctx`

- Thời điểm chạy: `2026-09-03T09:11:54+00:00`
- Số truy vấn: **242** (chấm điểm 209, bỏ qua 33 câu unanswerable)
- Config: `{"retriever": "qdrant-dense:rag_bgem3_ctx", "top_k": 20, "index_config": "configs\\indexing\\bgem3-contextual.yaml", "index_fingerprint": "ff0828fecae7998ad2bf04a389fbe2194ee62506468a6ceb6e9660a981eac52f", "collection": "rag_bgem3_ctx", "embedding_model": "BAAI/bge-m3", "retrieval_mode": "dense", "branch_options": {}, "chunking": {"strategy": "hybrid", "size_unit": "chars", "chunk_size": 1000, "chunk_overlap": 100, "separators": ["\n\n", "\n", ". ", " ", ""], "min_chunk_size": 200, "max_chunk_size": 1500, "semantic_buffer_size": 1, "semantic_threshold_percentile": 85.0, "semantic_min_sentences": 3, "hybrid_max_docs_for_semantic": 5, "parent_size_multiple": 4, "structure_merge_short_sections": true, "neighbor_context_chars": 100}, "span_resolution": {"resolved": 209, "kept_chunk_ids": 33, "unmatched_queries": [], "min_overlap_ratio": 0.5, "label_changed": 9}}`
- Môi trường: platform=Windows-11-10.0.26200-SP0, python=3.13.11

## Tổng thể

| Metric | Giá trị |
|---|---:|
| hit_rate@1 | 0.3971 |
| hit_rate@10 | 0.6842 |
| hit_rate@20 | 0.7464 |
| hit_rate@5 | 0.6172 |
| map@20 | 0.4489 |
| mrr | 0.4921 |
| ndcg@10 | 0.5019 |
| precision@1 | 0.3971 |
| precision@10 | 0.0890 |
| precision@20 | 0.0493 |
| precision@5 | 0.1579 |
| recall@1 | 0.3030 |
| recall@10 | 0.6348 |
| recall@20 | 0.7081 |
| recall@5 | 0.5654 |

## Độ trễ truy hồi (ms)

| Phân vị | ms |
|---|---:|
| mean | 41.4 |
| p50 | 40.3 |
| p95 | 61.5 |
| max | 69.8 |
| stdev | 9.8 |

## Theo nhóm truy vấn

| Nhóm | n | hit_rate@1 | hit_rate@10 | hit_rate@20 | hit_rate@5 | map@20 | mrr | ndcg@10 | precision@1 | precision@10 | precision@20 | precision@5 | recall@1 | recall@10 | recall@20 | recall@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adversarial | 34 | 0.3529 | 0.5588 | 0.6176 | 0.5000 | 0.4172 | 0.4172 | 0.4486 | 0.3529 | 0.0618 | 0.0338 | 0.1118 | 0.3235 | 0.5588 | 0.6176 | 0.5000 |
| aggregation | 26 | 0.5385 | 0.8462 | 0.8462 | 0.7692 | 0.5009 | 0.6529 | 0.5816 | 0.5385 | 0.1538 | 0.0827 | 0.2615 | 0.2500 | 0.6603 | 0.7115 | 0.5769 |
| cross_lingual | 43 | 0.2093 | 0.4651 | 0.5116 | 0.3953 | 0.2740 | 0.2869 | 0.3172 | 0.2093 | 0.0535 | 0.0302 | 0.0930 | 0.1744 | 0.4535 | 0.5116 | 0.3953 |
| factoid | 68 | 0.4265 | 0.7647 | 0.8676 | 0.6912 | 0.5313 | 0.5338 | 0.5818 | 0.4265 | 0.0779 | 0.0441 | 0.1412 | 0.4191 | 0.7647 | 0.8676 | 0.6912 |
| multi_hop | 34 | 0.5294 | 0.8235 | 0.8824 | 0.7941 | 0.5170 | 0.6448 | 0.5883 | 0.5294 | 0.1382 | 0.0765 | 0.2529 | 0.2598 | 0.6765 | 0.7500 | 0.6225 |
| table_lookup | 4 | 0.2500 | 0.5000 | 0.5000 | 0.2500 | 0.2812 | 0.2812 | 0.3289 | 0.2500 | 0.0500 | 0.0250 | 0.0500 | 0.2500 | 0.5000 | 0.5000 | 0.2500 |

## Theo ngôn ngữ

| Nhóm | n | hit_rate@1 | hit_rate@10 | hit_rate@20 | hit_rate@5 | map@20 | mrr | ndcg@10 | precision@1 | precision@10 | precision@20 | precision@5 | recall@1 | recall@10 | recall@20 | recall@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| en | 82 | 0.4268 | 0.6951 | 0.7805 | 0.6463 | 0.4766 | 0.5226 | 0.5247 | 0.4268 | 0.1000 | 0.0567 | 0.1756 | 0.3110 | 0.6382 | 0.7398 | 0.5833 |
| vi | 127 | 0.3780 | 0.6772 | 0.7244 | 0.5984 | 0.4310 | 0.4723 | 0.4872 | 0.3780 | 0.0819 | 0.0445 | 0.1465 | 0.2979 | 0.6325 | 0.6877 | 0.5538 |

> Câu thuộc nhóm `unanswerable` không có tài liệu liên quan nên bị loại khỏi
> mọi metric xếp hạng. Chúng được đo riêng bằng refusal correctness (W5-02).
