# Retrieval eval — `bgem3-sparse`

- Thời điểm chạy: `2026-08-20T14:19:26+00:00`
- Số truy vấn: **242** (chấm điểm 209, bỏ qua 33 câu unanswerable)
- Config: `{"retriever": "qdrant-sparse:rag_bgem3", "top_k": 20, "index_config": "configs\\indexing\\bgem3.yaml", "index_fingerprint": "0eaaf9265487eabb25eade5ecb6a85a74ebdbee194b1b5e215befe4bec474932", "collection": "rag_bgem3", "embedding_model": "BAAI/bge-m3", "retrieval_mode": "sparse", "chunking": {"strategy": "hybrid", "chunk_size": 1000, "chunk_overlap": 100, "separators": ["\n\n", "\n", ". ", " ", ""], "min_chunk_size": 200, "max_chunk_size": 1500, "semantic_buffer_size": 1, "semantic_threshold_percentile": 85.0, "semantic_min_sentences": 3, "hybrid_max_docs_for_semantic": 5, "neighbor_context_chars": 100}, "span_resolution": {"resolved": 209, "kept_chunk_ids": 33, "unmatched_queries": [], "min_overlap_ratio": 0.5, "label_changed": 9}}`
- Môi trường: platform=Windows-11-10.0.26200-SP0, python=3.13.11

## Tổng thể

| Metric | Giá trị |
|---|---:|
| hit_rate@1 | 0.2919 |
| hit_rate@10 | 0.5120 |
| hit_rate@20 | 0.5311 |
| hit_rate@5 | 0.4593 |
| map@20 | 0.3333 |
| mrr | 0.3623 |
| ndcg@10 | 0.3733 |
| precision@1 | 0.2919 |
| precision@10 | 0.0660 |
| precision@20 | 0.0352 |
| precision@5 | 0.1167 |
| recall@1 | 0.2225 |
| recall@10 | 0.4721 |
| recall@20 | 0.5024 |
| recall@5 | 0.4171 |

## Độ trễ truy hồi (ms)

| Phân vị | ms |
|---|---:|
| mean | 115.6 |
| p50 | 113.4 |
| p95 | 142.1 |
| max | 202.8 |
| stdev | 18.5 |

## Theo nhóm truy vấn

| Nhóm | n | hit_rate@1 | hit_rate@10 | hit_rate@20 | hit_rate@5 | map@20 | mrr | ndcg@10 | precision@1 | precision@10 | precision@20 | precision@5 | recall@1 | recall@10 | recall@20 | recall@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adversarial | 34 | 0.1471 | 0.3529 | 0.4118 | 0.3529 | 0.2194 | 0.2194 | 0.2496 | 0.1471 | 0.0382 | 0.0221 | 0.0765 | 0.1471 | 0.3529 | 0.4118 | 0.3529 |
| aggregation | 26 | 0.3077 | 0.7308 | 0.7308 | 0.6538 | 0.3359 | 0.4535 | 0.4084 | 0.3077 | 0.1154 | 0.0635 | 0.1923 | 0.1410 | 0.5000 | 0.5513 | 0.4231 |
| cross_lingual | 43 | 0.0000 | 0.0233 | 0.0233 | 0.0000 | 0.0033 | 0.0033 | 0.0078 | 0.0000 | 0.0023 | 0.0012 | 0.0000 | 0.0000 | 0.0233 | 0.0233 | 0.0000 |
| factoid | 68 | 0.4118 | 0.7500 | 0.7794 | 0.6618 | 0.5218 | 0.5218 | 0.5752 | 0.4118 | 0.0765 | 0.0397 | 0.1353 | 0.4044 | 0.7500 | 0.7794 | 0.6618 |
| multi_hop | 34 | 0.5588 | 0.6471 | 0.6471 | 0.6176 | 0.4910 | 0.5794 | 0.5335 | 0.5588 | 0.1176 | 0.0618 | 0.2176 | 0.2745 | 0.5784 | 0.6078 | 0.5343 |
| table_lookup | 4 | 0.2500 | 0.5000 | 0.5000 | 0.2500 | 0.2857 | 0.2857 | 0.3333 | 0.2500 | 0.0500 | 0.0250 | 0.0500 | 0.2500 | 0.5000 | 0.5000 | 0.2500 |

## Theo ngôn ngữ

| Nhóm | n | hit_rate@1 | hit_rate@10 | hit_rate@20 | hit_rate@5 | map@20 | mrr | ndcg@10 | precision@1 | precision@10 | precision@20 | precision@5 | recall@1 | recall@10 | recall@20 | recall@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| en | 82 | 0.3537 | 0.6585 | 0.6829 | 0.6098 | 0.4091 | 0.4571 | 0.4660 | 0.3537 | 0.0927 | 0.0488 | 0.1659 | 0.2378 | 0.5935 | 0.6280 | 0.5325 |
| vi | 127 | 0.2520 | 0.4173 | 0.4331 | 0.3622 | 0.2843 | 0.3011 | 0.3135 | 0.2520 | 0.0488 | 0.0264 | 0.0850 | 0.2126 | 0.3937 | 0.4213 | 0.3425 |

> Câu thuộc nhóm `unanswerable` không có tài liệu liên quan nên bị loại khỏi
> mọi metric xếp hạng. Chúng được đo riêng bằng refusal correctness (W5-02).
