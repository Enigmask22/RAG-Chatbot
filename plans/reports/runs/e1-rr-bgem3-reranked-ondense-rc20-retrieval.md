# Retrieval eval — `e1-rr-bgem3-reranked-ondense-rc20`

- Thời điểm chạy: `2026-08-21T09:49:56+00:00`
- Số truy vấn: **242** (chấm điểm 209, bỏ qua 33 câu unanswerable)
- Config: `{"retriever": "reranked[qdrant-dense:rag_bgem3]:BAAI/bge-reranker-v2-m3@cuda:L512:float16:n20", "top_k": 20, "index_config": "configs\\indexing\\bgem3.yaml", "index_fingerprint": "0eaaf9265487eabb25eade5ecb6a85a74ebdbee194b1b5e215befe4bec474932", "collection": "rag_bgem3", "embedding_model": "BAAI/bge-m3", "retrieval_mode": "reranked", "branch_options": {"base": "dense", "rerank_candidates": 20, "rerank_device": "cuda", "rerank_dtype": "float16"}, "chunking": {"strategy": "hybrid", "chunk_size": 1000, "chunk_overlap": 100, "separators": ["\n\n", "\n", ". ", " ", ""], "min_chunk_size": 200, "max_chunk_size": 1500, "semantic_buffer_size": 1, "semantic_threshold_percentile": 85.0, "semantic_min_sentences": 3, "hybrid_max_docs_for_semantic": 5, "neighbor_context_chars": 100}, "span_resolution": {"resolved": 209, "kept_chunk_ids": 33, "unmatched_queries": [], "min_overlap_ratio": 0.5, "label_changed": 9}}`
- Môi trường: platform=Windows-11-10.0.26200-SP0, python=3.13.11

## Tổng thể

| Metric | Giá trị |
|---|---:|
| hit_rate@1 | 0.5072 |
| hit_rate@10 | 0.6699 |
| hit_rate@20 | 0.6746 |
| hit_rate@5 | 0.6555 |
| map@20 | 0.5335 |
| mrr | 0.5756 |
| ndcg@10 | 0.5676 |
| precision@1 | 0.5072 |
| precision@10 | 0.0880 |
| precision@20 | 0.0445 |
| precision@5 | 0.1694 |
| recall@1 | 0.3971 |
| recall@10 | 0.6260 |
| recall@20 | 0.6324 |
| recall@5 | 0.6069 |

## Độ trễ truy hồi (ms)

| Phân vị | ms |
|---|---:|
| mean | 229.7 |
| p50 | 229.7 |
| p95 | 267.6 |
| max | 297.9 |
| stdev | 21.7 |

## Theo nhóm truy vấn

| Nhóm | n | hit_rate@1 | hit_rate@10 | hit_rate@20 | hit_rate@5 | map@20 | mrr | ndcg@10 | precision@1 | precision@10 | precision@20 | precision@5 | recall@1 | recall@10 | recall@20 | recall@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adversarial | 34 | 0.3529 | 0.5588 | 0.5588 | 0.5294 | 0.4230 | 0.4356 | 0.4572 | 0.3529 | 0.0618 | 0.0309 | 0.1118 | 0.3235 | 0.5441 | 0.5441 | 0.5147 |
| aggregation | 26 | 0.6538 | 0.8462 | 0.8462 | 0.8077 | 0.5331 | 0.7350 | 0.6072 | 0.6538 | 0.1462 | 0.0731 | 0.2769 | 0.2949 | 0.6218 | 0.6218 | 0.5833 |
| cross_lingual | 43 | 0.3953 | 0.4884 | 0.5116 | 0.4884 | 0.4409 | 0.4401 | 0.4520 | 0.3953 | 0.0581 | 0.0302 | 0.1163 | 0.3605 | 0.4884 | 0.5116 | 0.4884 |
| factoid | 68 | 0.5588 | 0.7500 | 0.7500 | 0.7353 | 0.6379 | 0.6379 | 0.6661 | 0.5588 | 0.0765 | 0.0382 | 0.1500 | 0.5515 | 0.7500 | 0.7500 | 0.7353 |
| multi_hop | 34 | 0.6176 | 0.7647 | 0.7647 | 0.7647 | 0.5860 | 0.6789 | 0.6342 | 0.6176 | 0.1382 | 0.0706 | 0.2647 | 0.3039 | 0.6814 | 0.6912 | 0.6520 |
| table_lookup | 4 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.0250 | 0.0125 | 0.0500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 |

## Theo ngôn ngữ

| Nhóm | n | hit_rate@1 | hit_rate@10 | hit_rate@20 | hit_rate@5 | map@20 | mrr | ndcg@10 | precision@1 | precision@10 | precision@20 | precision@5 | recall@1 | recall@10 | recall@20 | recall@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| en | 82 | 0.4634 | 0.6220 | 0.6220 | 0.6098 | 0.5030 | 0.5308 | 0.5313 | 0.4634 | 0.0927 | 0.0470 | 0.1805 | 0.3394 | 0.5813 | 0.5854 | 0.5691 |
| vi | 127 | 0.5354 | 0.7008 | 0.7087 | 0.6850 | 0.5532 | 0.6045 | 0.5910 | 0.5354 | 0.0850 | 0.0429 | 0.1622 | 0.4344 | 0.6549 | 0.6627 | 0.6312 |

> Câu thuộc nhóm `unanswerable` không có tài liệu liên quan nên bị loại khỏi
> mọi metric xếp hạng. Chúng được đo riêng bằng refusal correctness (W5-02).
