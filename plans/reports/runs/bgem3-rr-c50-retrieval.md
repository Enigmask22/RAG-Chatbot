# Retrieval eval — `bgem3-rr-c50`

- Thời điểm chạy: `2026-08-21T04:32:07+00:00`
- Số truy vấn: **242** (chấm điểm 209, bỏ qua 33 câu unanswerable)
- Config: `{"retriever": "reranked[qdrant-hybrid:rag_bgem3:rrf1-c20]:BAAI/bge-reranker-v2-m3@cuda:L512:float16:n50", "top_k": 20, "index_config": "configs\\indexing\\bgem3.yaml", "index_fingerprint": "0eaaf9265487eabb25eade5ecb6a85a74ebdbee194b1b5e215befe4bec474932", "collection": "rag_bgem3", "embedding_model": "BAAI/bge-m3", "retrieval_mode": "reranked", "branch_options": {"k": 1, "candidate_k": 20, "base": "hybrid", "rerank_candidates": 50}, "chunking": {"strategy": "hybrid", "chunk_size": 1000, "chunk_overlap": 100, "separators": ["\n\n", "\n", ". ", " ", ""], "min_chunk_size": 200, "max_chunk_size": 1500, "semantic_buffer_size": 1, "semantic_threshold_percentile": 85.0, "semantic_min_sentences": 3, "hybrid_max_docs_for_semantic": 5, "neighbor_context_chars": 100}, "span_resolution": {"resolved": 209, "kept_chunk_ids": 33, "unmatched_queries": [], "min_overlap_ratio": 0.5, "label_changed": 9}}`
- Môi trường: platform=Windows-11-10.0.26200-SP0, python=3.13.11

## Tổng thể

| Metric | Giá trị |
|---|---:|
| hit_rate@1 | 0.5598 |
| hit_rate@10 | 0.7703 |
| hit_rate@20 | 0.7751 |
| hit_rate@5 | 0.7512 |
| map@20 | 0.6051 |
| mrr | 0.6440 |
| ndcg@10 | 0.6481 |
| precision@1 | 0.5598 |
| precision@10 | 0.1010 |
| precision@20 | 0.0510 |
| precision@5 | 0.1914 |
| recall@1 | 0.4474 |
| recall@10 | 0.7352 |
| recall@20 | 0.7424 |
| recall@5 | 0.7026 |

## Độ trễ truy hồi (ms)

| Phân vị | ms |
|---|---:|
| mean | 529.4 |
| p50 | 534.4 |
| p95 | 604.0 |
| max | 642.9 |
| stdev | 44.4 |

## Theo nhóm truy vấn

| Nhóm | n | hit_rate@1 | hit_rate@10 | hit_rate@20 | hit_rate@5 | map@20 | mrr | ndcg@10 | precision@1 | precision@10 | precision@20 | precision@5 | recall@1 | recall@10 | recall@20 | recall@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adversarial | 34 | 0.4412 | 0.7059 | 0.7059 | 0.6765 | 0.5272 | 0.5446 | 0.5732 | 0.4412 | 0.0765 | 0.0382 | 0.1412 | 0.4118 | 0.6912 | 0.6912 | 0.6618 |
| aggregation | 26 | 0.6538 | 0.8462 | 0.8462 | 0.8462 | 0.5746 | 0.7385 | 0.6423 | 0.6538 | 0.1577 | 0.0808 | 0.3000 | 0.2949 | 0.6731 | 0.6923 | 0.6410 |
| cross_lingual | 43 | 0.4419 | 0.5814 | 0.6047 | 0.5581 | 0.4978 | 0.5009 | 0.5191 | 0.4419 | 0.0674 | 0.0349 | 0.1302 | 0.4070 | 0.5814 | 0.6047 | 0.5581 |
| factoid | 68 | 0.6176 | 0.8824 | 0.8824 | 0.8529 | 0.7182 | 0.7206 | 0.7596 | 0.6176 | 0.0897 | 0.0449 | 0.1735 | 0.6103 | 0.8824 | 0.8824 | 0.8529 |
| multi_hop | 34 | 0.6471 | 0.8235 | 0.8235 | 0.8235 | 0.6283 | 0.7157 | 0.6850 | 0.6471 | 0.1529 | 0.0765 | 0.2824 | 0.3186 | 0.7549 | 0.7549 | 0.6961 |
| table_lookup | 4 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.0500 | 0.0250 | 0.1000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |

## Theo ngôn ngữ

| Nhóm | n | hit_rate@1 | hit_rate@10 | hit_rate@20 | hit_rate@5 | map@20 | mrr | ndcg@10 | precision@1 | precision@10 | precision@20 | precision@5 | recall@1 | recall@10 | recall@20 | recall@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| en | 82 | 0.5366 | 0.7439 | 0.7439 | 0.7439 | 0.5907 | 0.6234 | 0.6291 | 0.5366 | 0.1073 | 0.0543 | 0.2098 | 0.4065 | 0.7012 | 0.7073 | 0.6911 |
| vi | 127 | 0.5748 | 0.7874 | 0.7953 | 0.7559 | 0.6144 | 0.6573 | 0.6604 | 0.5748 | 0.0969 | 0.0488 | 0.1795 | 0.4738 | 0.7572 | 0.7651 | 0.7100 |

> Câu thuộc nhóm `unanswerable` không có tài liệu liên quan nên bị loại khỏi
> mọi metric xếp hạng. Chúng được đo riêng bằng refusal correctness (W5-02).
