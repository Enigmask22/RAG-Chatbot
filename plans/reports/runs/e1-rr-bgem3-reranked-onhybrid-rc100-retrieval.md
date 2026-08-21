# Retrieval eval — `e1-rr-bgem3-reranked-onhybrid-rc100`

- Thời điểm chạy: `2026-08-21T10:03:41+00:00`
- Số truy vấn: **242** (chấm điểm 209, bỏ qua 33 câu unanswerable)
- Config: `{"retriever": "reranked[qdrant-hybrid:rag_bgem3:rrf60-c50]:BAAI/bge-reranker-v2-m3@cuda:L512:float16:n100", "top_k": 20, "index_config": "configs\\indexing\\bgem3.yaml", "index_fingerprint": "0eaaf9265487eabb25eade5ecb6a85a74ebdbee194b1b5e215befe4bec474932", "collection": "rag_bgem3", "embedding_model": "BAAI/bge-m3", "retrieval_mode": "reranked", "branch_options": {"base": "hybrid", "rerank_candidates": 100, "rerank_device": "cuda", "rerank_dtype": "float16"}, "chunking": {"strategy": "hybrid", "chunk_size": 1000, "chunk_overlap": 100, "separators": ["\n\n", "\n", ". ", " ", ""], "min_chunk_size": 200, "max_chunk_size": 1500, "semantic_buffer_size": 1, "semantic_threshold_percentile": 85.0, "semantic_min_sentences": 3, "hybrid_max_docs_for_semantic": 5, "neighbor_context_chars": 100}, "span_resolution": {"resolved": 209, "kept_chunk_ids": 33, "unmatched_queries": [], "min_overlap_ratio": 0.5, "label_changed": 9}}`
- Môi trường: platform=Windows-11-10.0.26200-SP0, python=3.13.11

## Tổng thể

| Metric | Giá trị |
|---|---:|
| hit_rate@1 | 0.5789 |
| hit_rate@10 | 0.8134 |
| hit_rate@20 | 0.8182 |
| hit_rate@5 | 0.7895 |
| map@20 | 0.6274 |
| mrr | 0.6694 |
| ndcg@10 | 0.6736 |
| precision@1 | 0.5789 |
| precision@10 | 0.1048 |
| precision@20 | 0.0531 |
| precision@5 | 0.1990 |
| recall@1 | 0.4665 |
| recall@10 | 0.7679 |
| recall@20 | 0.7775 |
| recall@5 | 0.7352 |

## Độ trễ truy hồi (ms)

| Phân vị | ms |
|---|---:|
| mean | 1038.4 |
| p50 | 1043.4 |
| p95 | 1163.9 |
| max | 1242.3 |
| stdev | 82.3 |

## Theo nhóm truy vấn

| Nhóm | n | hit_rate@1 | hit_rate@10 | hit_rate@20 | hit_rate@5 | map@20 | mrr | ndcg@10 | precision@1 | precision@10 | precision@20 | precision@5 | recall@1 | recall@10 | recall@20 | recall@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adversarial | 34 | 0.4706 | 0.7647 | 0.7647 | 0.7647 | 0.5774 | 0.5990 | 0.6258 | 0.4706 | 0.0824 | 0.0426 | 0.1588 | 0.4412 | 0.7500 | 0.7647 | 0.7353 |
| aggregation | 26 | 0.6538 | 0.8846 | 0.8846 | 0.8462 | 0.5687 | 0.7449 | 0.6392 | 0.6538 | 0.1577 | 0.0808 | 0.3000 | 0.2949 | 0.6667 | 0.6859 | 0.6346 |
| cross_lingual | 43 | 0.5116 | 0.6744 | 0.6977 | 0.6512 | 0.5693 | 0.5724 | 0.5958 | 0.5116 | 0.0767 | 0.0395 | 0.1488 | 0.4767 | 0.6744 | 0.6977 | 0.6512 |
| factoid | 68 | 0.6176 | 0.8824 | 0.8824 | 0.8529 | 0.7182 | 0.7206 | 0.7596 | 0.6176 | 0.0897 | 0.0449 | 0.1735 | 0.6103 | 0.8824 | 0.8824 | 0.8529 |
| multi_hop | 34 | 0.6471 | 0.8824 | 0.8824 | 0.8529 | 0.6295 | 0.7221 | 0.6947 | 0.6471 | 0.1588 | 0.0794 | 0.2882 | 0.3186 | 0.7843 | 0.7843 | 0.7108 |
| table_lookup | 4 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.0500 | 0.0250 | 0.1000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |

## Theo ngôn ngữ

| Nhóm | n | hit_rate@1 | hit_rate@10 | hit_rate@20 | hit_rate@5 | map@20 | mrr | ndcg@10 | precision@1 | precision@10 | precision@20 | precision@5 | recall@1 | recall@10 | recall@20 | recall@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| en | 82 | 0.5610 | 0.7683 | 0.7683 | 0.7561 | 0.6123 | 0.6467 | 0.6520 | 0.5610 | 0.1098 | 0.0549 | 0.2122 | 0.4309 | 0.7256 | 0.7256 | 0.7093 |
| vi | 127 | 0.5906 | 0.8425 | 0.8504 | 0.8110 | 0.6373 | 0.6840 | 0.6876 | 0.5906 | 0.1016 | 0.0520 | 0.1906 | 0.4895 | 0.7953 | 0.8110 | 0.7520 |

> Câu thuộc nhóm `unanswerable` không có tài liệu liên quan nên bị loại khỏi
> mọi metric xếp hạng. Chúng được đo riêng bằng refusal correctness (W5-02).
