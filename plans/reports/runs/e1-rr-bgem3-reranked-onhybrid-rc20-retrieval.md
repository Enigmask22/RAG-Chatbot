# Retrieval eval — `e1-rr-bgem3-reranked-onhybrid-rc20`

- Thời điểm chạy: `2026-08-21T09:57:18+00:00`
- Số truy vấn: **242** (chấm điểm 209, bỏ qua 33 câu unanswerable)
- Config: `{"retriever": "reranked[qdrant-hybrid:rag_bgem3:rrf60-c50]:BAAI/bge-reranker-v2-m3@cuda:L512:float16:n20", "top_k": 20, "index_config": "configs\\indexing\\bgem3.yaml", "index_fingerprint": "0eaaf9265487eabb25eade5ecb6a85a74ebdbee194b1b5e215befe4bec474932", "collection": "rag_bgem3", "embedding_model": "BAAI/bge-m3", "retrieval_mode": "reranked", "branch_options": {"base": "hybrid", "rerank_candidates": 20, "rerank_device": "cuda", "rerank_dtype": "float16"}, "chunking": {"strategy": "hybrid", "chunk_size": 1000, "chunk_overlap": 100, "separators": ["\n\n", "\n", ". ", " ", ""], "min_chunk_size": 200, "max_chunk_size": 1500, "semantic_buffer_size": 1, "semantic_threshold_percentile": 85.0, "semantic_min_sentences": 3, "hybrid_max_docs_for_semantic": 5, "neighbor_context_chars": 100}, "span_resolution": {"resolved": 209, "kept_chunk_ids": 33, "unmatched_queries": [], "min_overlap_ratio": 0.5, "label_changed": 9}}`
- Môi trường: platform=Windows-11-10.0.26200-SP0, python=3.13.11

## Tổng thể

| Metric | Giá trị |
|---|---:|
| hit_rate@1 | 0.5263 |
| hit_rate@10 | 0.6699 |
| hit_rate@20 | 0.6746 |
| hit_rate@5 | 0.6651 |
| map@20 | 0.5514 |
| mrr | 0.5902 |
| ndcg@10 | 0.5823 |
| precision@1 | 0.5263 |
| precision@10 | 0.0876 |
| precision@20 | 0.0445 |
| precision@5 | 0.1713 |
| recall@1 | 0.4211 |
| recall@10 | 0.6340 |
| recall@20 | 0.6396 |
| recall@5 | 0.6220 |

## Độ trễ truy hồi (ms)

| Phân vị | ms |
|---|---:|
| mean | 237.3 |
| p50 | 236.5 |
| p95 | 276.5 |
| max | 301.9 |
| stdev | 22.2 |

## Theo nhóm truy vấn

| Nhóm | n | hit_rate@1 | hit_rate@10 | hit_rate@20 | hit_rate@5 | map@20 | mrr | ndcg@10 | precision@1 | precision@10 | precision@20 | precision@5 | recall@1 | recall@10 | recall@20 | recall@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adversarial | 34 | 0.4706 | 0.5588 | 0.5588 | 0.5588 | 0.4944 | 0.5098 | 0.5147 | 0.4706 | 0.0618 | 0.0309 | 0.1176 | 0.4412 | 0.5588 | 0.5588 | 0.5441 |
| aggregation | 26 | 0.6538 | 0.8462 | 0.8846 | 0.8462 | 0.5466 | 0.7520 | 0.6193 | 0.6538 | 0.1462 | 0.0769 | 0.2846 | 0.2949 | 0.6282 | 0.6603 | 0.6090 |
| cross_lingual | 43 | 0.3488 | 0.3953 | 0.3953 | 0.3953 | 0.3682 | 0.3721 | 0.3763 | 0.3488 | 0.0442 | 0.0221 | 0.0884 | 0.3256 | 0.3953 | 0.3953 | 0.3953 |
| factoid | 68 | 0.6029 | 0.8235 | 0.8235 | 0.8088 | 0.6930 | 0.6930 | 0.7258 | 0.6029 | 0.0838 | 0.0419 | 0.1647 | 0.5956 | 0.8235 | 0.8235 | 0.8088 |
| multi_hop | 34 | 0.5882 | 0.7353 | 0.7353 | 0.7353 | 0.5961 | 0.6569 | 0.6345 | 0.5882 | 0.1382 | 0.0706 | 0.2706 | 0.2892 | 0.6814 | 0.6912 | 0.6667 |
| table_lookup | 4 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.0250 | 0.0125 | 0.0500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 |

## Theo ngôn ngữ

| Nhóm | n | hit_rate@1 | hit_rate@10 | hit_rate@20 | hit_rate@5 | map@20 | mrr | ndcg@10 | precision@1 | precision@10 | precision@20 | precision@5 | recall@1 | recall@10 | recall@20 | recall@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| en | 82 | 0.5366 | 0.6951 | 0.7073 | 0.6951 | 0.5731 | 0.6114 | 0.6027 | 0.5366 | 0.0988 | 0.0506 | 0.1951 | 0.4065 | 0.6484 | 0.6565 | 0.6423 |
| vi | 127 | 0.5197 | 0.6535 | 0.6535 | 0.6457 | 0.5374 | 0.5764 | 0.5692 | 0.5197 | 0.0803 | 0.0406 | 0.1559 | 0.4304 | 0.6247 | 0.6286 | 0.6089 |

> Câu thuộc nhóm `unanswerable` không có tài liệu liên quan nên bị loại khỏi
> mọi metric xếp hạng. Chúng được đo riêng bằng refusal correctness (W5-02).
