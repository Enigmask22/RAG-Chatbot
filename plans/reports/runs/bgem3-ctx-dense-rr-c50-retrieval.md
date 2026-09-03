# Retrieval eval — `bgem3-ctx-dense-rr-c50`

- Thời điểm chạy: `2026-09-03T09:59:46+00:00`
- Số truy vấn: **242** (chấm điểm 209, bỏ qua 33 câu unanswerable)
- Config: `{"retriever": "reranked[qdrant-dense:rag_bgem3_ctx]:BAAI/bge-reranker-v2-m3@cuda:L512:float16:n50", "top_k": 20, "index_config": "configs\\indexing\\bgem3-contextual.yaml", "index_fingerprint": "ff0828fecae7998ad2bf04a389fbe2194ee62506468a6ceb6e9660a981eac52f", "collection": "rag_bgem3_ctx", "embedding_model": "BAAI/bge-m3", "retrieval_mode": "reranked", "branch_options": {"base": "dense", "rerank_candidates": 50}, "chunking": {"strategy": "hybrid", "size_unit": "chars", "chunk_size": 1000, "chunk_overlap": 100, "separators": ["\n\n", "\n", ". ", " ", ""], "min_chunk_size": 200, "max_chunk_size": 1500, "semantic_buffer_size": 1, "semantic_threshold_percentile": 85.0, "semantic_min_sentences": 3, "hybrid_max_docs_for_semantic": 5, "parent_size_multiple": 4, "structure_merge_short_sections": true, "neighbor_context_chars": 100}, "span_resolution": {"resolved": 209, "kept_chunk_ids": 33, "unmatched_queries": [], "min_overlap_ratio": 0.5, "label_changed": 9}}`
- Môi trường: platform=Windows-11-10.0.26200-SP0, python=3.13.11

## Tổng thể

| Metric | Giá trị |
|---|---:|
| hit_rate@1 | 0.6077 |
| hit_rate@10 | 0.8134 |
| hit_rate@20 | 0.8230 |
| hit_rate@5 | 0.8038 |
| map@20 | 0.6496 |
| mrr | 0.6876 |
| ndcg@10 | 0.6937 |
| precision@1 | 0.6077 |
| precision@10 | 0.1091 |
| precision@20 | 0.0562 |
| precision@5 | 0.2115 |
| recall@1 | 0.4880 |
| recall@10 | 0.7887 |
| recall@20 | 0.8054 |
| recall@5 | 0.7679 |

## Độ trễ truy hồi (ms)

| Phân vị | ms |
|---|---:|
| mean | 682.5 |
| p50 | 691.9 |
| p95 | 757.7 |
| max | 805.5 |
| stdev | 53.6 |

## Theo nhóm truy vấn

| Nhóm | n | hit_rate@1 | hit_rate@10 | hit_rate@20 | hit_rate@5 | map@20 | mrr | ndcg@10 | precision@1 | precision@10 | precision@20 | precision@5 | recall@1 | recall@10 | recall@20 | recall@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adversarial | 34 | 0.5000 | 0.7353 | 0.7353 | 0.7353 | 0.5833 | 0.5907 | 0.6235 | 0.5000 | 0.0794 | 0.0397 | 0.1588 | 0.4706 | 0.7353 | 0.7353 | 0.7353 |
| aggregation | 26 | 0.6923 | 0.8846 | 0.8846 | 0.8846 | 0.6264 | 0.7821 | 0.7015 | 0.6923 | 0.1808 | 0.0981 | 0.3462 | 0.3141 | 0.7756 | 0.8397 | 0.7436 |
| cross_lingual | 43 | 0.4884 | 0.6512 | 0.6744 | 0.6279 | 0.5416 | 0.5407 | 0.5669 | 0.4884 | 0.0744 | 0.0384 | 0.1395 | 0.4535 | 0.6512 | 0.6744 | 0.6163 |
| factoid | 68 | 0.6471 | 0.8971 | 0.8971 | 0.8824 | 0.7434 | 0.7422 | 0.7817 | 0.6471 | 0.0912 | 0.0456 | 0.1794 | 0.6471 | 0.8971 | 0.8971 | 0.8824 |
| multi_hop | 34 | 0.7353 | 0.9118 | 0.9412 | 0.9118 | 0.7002 | 0.8113 | 0.7650 | 0.7353 | 0.1706 | 0.0882 | 0.3294 | 0.3627 | 0.8431 | 0.8676 | 0.8137 |
| table_lookup | 4 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.0500 | 0.0250 | 0.1000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |

## Theo ngôn ngữ

| Nhóm | n | hit_rate@1 | hit_rate@10 | hit_rate@20 | hit_rate@5 | map@20 | mrr | ndcg@10 | precision@1 | precision@10 | precision@20 | precision@5 | recall@1 | recall@10 | recall@20 | recall@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| en | 82 | 0.5976 | 0.8293 | 0.8293 | 0.8171 | 0.6382 | 0.6872 | 0.6916 | 0.5976 | 0.1220 | 0.0622 | 0.2341 | 0.4370 | 0.7988 | 0.8069 | 0.7703 |
| vi | 127 | 0.6142 | 0.8031 | 0.8189 | 0.7953 | 0.6570 | 0.6879 | 0.6950 | 0.6142 | 0.1008 | 0.0524 | 0.1969 | 0.5210 | 0.7822 | 0.8045 | 0.7664 |

> Câu thuộc nhóm `unanswerable` không có tài liệu liên quan nên bị loại khỏi
> mọi metric xếp hạng. Chúng được đo riêng bằng refusal correctness (W5-02).
