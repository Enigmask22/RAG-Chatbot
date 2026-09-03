# Retrieval eval — `bgem3-ctx-rr-c50`

- Thời điểm chạy: `2026-09-03T09:16:14+00:00`
- Số truy vấn: **242** (chấm điểm 209, bỏ qua 33 câu unanswerable)
- Config: `{"retriever": "reranked[qdrant-hybrid:rag_bgem3_ctx:rrf1-c20]:BAAI/bge-reranker-v2-m3@cuda:L512:float16:n50", "top_k": 20, "index_config": "configs\\indexing\\bgem3-contextual.yaml", "index_fingerprint": "ff0828fecae7998ad2bf04a389fbe2194ee62506468a6ceb6e9660a981eac52f", "collection": "rag_bgem3_ctx", "embedding_model": "BAAI/bge-m3", "retrieval_mode": "reranked", "branch_options": {"k": 1, "candidate_k": 20, "base": "hybrid"}, "chunking": {"strategy": "hybrid", "size_unit": "chars", "chunk_size": 1000, "chunk_overlap": 100, "separators": ["\n\n", "\n", ". ", " ", ""], "min_chunk_size": 200, "max_chunk_size": 1500, "semantic_buffer_size": 1, "semantic_threshold_percentile": 85.0, "semantic_min_sentences": 3, "hybrid_max_docs_for_semantic": 5, "parent_size_multiple": 4, "structure_merge_short_sections": true, "neighbor_context_chars": 100}, "span_resolution": {"resolved": 209, "kept_chunk_ids": 33, "unmatched_queries": [], "min_overlap_ratio": 0.5, "label_changed": 9}}`
- Môi trường: platform=Windows-11-10.0.26200-SP0, python=3.13.11

## Tổng thể

| Metric | Giá trị |
|---|---:|
| hit_rate@1 | 0.6077 |
| hit_rate@10 | 0.8134 |
| hit_rate@20 | 0.8278 |
| hit_rate@5 | 0.8086 |
| map@20 | 0.6449 |
| mrr | 0.6912 |
| ndcg@10 | 0.6888 |
| precision@1 | 0.6077 |
| precision@10 | 0.1067 |
| precision@20 | 0.0555 |
| precision@5 | 0.2105 |
| recall@1 | 0.4856 |
| recall@10 | 0.7759 |
| recall@20 | 0.7967 |
| recall@5 | 0.7663 |

## Độ trễ truy hồi (ms)

| Phân vị | ms |
|---|---:|
| mean | 726.4 |
| p50 | 730.9 |
| p95 | 809.3 |
| max | 900.3 |
| stdev | 56.4 |

## Theo nhóm truy vấn

| Nhóm | n | hit_rate@1 | hit_rate@10 | hit_rate@20 | hit_rate@5 | map@20 | mrr | ndcg@10 | precision@1 | precision@10 | precision@20 | precision@5 | recall@1 | recall@10 | recall@20 | recall@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adversarial | 34 | 0.5000 | 0.7353 | 0.7353 | 0.7353 | 0.5833 | 0.5907 | 0.6235 | 0.5000 | 0.0794 | 0.0397 | 0.1588 | 0.4706 | 0.7353 | 0.7353 | 0.7353 |
| aggregation | 26 | 0.6923 | 0.8846 | 0.9231 | 0.8846 | 0.5961 | 0.7853 | 0.6797 | 0.6923 | 0.1731 | 0.0962 | 0.3385 | 0.3141 | 0.7500 | 0.8205 | 0.7308 |
| cross_lingual | 43 | 0.4419 | 0.5581 | 0.5814 | 0.5581 | 0.4876 | 0.4866 | 0.5041 | 0.4419 | 0.0651 | 0.0337 | 0.1256 | 0.4070 | 0.5581 | 0.5814 | 0.5465 |
| factoid | 68 | 0.6618 | 0.9412 | 0.9412 | 0.9265 | 0.7728 | 0.7716 | 0.8150 | 0.6618 | 0.0956 | 0.0478 | 0.1882 | 0.6618 | 0.9412 | 0.9412 | 0.9265 |
| multi_hop | 34 | 0.7647 | 0.9412 | 0.9706 | 0.9412 | 0.7042 | 0.8403 | 0.7647 | 0.7647 | 0.1647 | 0.0868 | 0.3294 | 0.3775 | 0.8137 | 0.8578 | 0.8137 |
| table_lookup | 4 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.0500 | 0.0250 | 0.1000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |

## Theo ngôn ngữ

| Nhóm | n | hit_rate@1 | hit_rate@10 | hit_rate@20 | hit_rate@5 | map@20 | mrr | ndcg@10 | precision@1 | precision@10 | precision@20 | precision@5 | recall@1 | recall@10 | recall@20 | recall@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| en | 82 | 0.6220 | 0.8780 | 0.8902 | 0.8659 | 0.6704 | 0.7248 | 0.7259 | 0.6220 | 0.1256 | 0.0652 | 0.2463 | 0.4553 | 0.8374 | 0.8557 | 0.8191 |
| vi | 127 | 0.5984 | 0.7717 | 0.7874 | 0.7717 | 0.6285 | 0.6695 | 0.6649 | 0.5984 | 0.0945 | 0.0492 | 0.1874 | 0.5052 | 0.7362 | 0.7585 | 0.7323 |

> Câu thuộc nhóm `unanswerable` không có tài liệu liên quan nên bị loại khỏi
> mọi metric xếp hạng. Chúng được đo riêng bằng refusal correctness (W5-02).
