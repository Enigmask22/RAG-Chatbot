# Retrieval eval — `bgem3-rrf-c100`

- Thời điểm chạy: `2026-08-20T15:37:45+00:00`
- Số truy vấn: **242** (chấm điểm 209, bỏ qua 33 câu unanswerable)
- Config: `{"retriever": "qdrant-hybrid:rag_bgem3:rrf60-c100", "top_k": 20, "index_config": "configs\\indexing\\bgem3.yaml", "index_fingerprint": "0eaaf9265487eabb25eade5ecb6a85a74ebdbee194b1b5e215befe4bec474932", "collection": "rag_bgem3", "embedding_model": "BAAI/bge-m3", "retrieval_mode": "hybrid", "branch_options": {"candidate_k": 100}, "chunking": {"strategy": "hybrid", "chunk_size": 1000, "chunk_overlap": 100, "separators": ["\n\n", "\n", ". ", " ", ""], "min_chunk_size": 200, "max_chunk_size": 1500, "semantic_buffer_size": 1, "semantic_threshold_percentile": 85.0, "semantic_min_sentences": 3, "hybrid_max_docs_for_semantic": 5, "neighbor_context_chars": 100}, "span_resolution": {"resolved": 209, "kept_chunk_ids": 33, "unmatched_queries": [], "min_overlap_ratio": 0.5, "label_changed": 9}}`
- Môi trường: platform=Windows-11-10.0.26200-SP0, python=3.13.11

## Tổng thể

| Metric | Giá trị |
|---|---:|
| hit_rate@1 | 0.3014 |
| hit_rate@10 | 0.5024 |
| hit_rate@20 | 0.6220 |
| hit_rate@5 | 0.4498 |
| map@20 | 0.3472 |
| mrr | 0.3762 |
| ndcg@10 | 0.3785 |
| precision@1 | 0.3014 |
| precision@10 | 0.0656 |
| precision@20 | 0.0404 |
| precision@5 | 0.1148 |
| recall@1 | 0.2360 |
| recall@10 | 0.4697 |
| recall@20 | 0.5829 |
| recall@5 | 0.4075 |

## Độ trễ truy hồi (ms)

| Phân vị | ms |
|---|---:|
| mean | 43.0 |
| p50 | 43.8 |
| p95 | 57.8 |
| max | 225.9 |
| stdev | 14.7 |

## Theo nhóm truy vấn

| Nhóm | n | hit_rate@1 | hit_rate@10 | hit_rate@20 | hit_rate@5 | map@20 | mrr | ndcg@10 | precision@1 | precision@10 | precision@20 | precision@5 | recall@1 | recall@10 | recall@20 | recall@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adversarial | 34 | 0.2353 | 0.4412 | 0.5588 | 0.3235 | 0.2866 | 0.2873 | 0.3165 | 0.2353 | 0.0471 | 0.0309 | 0.0706 | 0.2353 | 0.4412 | 0.5588 | 0.3235 |
| aggregation | 26 | 0.3462 | 0.6923 | 0.8462 | 0.6538 | 0.3778 | 0.5094 | 0.4399 | 0.3462 | 0.1154 | 0.0692 | 0.2077 | 0.1731 | 0.5000 | 0.6026 | 0.4615 |
| cross_lingual | 43 | 0.0000 | 0.0233 | 0.2326 | 0.0233 | 0.0230 | 0.0223 | 0.0116 | 0.0000 | 0.0023 | 0.0128 | 0.0047 | 0.0000 | 0.0233 | 0.2326 | 0.0233 |
| factoid | 68 | 0.4118 | 0.7206 | 0.7941 | 0.6324 | 0.5152 | 0.5177 | 0.5616 | 0.4118 | 0.0735 | 0.0404 | 0.1294 | 0.4044 | 0.7206 | 0.7941 | 0.6324 |
| multi_hop | 34 | 0.5000 | 0.6176 | 0.7059 | 0.6176 | 0.4696 | 0.5429 | 0.5066 | 0.5000 | 0.1147 | 0.0662 | 0.2059 | 0.2451 | 0.5637 | 0.6520 | 0.5049 |
| table_lookup | 4 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.0250 | 0.0125 | 0.0500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 |

## Theo ngôn ngữ

| Nhóm | n | hit_rate@1 | hit_rate@10 | hit_rate@20 | hit_rate@5 | map@20 | mrr | ndcg@10 | precision@1 | precision@10 | precision@20 | precision@5 | recall@1 | recall@10 | recall@20 | recall@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| en | 82 | 0.3293 | 0.6463 | 0.6829 | 0.5488 | 0.4016 | 0.4366 | 0.4586 | 0.3293 | 0.0927 | 0.0488 | 0.1561 | 0.2297 | 0.5996 | 0.6341 | 0.4898 |
| vi | 127 | 0.2835 | 0.4094 | 0.5827 | 0.3858 | 0.3120 | 0.3372 | 0.3268 | 0.2835 | 0.0480 | 0.0350 | 0.0882 | 0.2402 | 0.3858 | 0.5499 | 0.3543 |

> Câu thuộc nhóm `unanswerable` không có tài liệu liên quan nên bị loại khỏi
> mọi metric xếp hạng. Chúng được đo riêng bằng refusal correctness (W5-02).
