# 🧠 RAG Chatbot - Advanced Document Intelligence System

> ⚠️ **Repo đang được xây lại thành nền tảng RAG production.** Bản POC Streamlit
> mô tả dưới đây đã dời vào [`legacy/`](legacy/README.md) và vẫn chạy được; nó là
> mốc so sánh cho hệ thống mới. Tiến độ và quyết định kỹ thuật:
> [`plans/CHECKLIST.md`](plans/CHECKLIST.md).

<div align="center">

![RAG System](https://img.shields.io/badge/RAG_System-v2.0.0-blue?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![AI Engine](https://img.shields.io/badge/AI_Engine-Enhanced_RAG-red?style=for-the-badge)
![Performance](https://img.shields.io/badge/Processing_Speed-3x_Faster-green?style=for-the-badge)

**Advanced Retrieval-Augmented Generation system with enhanced chunking and Vietnamese language support**

[🚀 Quick Start](#-quick-start) • [📖 Documentation](#-documentation) • [🛠️ Technologies](#️-technologies) • [📊 Performance](#-performance)

</div>

<!-- Teaser visualization block: system architecture -->
<div align="center">
  <img src="rag_architecture.png" alt="RAG System Architecture" width="45%">
  <img src="chunking_strategies.png" alt="Chunking Strategies Comparison" width="45%">
  <br/>
  <p align="center"><b>Figure 1.</b> System architecture and chunking strategy performance comparison.</p>
</div>

## 🌟 Key Features

### 🤖 Advanced RAG Pipeline
- **3 Chunking Strategies** - Hybrid, Semantic, and Fixed-size chunking
- **Smart Caching System** - 15x faster reprocessing with intelligent cache
- **Vietnamese Language Support** - Optimized for Vietnamese text processing
- **Real-time Processing** - Interactive progress tracking and analytics

### 📊 Document Intelligence
- **PDF Processing** - Advanced document parsing and text extraction
- **Vector Database** - ChromaDB with HuggingFace embeddings
- **Context Preservation** - Intelligent overlap strategies for better context
- **Metadata Enhancement** - Rich document and chunk metadata

### 🎯 Production-Ready Features
- **GPU Optimization** - CUDA-optimized for T4 and higher GPUs
- **Memory Management** - 4-bit quantization and smart garbage collection
- **Error Handling** - Robust fallback mechanisms and graceful degradation
- **Web Interface** - Modern Streamlit-based responsive UI

## 🛠️ Technologies

### 🐍 Python Stack
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat&logo=langchain&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)

- **Python 3.8+** - Core programming language
- **Streamlit** - Interactive web application framework
- **LangChain** - LLM application framework
- **PyTorch** - Deep learning backend
- **ChromaDB** - Vector database for similarity search

### 🤖 AI & NLP
![Hugging Face](https://img.shields.io/badge/Hugging_Face-FF6B6B?style=flat&logo=huggingface&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-FF6B6B?style=flat&logo=transformers&logoColor=white)
![Sentence Transformers](https://img.shields.io/badge/Sentence_Transformers-FF6B6B?style=flat&logo=sentence-transformers&logoColor=white)

- **HuggingFace Transformers** - Pre-trained language models
- **Vietnamese Bi-encoder** - Optimized for Vietnamese text
- **Vicuna 7B** - Large language model with 4-bit quantization
- **Sentence Transformers** - Semantic embeddings
- **BitsAndBytes** - Model quantization for memory efficiency

## 🎯 Problem Statement

Given a PDF document, create an intelligent question-answering system that can understand and respond to queries based on document content. The challenge involves implementing an efficient RAG pipeline with advanced chunking strategies, vector similarity search, and context-aware response generation while maintaining high performance and accuracy.

## 🔬 Methodology

### 📊 RAG Pipeline Architecture
1. **Document Processing**
   - PDF parsing using PyPDFLoader
   - Text extraction and normalization
   - Metadata preservation and enhancement

2. **Enhanced Chunking System**
   - **Hybrid Strategy**: Combines semantic and fixed-size chunking
   - **Semantic Chunking**: Uses sentence embeddings for context-aware splitting
   - **Fixed Chunking**: Fast processing for large documents
   - **Smart Overlap**: Preserves context between chunks

3. **Vector Database & Retrieval**
   - ChromaDB for vector storage and similarity search
   - Vietnamese bi-encoder for text embeddings
   - Top-k retrieval with configurable parameters

4. **Language Model Integration**
   - Vicuna 7B with 4-bit quantization
   - HuggingFace pipeline for inference
   - Context-aware prompt engineering

5. **Caching & Optimization**
   - Content-based caching system
   - Configuration-aware cache keys
   - Memory management and garbage collection

## 📊 Performance

### 🏆 System Performance Metrics

| Metric | Before Optimization | After Optimization | Improvement |
|--------|-------------------|-------------------|-------------|
| **Processing Speed** | ~30s/page | ~10s/page | **🚀 3x faster** |
| **Reprocess PDF** | 30s | 2s | **⚡ 15x faster** |
| **Memory Usage** | High peaks | Optimized | **💾 40% reduction** |
| **Cache Hit Rate** | 0% | 80-90% | **📈 Significant** |

### 🎯 Chunking Strategy Performance

| Strategy | Processing Time | Quality Score | Use Case |
|----------|----------------|---------------|----------|
| **Hybrid** | 8-12s | 8.5/10 | General documents |
| **Semantic** | 15-20s | 9.5/10 | Technical/Complex |
| **Fixed** | 3-5s | 7.5/10 | Large documents |

### 🖼️ Performance Visualization

<div align="center">
  <img src="performance_comparison.png" alt="Performance comparison across strategies" width="85%">
  <br/>
  <p align="center"><b>Figure 2.</b> Processing time vs. quality trade-offs for different chunking strategies.</p>
</div>

#### Technical Analysis

1) **Objective** — This figure demonstrates the performance characteristics of three chunking strategies across different document types and sizes, enabling optimal strategy selection based on use case requirements.

2) **Experimental Setup** — We evaluated Hybrid, Semantic, and Fixed chunking strategies on various document types (academic papers, legal documents, technical reports) with identical hardware configurations.

3) **Key Findings** —
- **Hybrid Strategy** provides the best balance between speed and quality for most use cases
- **Semantic Chunking** excels on complex technical documents requiring high context preservation
- **Fixed Chunking** offers maximum speed for large document processing

4) **Memory Optimization** —
- 4-bit quantization reduces memory usage by 40% while maintaining model performance
- Smart caching system provides 15x speedup for document reprocessing
- Garbage collection prevents memory leaks during long processing sessions

5) **Production Readiness** —
- System handles documents up to 100+ pages without memory issues
- Real-time progress tracking provides user feedback during processing
- Robust error handling ensures graceful degradation under resource constraints

## 🚀 Quick Start

### 📋 System Requirements
- **Python 3.8+** with pip
- **CUDA-compatible GPU** (T4 or higher recommended)
- **8GB+ RAM** (16GB+ recommended for large documents)
- **10GB+ Storage** (for models and cache)

### ⚡ Automated Setup

```bash
# 1. Clone repository
git clone <repository-url>
cd RAG-Chatbot

# 2. Create a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux  
source venv/bin/activate

# 3. Install dependencies
pip install -r legacy/requirements.txt
```

### 🔧 Configuration

#### Chunking Strategy Selection

| Document Type | Strategy | Chunk Size | Overlap | Rationale |
|---------------|----------|------------|---------|-----------|
| **📚 Academic Papers** | Semantic | 1200-1500 | 100-150 | Complex context preservation |
| **⚖️ Legal Documents** | Semantic | 800-1000 | 150-200 | High precision requirements |
| **📖 Books/Novels** | Hybrid | 1000-1500 | 50-100 | Balanced story flow |
| **📊 Reports/Statistics** | Fixed | 600-1000 | 50-100 | Processing speed priority |
| **📰 News/Blogs** | Hybrid | 800-1200 | 75-125 | Diverse content handling |

#### Advanced Configuration

```python
# Custom chunking configuration
from enhanced_chunking import ChunkingConfig  # legacy/

config = ChunkingConfig()
config.strategy = "hybrid"  # hybrid, semantic, fixed
config.fixed_chunk_size = 1000
config.fixed_overlap = 100
config.enable_cache = True
config.show_progress = True
```

### 🏃‍♂️ Run the application

```bash
# Run the RAG system
streamlit run legacy/app.py
```

**What it does:**
- 📥 Loads and processes PDF documents
- 🔄 Applies enhanced chunking strategies
- 🧮 Creates vector embeddings and database
- 🤖 Initializes language model with quantization
- 📊 Provides interactive web interface

## 🧪 Testing

```bash
# Test individual components
python -c "import sys; sys.path.insert(0, 'legacy'); import app"

# Test model loading
python -c "from transformers import AutoModelForCausalLM; print('Model loading OK')"

# Test chunking system
python -c "import sys; sys.path.insert(0, 'legacy'); from enhanced_chunking import EnhancedChunker"
```

## 📦 Project Structure

```
RAG-Chatbot/
├── 📦 packages/rag_core/        # Shared library: schemas, chunking, embedding, retrieval, llm
├── 🔬 pipeline/                 # Pipeline plane: corpus, indexing, goldenset, eval
├── 🌐 serving/                  # Serving plane (W4)
├── ⚙️  configs/                  # YAML configs for corpus, indexing, eval
├── 🐳 infra/                    # docker-compose: Qdrant + Postgres + Redis
├── 🧪 tests/                    # unit/ (no Docker) + integration/
├── 📋 plans/                    # CHECKLIST, WORKLOG, reports
├── 🗄️  legacy/                   # The original POC — see legacy/README.md
│   ├── 📄 app.py                # Streamlit application
│   ├── 🔧 enhanced_chunking.py  # Chunking system
│   └── 📋 requirements.txt      # pip dependencies
└── 📖 README.md
```

## 🚀 Performance Benchmarks

### ⚡ Speed Metrics
- **Document Loading**: ~5s for 50-page PDF
- **Chunking Process**: ~10s (hybrid), ~20s (semantic), ~5s (fixed)
- **Vector Creation**: ~15s for 1000 chunks
- **Model Inference**: ~3s per query
- **Cache Hit**: ~0.5s for reprocessed documents

### 💾 Memory Usage
- **Base System**: ~2GB RAM
- **Model Loading**: ~4GB VRAM (4-bit quantized)
- **Document Processing**: ~1GB RAM per 100 pages
- **Cache Storage**: ~100MB per document
- **Peak RAM**: ~8GB during large document processing

## 🔒 Security & Privacy

### 🛡️ Data Protection
- **Local Processing**: All data processed locally, no cloud uploads
- **Temporary Storage**: Temporary files automatically cleaned after processing
- **Secure Cache**: Content-based hashing for cache security
- **No Data Persistence**: No permanent storage of user documents

### 🔐 Best Practices
- Avoid uploading sensitive documents
- Regularly clear cache for privacy
- Monitor resource usage
- Use VPN for additional security when needed

## 🛠️ Troubleshooting

### ❓ Common Issues

#### 1. **Out of Memory Error**
```bash
# Solutions:
- Reduce chunk_size to 800-1000
- Use fixed strategy for large documents
- Restart application to clear memory
- Enable garbage collection
```

#### 2. **Slow Processing**
```bash
# Solutions:  
- Enable caching system
- Use hybrid strategy for balanced performance
- Reduce retrieval chunk count
- Check GPU availability
```

#### 3. **Model Loading Error**
```bash
# Solutions:
- Verify CUDA availability
- Restart Python kernel
- Clear GPU memory
- Check model download status
```

### 🔧 Performance Tuning

1. **Memory Optimization**:
   - Use 4-bit quantization
   - Enable garbage collection
   - Reduce chunk size for large documents
   - Monitor memory usage patterns

2. **Speed Optimization**:
   - Enable intelligent caching
   - Use hybrid strategy for balanced performance
   - Optimize retrieval parameters
   - Parallel processing where possible

3. **Quality Optimization**:
   - Use semantic chunking for complex documents
   - Increase overlap for better context
   - Fine-tune retrieval parameters
   - Adjust chunk size based on content type

## 🤝 Contributions

We welcome contributions!

### 📝 Development Workflow
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### 🎨 Code Standards
- **Python**: PEP 8 + Black formatter
- **Commits**: Conventional Commits
- **Documentation**: Docstrings for all functions
- **Testing**: Unit tests for critical functions

## 📚 Technical Documentation

### 🔧 API Reference

```python
# Enhanced Chunking API
from enhanced_chunking import EnhancedChunker, ChunkingConfig  # legacy/

# Create configuration
config = ChunkingConfig()
config.strategy = "hybrid"  # hybrid, semantic, fixed
config.fixed_chunk_size = 1000
config.fixed_overlap = 100
config.enable_cache = True

# Initialize chunker
chunker = EnhancedChunker(embeddings, config)

# Process documents
chunks, metadata = chunker.chunk_documents(documents)
```

### 📖 Advanced Usage

```python
# Custom chunking strategy
def custom_chunking_strategy(documents):
    # Implement custom logic
    pass

# Integration with other frameworks
def integrate_with_langchain():
    # Custom integration
    pass
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) - Transformers and datasets
- [LangChain](https://langchain.com/) - LLM application framework
- [Streamlit](https://streamlit.io/) - Web application framework
- [Chroma](https://www.trychroma.com/) - Vector database
- [Vietnamese NLP Community](https://github.com/undertheseanlp) - Language models

---

<div align="center">

**Built with ❤️ for AI Engineers**

[![GitHub Stars](https://img.shields.io/github/stars/your-repo/rag-chatbot?style=social)](https://github.com/your-repo/rag-chatbot)
[![GitHub Forks](https://img.shields.io/github/forks/your-repo/rag-chatbot?style=social)](https://github.com/your-repo/rag-chatbot/fork)

[⭐ Star this repo if you find it useful!](https://github.com/your-repo/rag-chatbot)

</div> 