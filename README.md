Advanced GenAI & Agentic AI Learning Repository

This repository documents my structured journey in Advanced Generative AI (GenAI) and Agentic AI systems, covering both theoretical foundations and hands-on implementations.

📂 Repository Structure


📁 1. Getting Started with GenAI

Sessions: 10–11

- Introduction to Generative AI
- OpenAI SDK & Google Gemini SDK
- API consumption and integration
- Chat APIs: chat.completions vs responses.create
- Building basic GenAI applications


📁 2. FastAPI Basics

Sessions: 12

- FastAPI fundamentals
- Building APIs
- Running servers using Uvicorn
- Backend setup for AI applications


📁 3. Deep Learning Foundations

Sessions: 13–15

- Neural Networks & Deep Learning basics
- MCP (Model Context Protocol)
- Activation Functions: Sigmoid, Softmax, ReLU, Tanh
- Gradient Descent: Batch, SGD, Mini-batch
- Optimization: Momentum, Learning Rate, Adam
- Vanishing Gradient Problem



📁 4. NLP (Natural Language Processing)

Sessions: 16

- Text preprocessing
- Bag of Words (BoW), TF-IDF
- Word Embeddings: Word2Vec, GloVe
- RNN, LSTM
- Introduction to Transformers
- Intro to Prompt Engineering


📁 5. Prompt Engineering

Sessions: 17–18

🔹 Components of a Prompt

- Role
- Task
- Instructions
- Input
- Context
- Output Format

🔹 Techniques

- Zero-shot
- Few-shot
- Chain of Thought (CoT)
- ReAct
- Self-consistency

🔹 Concepts

- Transformer basics (Encoder–Decoder)
- Self-attention (Cosine similarity, Dot product)


📁 6. Transformers (Deep Dive)

Sessions: 19–20

- Transformer architecture (detailed)
- Decoder & masked multi-head attention
- Autoregressive & causal attention
- Layer normalization
- Residual connections

🔹 Tokenization

- BPE (Byte Pair Encoding)
- WordPiece
- SentencePiece


📁 7. Hugging Face & Open Source Models

Sessions: 21–22

- Using Hugging Face models
- Transformers library
- Model integration
- Introduction to RAG


📁 8. RAG & Graph RAG

Sessions: 23–25

- Conversational RAG systems
- RAG evaluation
- Graph RAG
- Cypher query language
- PDF extraction (Docling)
- GraphRAG implementation


📁 9. LangChain (Core + Advanced)

Sessions: 26–34

🔹 Core Concepts

- Models, Prompts, Outputs
- Prompt types: Static, Dynamic (f-strings), Templates

🔹 Chains

- Sequential
- Parallel
- Conditional
- Lambda functions

🔹 Advanced

- Tool usage (with & without LangChain)
- LangChain RAG:
- Document Loaders
- Text Splitters (Recursive, Character-based, Semantic)
- Embeddings
- Vector Databases

Retrievers:
- MMR (Max Marginal Relevance)
- MultiQuery Retriever
- Contextual Compression


📁 10. LangGraph

Sessions: 34–38

- LangGraph vs LangChain
- Core building blocks

🔹 Graph Workflows

- Parallelization
- Routing

🔹 Advanced Patterns

- Generator–Evaluator
- Agents
- Orchestration
- Persistence
- Streaming

 
 - Project: Stock Recommender System



📁 11. MCP (Model Context Protocol)

Sessions: 39–46

🔹 Architecture

- Client, Host, Server
- Claude as MCP host
- Data layer & Transport layer
- Connectors vs Developers

🔹 Setup

- Local MCP server (npm, npx, Docker)
- Remote MCP servers (Kubernetes, Nomad)

🔹 Integrations

- Claude & Manim

- Projects

- Local MCP server (FastAPI)
- MCP client & host

- Streamlit UI client
- Remote MCP server (FastMCP Cloud)
- GitHub deployment & client integration


📁 12. Docker

Sessions: 47–48

- Docker Desktop setup
- Dockerfile creation
- Image building
- Running containers (local & Docker Hub)
- Docker Compose (.yaml)
- Running n8n using npx


📁 13. n8n

Sessions: 49

- Introduction to n8n
- Workflow automation
- Triggers and pipelines


📁 14. Model Tuning & Optimization

Sessions: 50–58


🔹 Model Training & Fine-Tuning

- Pre-training
- Fine-tuning: Supervised, Unsupervised, RLHF
- PEFT techniques
- LoRA
- Adapters (Pre-token, Series, Parallel)
- Tuning vs RAG

🔹 PyTorch

- Tensors & operations
- Neural networks using nn module
- Model training pipeline
- Dataset & DataLoader

🔹 Practical Implementation

- Sentiment analysis fine-tuning
- LoRA-based tuning

🔹 Inference Optimization

- GPU optimization techniques
- KV Cache (with calculation)
- Continuous batching
- Speculative decoding

🔹 Model Compression

- Quantization


🎯 Learning Goals
- Build production-ready GenAI systems
- Understand deep learning fundamentals behind LLMs
- Implement RAG and Agentic workflows
- Master LangChain, LangGraph, FastAPI, Docker
- Learn model tuning and optimization techniques


🛠️ Tech Stack
- 🐍 Python
- ⚡ FastAPI
- 🤖 OpenAI API
- 🌐 Google Gemini API
- 🤗 Hugging Face
- 🔗 LangChain & LangGraph
- 🐳 Docker
- 🔥 PyTorch
- 🔄 n8n


⭐ Final Note

This repository reflects a deep, structured, and practical learning journey into modern AI systems — from fundamentals → production-grade architectures → optimization.

