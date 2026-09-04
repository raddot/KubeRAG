# KubeRAG
Scalable RAG Chatbot Platform

Architecture Flow
User -> FastAPI -> Redis Cache -> PostgreSQL+ pgvector -> LangChain RAG -> LLM -> Response
