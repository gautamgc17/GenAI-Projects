### Setup RAG pipeline with Milvus

To set up **Milvus** as the vectorDB used by **Sports RAG Agent** in the Agent Connect Framework + CrewAI example: follow these steps:


#### 1. Start Milvus

```bash
cd acf-examples/assets

# Download Milvus standalone docker-compose file
wget https://github.com/milvus-io/milvus/releases/download/v2.6.5/milvus-standalone-docker-compose.yml -O docker-compose.yml

# Start Milvus stack (Milvus + etcd + MinIO)
docker compose up -d

# Verify containers are running
docker compose ps
```

By default, Milvus will be available at: `gRPC: localhost:19530`


#### 2. Load Sports Data into Milvus

- Open `setup-milvus.ipynb`
- Run the notebook cells to:
    - Create collections
    - Embed and insert sports documents into Milvus

After running the notebook end-to-end, the `rag_documents` collection will be ready for queries.