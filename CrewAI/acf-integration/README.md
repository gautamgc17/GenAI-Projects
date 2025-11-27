## Agent Connect Framework + CrewAI

This repo contains an end-to-end example of how to expose an existing external agent **(built with CrewAI)** to **IBM Agent Connect Framework**.
_Note: The example showcases a **Sports RAG Agent** that can answer questions about multiple sports using both a vector store and web-search._


### Repo Structure

```
ACF-External-Agents/
└── acf-examples/
    ├── assets/
    │   ├── docker-compose.yml        # Milvus standalone compose file
    │   └── setup-milvus.ipynb        # Notebook to prepare Milvus + load data
    └── crewai/
        ├── config/
        │   ├── agents.yaml           # sports_rag_agent configuration
        │   └── tasks.yaml            # tasks configuration
        ├── .env.sample               # sample environment variables
        ├── Dockerfile                # Container for the CrewAI agent server
        ├── models.py                 # Pydantic models
        ├── requirements.txt          # Python dependencies
        ├── server.py                 # FastAPI app exposing /v1/chat/completions
        └── tools.py                  # search_knowledgebase, search_internet and other tools etc.
```

### How It All Fits Together

1. Start Milvus

```
cd acf-examples/assets
docker compose up -d
docker compose ps
```

2. Load Sports Data into Milvus

- Open setup-milvus.ipynb
- Run the notebook cells to:
    - Create collection
    - Embed and insert sports documents into Milvus


3. Configure .env (based on .env.sample)

```
cd acf-examples/crewai
cp .env.sample .env
```

Open .env and fill in the values:

```
WATSONX_URL=
WATSONX_API_KEY=
WATSONX_PROJECT_ID=

MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_USER=
MILVUS_PASSWORD=
MILVUS_COLLECTION_NAME=

TAVILY_API_KEY=
```

4. Configure the Agent & Tasks

- `config/agents.yaml`
- `config/tasks.yaml`

5. Run the CrewAI API Server

```
pip install -r requirements.txt
uvicorn server:app --reload
```

6. Call the Agent

- `GET /v1/agents` → returns metadata describing this agent for the Agent Connect Framework
- `POST /v1/chat/completions` → main endpoint that:
    - Accepts a chat-style request
    - Calls your CrewAI pipeline (process_with_crew)
    - Streams back answers (optionally using your CrewAIStreamingListener)

- Example curl command
```
curl -X 'POST' \
  'http://127.0.0.1:8080/v1/chat/completions' \
  -H 'accept: application/json' \
  -H 'X-THREAD-ID: 1234' \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "",
  "messages": [
    {
      "role": "user",
      "content": "Who won the last Women's ODI world cup in Cricket?"
    }
  ],
  "stream": true
}'
```

The `thread_id` coming from watsonx Orchestrate is passed through to the service so that `conversation history / memory is preserved across turns`. The server is `OpenAI-compatible and supports SSE-based streaming`.

7. Integrate with watsonx Orchestrate

- Use the `/v1/chat/completions` endpoint to configure your CrewAI agent as an external agent in watsonx Orchestrate.
- This API matches the Agent Connect / OpenAI-style completion format, so it can be plugged into watsonx Orchestrate.

<br>

### References

1. [Example implementations of Agent Connect Framework with popular frameworks](https://connect.watson-orchestrate.ibm.com/examples#crewai-example)
2. [IBM watsonx orchestrate - external agent support](https://github.com/watson-developer-cloud/watsonx-orchestrate-developer-toolkit/blob/main/external_agent/spec.yaml)
