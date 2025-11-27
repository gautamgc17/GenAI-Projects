import os
from datetime import datetime
from pydantic import BaseModel
from crewai.tools import tool, BaseTool
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Embeddings
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames as EmbedParams
from pymilvus import Collection, connections as milvus_connections
from tavily import TavilyClient 


@tool
def get_current_datetime():
    """
    Get the current system date.
    """
    try:
        now = datetime.now()
        formatted_date = now.strftime("%d %B %Y")
        return formatted_date
    except Exception as e:
        return f"Error getting current date: {str(e)}"


@tool
def search_knowledgebase(query: str):
    """
    Search the Internal Knowledge Base for relevant documents based on the query.
    """
    try:
        watsonx_url = os.getenv("WATSONX_URL")
        watsonx_apikey = os.getenv("WATSONX_API_KEY")
        watsonx_projectId = os.getenv("WATSONX_PROJECT_ID")
        milvus_host = os.getenv("MILVUS_HOST")
        milvus_port = int(os.getenv("MILVUS_PORT", 19530))
        milvus_user = os.getenv("MILVUS_USER")
        milvus_password = os.getenv("MILVUS_PASSWORD")
        milvus_collection_name = os.getenv("MILVUS_COLLECTION_NAME")
        if milvus_host in ("localhost", "127.0.0.1"):
            milvus_connection_args = {
                "host": milvus_host,
                "port": milvus_port,
                "secure": False,  
            }
        else:
            milvus_connection_args = {
                "host": milvus_host,
                "port": milvus_port,
                "secure": True,
                "user": milvus_user,
                "password": milvus_password
            }
        credentials = Credentials(
            url=watsonx_url,
            api_key=watsonx_apikey,
            verify=False
        )
        model_id = "intfloat/multilingual-e5-large"
        embed_params = {
            EmbedParams.TRUNCATE_INPUT_TOKENS: 512,
            EmbedParams.RETURN_OPTIONS: {
                'input_text': True
            }
        }
        embedding = Embeddings(
            model_id=model_id,
            credentials=credentials,
            params=embed_params,
            project_id=watsonx_projectId
        )
        query_embeddings = embedding.embed_query(query)
        milvus_connections.connect(**milvus_connection_args)
        collection = Collection(milvus_collection_name)
        collection.load()
        output_fields = ["chunk", "metadata", "title", "source_url"]
        search_params = {"metric_type": "COSINE"}
        raw_results = collection.search(
            data=[query_embeddings],
            anns_field="chunk_embeddings",
            param=search_params,
            limit=3,
            output_fields=output_fields,
            timeout=10
        )
        results = []
        for hit in raw_results[0]:
            title = hit.entity.get("title")
            chunk = hit.entity.get("chunk")
            source_url = hit.entity.get("source_url", "")
            results.append({"title": title, "chunk": chunk, "source_url": source_url})
        return results
    except Exception as e:
        print(f"Search failed: {e}")
        return []


@tool
def search_internet(query: str):
    """
    Use Tavily Search API to search the web and return the search results
    """
    try:
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        tavily_client = TavilyClient(api_key=tavily_api_key)
        response = tavily_client.search(
            query=query,
            max_results=5,
            include_images=False,
            include_raw_content=True
        )
        results = response.get("results", [])
        return results
    except Exception as e:
        print(f"Search failed: {e}")
        return []