# Import libraries
import os
import time
import uuid
import json
import uvicorn
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, RedirectResponse

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END 
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from langchain_ibm import ChatWatsonx
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters

from models import ChatRequest
from tools import get_current_datetime, search_knowledgebase, search_internet


# Initialize LLM
WATSONX_URL = os.environ["WATSONX_URL"] = os.getenv("WATSONX_URL")  
WATSONX_API_KEY = WATSONX_APIKEY = os.environ["WATSONX_APIKEY"] = os.environ["WATSONX_API_KEY"] = os.getenv("WATSONX_API_KEY") 
WATSONX_PROJECT_ID = os.environ["WATSONX_PROJECT_ID"] = os.getenv("WATSONX_PROJECT_ID")
WATSONX_MODEL_ID = os.environ["WATSONX_MODEL_ID"] = os.getenv("WATSONX_MODEL_ID", "meta-llama/llama-4-maverick-17b-128e-instruct-fp8")  
llm = ChatWatsonx(
    model_id = WATSONX_MODEL_ID,
    url = WATSONX_URL,
    project_id = WATSONX_PROJECT_ID,
    api_key = WATSONX_API_KEY,
    max_tokens = 16000,
    temperature = 0.1
)


# Create agent
def create_agent():
    tools = [get_current_datetime, search_knowledgebase, search_internet]    
    llm_with_tools = llm.bind_tools(tools)
    print(f"Tools registered: {[t.name for t in tools]}")

    sys_msg = SystemMessage(content="""You are a Multi-Sport Knowledge Agent - a well-rounded expert trained on general sports knowledge across football, cricket, basketball, tennis, motorsports, athletics, and more.

Answer user questions about sports, sporting events, rules, teams, players, tournaments, and historical or current results. Use the vector-based Sports RAG knowledgebase first, and supplement with internet search only when needed.

**Process for Answering**:
1. Detect the user's query language.
2. First use the `search_knowledgebase` tool to retrieve relevant information from the Sports RAG vector database.
3. If no relevant information is found, use the `search_internet` tool to gather recent or missing information.
4. Generate a clear, friendly, and accurate answer in the user's language.
5. When the answer uses retrieved information, include source links in a new line using: **Source**: [Link]
6. If the answer is based solely on general internal knowledge, do NOT include a source link.
7. Prioritize reliable multisport official sources such as:
   - https://www.espn.com
   - https://www.olympics.com
   - https://www.fifa.com
   - https://www.icc-cricket.com
   - https://www.nba.com

Provide friendly, clear, and accurate responses. Respond in the same language as the user's query.""")

    def call_model(state: MessagesState):
        print(f"call_model invoked. Message count: {len(state['messages'])}")
        messages = state["messages"]   
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [sys_msg] + messages
        print("Invoking LLM with tool binding...")
        print("\n=== MESSAGES SENT TO LLM ===")
        for i, msg in enumerate(messages):
            print(f"\nMessage {i} ({type(msg).__name__}):")
            print(f"Content: {msg.content}")
        print("=== END MESSAGES ===\n")
        response = llm_with_tools.invoke(messages)
        if hasattr(response, "tool_calls") and response.tool_calls:
            print(f"LLM returned tool calls: {[tc['name'] for tc in response.tool_calls]}")
        else:
            print("LLM returned text response")  
        return {"messages": [response]}
    
    # Router function
    def should_continue(state: MessagesState):
        messages = state["messages"]
        last_message = messages[-1]
        print(f"Router evaluating last message type: {type(last_message).__name__}")
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            print(f"Routing to tools: {len(last_message.tool_calls)} tool call(s)")
            return "tools"
        return "__end__"
    
    # Build graph
    print("Building LangGraph workflow...")
    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")
    
    # Add memory
    memory = MemorySaver()
    compiled_graph = workflow.compile(checkpointer=memory)
    print("LangGraph workflow compiled successfully with memory.")
    return compiled_graph


# Initialize the App
app = FastAPI(
    title = "Lang-Graph External Agent",
    description = "Lang-Graph  Agent Connect Framework Implementation",
    version = "0.1.0",
    docs_url="/docs",
)
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Create the agent
agent = create_agent()


# Root Redirect Endpoint
@app.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse(url="/docs")


# # Discover Agents Endpoint
@app.get("/v1/agents")
async def discover_agents():
    return {
        "agents": [{
            "name": "LangGraph Agent",
            "description": "A LangGraph-based agent that can utilize tools to enhance its capabilities.",
            "provider": {
                "organization": "",
                "url": ""
            },
            "version": "1.0.0",
            "documentation_url": "https://connect.watson-orchestrate.ibm.com/examples#langgraph-agent-with-tool-calling",
            "capabilities": {
                "streaming": True,
                "memory": True
            }
        }]
    }


# Chat Completion Endpoint
@app.post("/v1/chat")
async def chat_completion(request: ChatRequest, x_thread_id: str = Header(None, alias="X-THREAD-ID")):  
    thread_id = x_thread_id or str(uuid.uuid4())
    print(f"Received {len(request.messages)} messages")
    
    # Convert messages to LangChain format
    messages = []
    for msg in request.messages:
        if msg.get("role") == "user":
            messages.append(HumanMessage(content=msg.get("content")))
        elif msg.get("role") == "assistant":
            messages.append(AIMessage(content=msg.get("content")))
        elif msg.get("role") == "system":
            messages.append(SystemMessage(content=msg.get("content")))
    
    # Config with thread_id for memory
    config = {"configurable": {"thread_id": thread_id}}
    
    if request.stream:
        print("Streaming response requested")
        return StreamingResponse(
            stream_agent_response(messages, thread_id, request.model, config),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Thread-ID": thread_id
            }
        )
    else:
        print("Non-streaming response requested")
        result = agent.invoke({"messages": messages}, config)
        final_message = result["messages"][-1]
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": final_message.content
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }


# Utility function to stream agent response
async def stream_agent_response(messages: list, thread_id: str, model: str, config: dict):
    try:
        final_response_content = ""
        async for event in agent.astream_events(
            {"messages": messages},
            config=config,
            version="v2"
        ):
            kind = event.get("event", "")
            print(f"Event received: {kind}")
            
            if kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                
                if chunk and hasattr(chunk, "content") and chunk.content:
                    delta_content = chunk.content
                    final_response_content += delta_content
                    chunk_data = {
                        "id": f"run-{uuid.uuid4()}",
                        "object": "thread.message.delta",
                        "thread_id": thread_id,
                        "model": model,
                        "created": int(time.time()),
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "content": delta_content
                            }
                        }]
                    }
                    
                    yield f"data: {json.dumps(chunk_data)}\n\n"

            elif kind == "on_tool_start":
                tool_name = event.get("name", "unknown")
                print(f"Tool execution started: {tool_name}")
            
            elif kind == "on_tool_end":
                tool_name = event.get("name", "unknown")
                print(f"Tool execution completed: {tool_name}")
        
        print(f"Final response content length: {len(final_response_content)} chars")
        
        final_chunk = {
            "id": f"run-{uuid.uuid4()}",
            "object": "thread.message.delta",
            "thread_id": thread_id,
            "model": model,
            "created": int(time.time()),
            "choices": [{
                "index": 0,
                "delta": {}
            }]
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        print(f"Error during streaming: {str(e)}", exc_info=True)
        error_chunk = {
            "error": {
                "message": str(e),
                "type": "server_error"
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"

if __name__ == "__main__":
    print("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8080)