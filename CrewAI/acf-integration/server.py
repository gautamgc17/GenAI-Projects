# Import libraries
import os
os.environ["CREWAI_INTERACTIVE"] = "False"
import yaml
import time
import uuid
import json
import queue
import threading
import asyncio
import uvicorn
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, RedirectResponse

from crewai import Agent, Task, Crew, Process, LLM
from crewai.events import AgentExecutionCompletedEvent
from crewai.utilities.events.base_event_listener import BaseEventListener

from models import ChatRequest 
from tools import get_current_datetime, search_knowledgebase, search_internet


# Initialize LLM
WATSONX_URL = os.environ["WATSONX_URL"] = os.getenv("WATSONX_URL")  
WATSONX_API_KEY = WATSONX_APIKEY = os.environ["WATSONX_APIKEY"] = os.environ["WATSONX_API_KEY"] = os.getenv("WATSONX_API_KEY") 
WATSONX_PROJECT_ID = os.environ["WATSONX_PROJECT_ID"] = os.getenv("WATSONX_PROJECT_ID")
WATSONX_MODEL_ID = os.environ["WATSONX_MODEL_ID"] = os.getenv("WATSONX_MODEL_ID", "watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8")   
llm = LLM(
    model = WATSONX_MODEL_ID,
    base_url = WATSONX_URL,
    project_id = WATSONX_PROJECT_ID,
    api_key = WATSONX_API_KEY,
    max_tokens = 16000,
    temperature = 0.1
)


# Import agents and tasks config
files = {
    'agents': 'config/agents.yaml',
    'tasks': 'config/tasks.yaml'
}
configs = {}
for config_type, file_path in files.items():
    with open(file_path, 'r') as file:
        configs[config_type] = yaml.safe_load(file)
agents_config = configs['agents']
tasks_config = configs['tasks']


# Create agent
def create_agent():
    return Agent(
        config = agents_config['sports_rag_agent'],
        tools = [get_current_datetime, search_knowledgebase, search_internet],
        verbose = True,
        llm = llm,
        allow_delegation = False
    )


# Process crew
def process_with_crew(query, chat_history):
    agent = create_agent()
    
    task = Task(
        config = tasks_config['sports_question_task'],
        agent = agent
    )
    
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose = True
    )
    
    result = crew.kickoff(
        inputs = {
            "query": query,
            "chat_history": chat_history
        }
    )
    return result


# Initialize the App
app = FastAPI(
    title = "CrewAI External Agent",
    description = "CrewAI Agent Connect Framework Implementation",
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


# Root Redirect Endpoint
@app.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse(url="/docs")


# Discover Agents Endpoint
@app.get("/v1/agents")
async def discover_agents():
    return {
        "agents": [
            {
                "name": "CrewAI Agent",
                "description": "A CrewAI-based agent that can utilize tools to enhance its capabilities.",
                "provider": {
                    "organization": "",
                    "url": ""
                },
                "version": "1.0.0",
                "documentation_url": "https://connect.watson-orchestrate.ibm.com/examples#crewai-example",
                "capabilities": {
                    "streaming": True
                }
            }
        ]
    }


# Chat Completion Endpoint
@app.post("/v1/chat/completions")
async def chat_completion(request: ChatRequest, x_thread_id: str = Header(None, alias="X-THREAD-ID")):
    thread_id = x_thread_id or str(uuid.uuid4())
    
    user_messages = [msg for msg in request.messages if msg["role"] == "user"]
    if not user_messages:
        return {"error": "No user message found"}
    
    query = user_messages[-1]["content"]

    chat_history = []
    for msg in request.messages:
        if msg.get("role") in ["user", "assistant"]:
            if not any(
                key in msg.get("content", "").lower()
                for key in ["using tools", "tool_calls", "function"]
            ):
                chat_history.append({
                    "role": msg["role"],
                    "content": msg["content"].strip()
                })
    if chat_history and chat_history[-1]["role"] == "user" and chat_history[-1]["content"] == query:
        chat_history.pop()
    
    if request.stream:
        return StreamingResponse(
            stream_crew_response(query, chat_history, thread_id),
            media_type="text/event-stream"
        )
    else:
        result = process_with_crew(query, chat_history)
        
        response = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": WATSONX_MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
        return response


# A listener class that captures CrewAI agent events and streams them as incremental responses.
class CrewAIStreamingListener(BaseEventListener):
    def __init__(self, thread_id, request_id):
        super().__init__()
        self.thread_id = thread_id
        self.request_id = request_id
        self.thread_queue = queue.Queue()
        self.agent_instance_ids = set()
        self.processed_event_ids = set()
        
    def set_crew_context(self, crew, agents):
        self.agent_instance_ids = {id(agent) for agent in agents}
        
    def belongs_to_this_request(self, event):
        if hasattr(event, 'agent'):
            return id(event.agent) in self.agent_instance_ids
        return False
        
    def setup_listeners(self, crewai_event_bus):
        
        @crewai_event_bus.on(AgentExecutionCompletedEvent)
        def on_agent_execution_completed(source, event):
            if not self.belongs_to_this_request(event):
                return
            
            event_id = f"agent-complete-{id(event)}"
            if event_id in self.processed_event_ids:
                return
            self.processed_event_ids.add(event_id)
            
            output = str(event.output)
            print(f"[{self.request_id}] Agent completed with output length: {len(output)}")
            
            words = output.split()
            chunk_size = 3
            
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i+chunk_size])
                if i + chunk_size < len(words):
                    chunk += " "
                
                message_delta = {
                    "id": f"msg-{uuid.uuid4()}",
                    "object": "thread.message.delta",
                    "thread_id": self.thread_id,
                    "model": WATSONX_MODEL_ID,
                    "created": int(time.time()),
                    "choices": [
                        {
                            "delta": {
                                "role": "assistant",
                                "content": chunk
                            }
                        }
                    ]
                }
                self.thread_queue.put(("thread.message.delta", message_delta))


# Runs the agent execution in a background thread and yields streaming SSE responses to the client
async def stream_crew_response(query, chat_history, thread_id):
    request_id = str(uuid.uuid4())
    listener = CrewAIStreamingListener(thread_id, request_id)
    
    thread = threading.Thread(
        target=process_with_crew_thread,
        args=(query, chat_history, listener, request_id)
    )
    thread.daemon = True
    thread.start()
    
    async_queue = asyncio.Queue()
    asyncio.create_task(transfer_queue_items(listener.thread_queue, async_queue))
    
    while True:
        event_data = await async_queue.get()
        if event_data is None:
            yield "data: [DONE]\n\n"
            break
        
        event_type, event_content = event_data
        yield f"event: {event_type}\n"
        yield f"data: {json.dumps(event_content)}\n\n"


# Continuously moves items from a blocking thread queue to an async queue for streaming
async def transfer_queue_items(thread_queue, async_queue):
    while True:
        try:
            item = thread_queue.get(block=True, timeout=0.1)
            if item is None:
                await async_queue.put(None)
                break
            await async_queue.put(item)
        except queue.Empty:
            await asyncio.sleep(0.01)


# Runs the CrewAI workflow synchronously in a thread and pushes streamed deltas to the listener
def process_with_crew_thread(query, chat_history, listener, request_id):
    try:
        from crewai.events import crewai_event_bus
        
        agent = create_agent()
        
        task = Task(
            config = tasks_config['sports_question_task'],
            agent = agent
        )
        
        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose = True
        )
        
        listener.set_crew_context(crew, [agent])
        listener.setup_listeners(crewai_event_bus)
        
        print(f"[{request_id}] Starting crew execution...")
        result = crew.kickoff(
            inputs = {
                "query": query,
                "chat_history": chat_history
            }
        )
        print(f"[{request_id}] Crew execution completed")
        listener.thread_queue.put(None)
        return result
    
    except Exception as e:
        print(f"Error in crew execution for request {request_id}: {str(e)}")
        listener.thread_queue.put(None)


# Run the script
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)