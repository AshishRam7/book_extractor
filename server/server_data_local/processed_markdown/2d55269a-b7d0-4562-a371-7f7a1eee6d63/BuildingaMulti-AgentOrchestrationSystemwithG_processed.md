# Building a Multi-Agent Orchestration System with Google's A2A Protocol and ADK

This comprehensive guide will walk you through setting up and implementing a multi-agent system using Google's Agent-to-Agent (A2A) protocol and Agent Development Kit (ADK). By the end, you'll be able to create an orchestrator agent that efficiently delegates tasks to specialized worker agents based on task requirements.

# Setting Up Your Environment

Before diving into development, you need to set up your environment with the necessary tools and API access.

# Creating a Gemini AI API Key

To power your intelligent agents with Gemini AI models, you'll need an API key:

- . Log in to the Google AI Studio for Gemini AI and click "Get API Key"
- 2. In the API Key section, click on "Create API Key"
- . Select an existing Google Cloud Project (GCP) or create a new project
- . Copy the generated API key and store it securely this will be used to authenticate your agents [1](#page-10-0)

# Setting Up Your Development Environment

Install the required packages:

```
# Install Google's Agent Development Kit and A2A libraries
pip install google-adk google-a2a
# Additional utilities
pip install requests pydantic
```
Create a basic project structure:

```
multi_agent_system/
├── config.py # Configuration including API keys
├── orchestrator.py # Orchestrator agent implementation
├── worker_agents.py # Worker agent implementations
├── agent_cards/ # A2A agent card definitions
└── main.py # Entry point for your application
```
#### Understanding the A2A Protocol

The Agent-to-Agent (A2A) protocol is an open standard enabling different AI agents to communicate regardless of their framework or vendor . [2](#page-10-1)

# Key A2A Concepts

- . Agent Card: A public metadata file (JSON) describing an agent's capabilities, skills, endpoint URL, and authentication requirements. Typically hosted at /.wellknown/agent.json [2](#page-10-1)
- 2. A2A Server: An agent exposing an HTTP API endpoint implementing A2A methods that receives requests and executes tasks on behalf of other agents [2](#page-10-1)
- . A2A Client: An application or agent that sends requests to an A2A server to initiate tasks or conversations [2](#page-10-1)
- . Task: The fundamental unit of work in A2A. A client starts a task by sending a message to the server [2](#page-10-1)

# Agent Development Kit ADK Fundamentals

Google's Agent Development Kit provides the framework for building sophisticated agent-based applications.

#### Agent Hierarchy in ADK

In ADK, agents exist in a hierarchical parent-child relationship:

- A parent agent can have multiple sub-agents
- Each agent can only have one parent (enforced by the framework)
- Hierarchy is established by passing a list of agent instances to the sub\_agents parameter when initializing a parent agent [3](#page-10-2)

```
# Example of agent hierarchy
from google.adk.agents import LlmAgent, BaseAgent
# Define worker agents
worker1 = LlmAgent(name="ResearchAgent", model="gemini-2.0-flash")
worker2 = LlmAgent(name="WritingAgent", model="gemini-2.0-flash")
# Create orchestrator with workers as sub-agents
orchestrator = LlmAgent(
    name="TaskCoordinator",
    model="gemini-2.0-flash",
    description="I coordinate research and writing tasks.",
    sub_agents=[worker1, worker2]
)
```
#### Workflow Agents for Orchestration

ADK provides specialized workflow agents designed specifically for orchestrating the execution of sub-agents:

- . SequentialAgent: Executes sub-agents one after another in sequence
- 2. ParallelAgent: Executes multiple sub-agents simultaneously
- . LoopAgent: Repeatedly executes sub-agents until a termination condition is met [4](#page-10-3) [3](#page-10-2)

These workflow agents operate based on predefined logic, providing predictable and deterministic execution patterns without requiring LLM reasoning for the orchestration itself . [4](#page-10-3)

# Building Your Multi-Agent Orchestrator

For our multi-agent orchestration system, we'll create an orchestrator that can intelligently route tasks to specialized worker agents.

### Implementing the Orchestrator

```
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import AgentTool
def create_orchestrator(worker_agents):
    """
    Create an orchestrator agent that delegates tasks to worker agents.
    Args:
        worker_agents: List of specialized worker agents
    Returns:
        The configured orchestrator agent
    """
    # Create tools for explicitly invoking each worker agent
    agent_tools = []
    for agent in worker_agents:
        tool = AgentTool(
            agent=agent,
            name=f"use_{agent.name}",
            description=f"Use {agent.name} to {agent.description}"
        )
        agent_tools.append(tool)
    # Create the orchestrator using an LLM Agent
    orchestrator = LlmAgent(
        name="TaskOrchestrator",
        model="gemini-2.0-flash",
        description="I analyze tasks and delegate them to specialized worker agents.",
        instruction="""
        You are an orchestrator agent that routes tasks to specialized worker agents.
        Analyze the user's task and delegate it to the most appropriate worker agent.
        Each worker has different specialties, so choose wisely based on the task require
        """,
        tools=agent_tools,
```

```
sub_agents=worker_agents
)
return orchestrator
```
# Coordinator/Dispatcher Pattern

The Coordinator/Dispatcher pattern is particularly well-suited for our use case:

- A central LlmAgent (Orchestrator) manages several specialized sub\_agents
- The orchestrator routes incoming requests to the appropriate specialist agent
- This can be implemented using LLM-Driven Delegation or Explicit Invocation [3](#page-10-2)

```
# Example of Coordinator pattern implementation
from google.adk.agents import LlmAgent
# Create specialized worker agents
research_agent = LlmAgent(
    name="Research",
    model="gemini-2.0-flash",
    description="Handles information gathering and research tasks."
)
writing_agent = LlmAgent(
    name="Writing",
    model="gemini-2.0-flash",
    description="Handles content creation and writing tasks."
)
data_agent = LlmAgent(
    name="DataAnalysis",
    model="gemini-2.0-flash",
    description="Handles data processing and analysis tasks."
)
# Create the orchestrator with LLM-driven delegation
orchestrator = LlmAgent(
    name="TaskCoordinator",
    model="gemini-2.0-flash",
    instruction="""
    Route user tasks to the most appropriate agent:
    - Use Research agent for information gathering and research
    - Use Writing agent for content creation and writing
    - Use DataAnalysis agent for data processing and analysis
    Select the best agent based on the nature of the task.
    """,
    description="Main task router and coordinator.",
    sub_agents=[research_agent, writing_agent, data_agent]
)
```
#### Creating Worker Agents

Worker agents can be specialized for different types of tasks. Let's implement several worker agents with different capabilities.

### Specialized LLM Agents

```
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
def create_research_agent():
    """Create a specialized research agent."""
    return LlmAgent(
        name="ResearchAgent",
        model="gemini-2.0-flash",
        description="Specializes in finding and organizing information.",
        instruction="""
        You are a research specialist. When assigned a task:
        1. Break it down into research questions
        2. Gather relevant information
        3. Synthesize the findings into a coherent response
        """,
        output_key="research_results" # Store results in state for other agents
    )
def create_writing_agent():
    """Create a specialized writing agent."""
    return LlmAgent(
        name="WritingAgent",
        model="gemini-2.0-flash",
        description="Specializes in creating well-structured written content.",
        instruction="""
        You are a writing specialist. When assigned a task:
        1. Understand the content requirements
        2. Create a logical outline
        3. Write clear, engaging content following best practices
        """
    )
def create_data_analysis_agent():
    """Create a specialized data analysis agent."""
    return LlmAgent(
        name="DataAnalysisAgent",
        model="gemini-2.0-flash",
        description="Specializes in processing and analyzing data.",
        instruction="""
        You are a data analysis specialist. When assigned a task:
        1. Understand the data structure
        2. Apply appropriate analysis techniques
        3. Present insights in a clear, actionable format
        """
    )
```
#### Implementing A2A Communication for Your Agents

To enable agent-to-agent communication using the A2A protocol, we need to implement both the server and client components.

#### Creating an A2A Server Agent

```
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread
class A2AServer:
    def __init__(self, agent, host="localhost", port=8000):
        self.agent = agent
        self.host = host
        self.port = port
        # Create agent card
        self.agent_card = {
            "name": agent.name,
            "description": agent.description,
            "endpoint": f"http://{host}:{port}/agent/execute",
            "skills": ["task_execution"],
            "authentication": {"type": "none"}
        }
        # Set up HTTP server
        class AgentHandler(BaseHTTPRequestHandler):
            def do_GET(self_handler):
                if self_handler.path == "/.well-known/agent.json":
                    self_handler.send_response(200)
                    self_handler.send_header("Content-type", "application/json")
                    self_handler.end_headers()
                    self_handler.wfile.write(json.dumps(self.agent_card).encode())
                else:
                    self_handler.send_response(404)
                    self_handler.end_headers()
            def do_POST(self_handler):
                if self_handler.path == "/agent/execute":
                    content_length = int(self_handler.headers["Content-Length"])
                    post_data = self_handler.rfile.read(content_length)
                    task_request = json.loads(post_data.decode())
                    # Process the task with the agent
                    result = self.agent.execute(task_request["message"])
                    self_handler.send_response(200)
                    self_handler.send_header("Content-type", "application/json")
                    self_handler.end_headers()
                    self_handler.wfile.write(json.dumps({"result": result}).encode())
                else:
                    self_handler.send_response(404)
                    self_handler.end_headers()
```
```
self.server = HTTPServer((host, port), AgentHandler)
    self.server_thread = None
def start(self):
    """Start the A2A server in a separate thread."""
    self.server_thread = Thread(target=self.server.serve_forever)
    self.server_thread.daemon = True
    self.server_thread.start()
    print(f"A2A Server for {self.agent.name} started at http://{self.host}:{self.port
def stop(self):
    """Stop the A2A server."""
    if self.server:
        self.server.shutdown()
        print(f"A2A Server for {self.agent.name} stopped")
```
#### Implementing A2A Client Functionality

```
import requests
class A2AClient:
    def __init__(self):
        self.agent_cards = {} # Cache for agent cards
    def discover_agent(self, agent_url):
        """
        Discover an agent's capabilities by retrieving its agent card.
        Args:
            agent_url: Base URL of the agent
        Returns:
            Agent card dictionary
        """
        if agent_url in self.agent_cards:
            return self.agent_cards[agent_url]
        card_url = f"{agent_url.rstrip('/')}/.well-known/agent.json"
        response = requests.get(card_url)
        if response.status_code == 200:
            agent_card = response.json()
            self.agent_cards[agent_url] = agent_card
            return agent_card
        else:
            raise Exception(f"Failed to discover agent at {agent_url}")
    def execute_task(self, agent_url, message):
        """
        Execute a task on a remote agent.
        Args:
            agent_url: Base URL of the agent
            message: Task message to send
        Returns:
```

```
Task execution result
"""
agent_card = self.discover_agent(agent_url)
endpoint = agent_card["endpoint"]
response = requests.post(
    endpoint,
    json={"message": message}
)
if response.status_code == 200:
    return response.json()["result"]
else:
    raise Exception(f"Task execution failed with status {response.status_code}")
```
### Task Assignment Mechanism

Let's implement the mechanism for the orchestrator to delegate tasks to worker agents.

### LLM-Driven Delegation

In this approach, the LLM (Gemini model) makes the decision about which worker agent should handle a task:

```
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.memory import SimpleMemory
def create_llm_driven_orchestrator(worker_agents):
    """Create an orchestrator using LLM-driven delegation."""
    # Create descriptions of each worker agent
    agent_descriptions = "\n".join([
        f"- {agent.name}: {agent.description}"
        for agent in worker_agents
    ])
    # Create the orchestrator
    orchestrator = LlmAgent(
        name="TaskOrchestrator",
        model="gemini-2.0-flash",
        memory=SimpleMemory(), # Use memory to retain context between interactions
        instruction=f"""
        You are an orchestrator agent that analyzes tasks and delegates them to
        specialized worker agents. Available workers:
        {agent_descriptions}
        For each user task:
        1. Analyze the nature and requirements of the task
        2. Select the most appropriate worker agent
        3. Use the transfer_to_agent function to pass the task to that worker
        4. Include clear instructions about what the worker should do
        If a task requires multiple workers, break it down and delegate each part.
```

```
""",
    sub_agents=worker_agents,
    allow_transfer=True # Enable the LLM to transfer execution to sub-agents
)
return orchestrator
```
# Using SequentialAgent for Task Routing

Alternatively, we can use a SequentialAgent to implement a more structured routing approach:

```
from google.adk.agents import LlmAgent, SequentialAgent
def create_sequential_orchestrator(worker_agents):
    """Create an orchestrator using SequentialAgent for routing."""
    # First, create a router agent to decide which worker to use
    router = LlmAgent(
        name="TaskRouter",
        model="gemini-2.0-flash",
        instruction="""
        Analyze the user's task and determine which specialized agent should handle it.
        Output ONLY the name of the chosen agent in the format: "AGENT: <agent_name&gt
        """,
        output_key="selected_agent" # Store the selection in state
    )
    # Then create a dispatcher agent that sends the task to the selected worker
    dispatcher = LlmAgent(
        name="TaskDispatcher",
        model="gemini-2.0-flash",
        instruction="""
        Take the task and forward it to the agent specified in state['selected_agent'].
        Add any clarifying instructions needed for the worker agent.
        """,
        sub_agents=worker_agents,
        allow_transfer=True
    )
    # Combine router and dispatcher in a sequential workflow
    orchestrator = SequentialAgent(
        name="OrchestratorWorkflow",
        sub_agents=[router, dispatcher]
    )
    return orchestrator
```
### Setting Up the Complete System

Now let's put everything together in a complete multi-agent orchestration system:

```
from google.adk.agents import LlmAgent
from google.adk import ADKConfig
def build_multi_agent_system(gemini_api_key):
    """
    Build a complete multi-agent orchestration system.
    Args:
        gemini_api_key: Your Gemini API key
    Returns:
        The configured orchestrator agent
    """
    # Configure ADK with your API key
    config = ADKConfig(api_key=gemini_api_key)
    # Create worker agents
    research_agent = create_research_agent()
    writing_agent = create_writing_agent()
    data_agent = create_data_analysis_agent()
    worker_agents = [research_agent, writing_agent, data_agent]
    # Create the orchestrator
    orchestrator = create_llm_driven_orchestrator(worker_agents)
    return orchestrator
def main():
    # Load API key from environment or configuration
    gemini_api_key = "YOUR_GEMINI_API_KEY" # Replace with your actual key
    # Build the multi-agent system
    orchestrator = build_multi_agent_system(gemini_api_key)
    # Example interaction
    while True:
        user_input = input("Enter your task (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        # Process the task with the orchestrator
        result = orchestrator.execute(user_input)
        print(f"\nResult: {result}\n")
if __name__ == "__main__":
    main()
```
# Security Considerations

When implementing an A2A-based multi-agent system, security is a critical concern:

- . API Key Protection: Store your Gemini API key securely, never hardcoding it in source files [5](#page-10-0)
- 2. Agent Card Management: Ensure agent cards are properly secured and validated [5](#page-10-0)
- . Authentication: Implement proper authentication mechanisms for A2A communication [5](#page-10-0)
- . Task Execution Integrity: Validate tasks before execution to prevent malicious instructions [5](#page-10-0)

```
# Example of secure Gemini API key handling
import os
from dotenv import load_dotenv
# Load API key from environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("Gemini API key not found in environment variables")
```
# Conclusion

This guide has provided a comprehensive framework for building a multi-agent orchestration system using Google's A2A protocol and Agent Development Kit. By implementing an orchestrator agent that delegates tasks to specialized worker agents, you can create a flexible and powerful system that handles a wide range of tasks efficiently.

The key advantages of this approach include:

- . Modularity: Each agent can be developed, tested, and improved independently
- 2. Specialization: Worker agents can be optimized for specific types of tasks
- . Scalability: New worker agents can be added to expand system capabilities
- . Flexibility: The orchestrator can route tasks based on dynamic requirements

As Google's A2A protocol and ADK continue to evolve, this foundation will allow you to leverage advanced features for even more sophisticated agent interactions and collaborations.

⁂

- 1. <https://www.geminiforwork.gwaddons.com/setup-api-keys/create-geminiai-api-key>
- 2. <https://docs.kanaries.net/articles/build-agent-with-a2a>
- <span id="page-10-0"></span>. <https://google.github.io/adk-docs/agents/multi-agents/>
- 4. <https://google.github.io/adk-docs/agents/workflow-agents/>
- 5. <https://arxiv.org/abs/2504.16902>