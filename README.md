# Agent Framework Comparison Project
-----------

## Overview

This project provides a collection of examples demonstrating how to implement the same agent using different frameworks. The goal is to facilitate comparison and evaluation of various agent frameworks by providing a common interface and use case.
A simple UI is included to make testing more fun, but the main goal is to provide an simple way to compare the approaches to agent creation offered by each framework. 

The example agent is very simple. It's a decision support agent which has been given instructions to help a use follow the decision support process described here: https://thedecisionlab.com/reference-guide/psychology/decision-making-process

The agent has two tools it can use:
   - a date tool, which it can use to figure out what today's date it
   - a web search tool, which allows it to help the decision-making process by conducting some research on behalf of the user. 

## Project Structure

The project consists of the following components:

*   **Agent Implementations**: A set of Python modules, each implementing the agent using a different framework (e.g., `langchain_agent.py` files).
*   **Streamlit App**: A user interface application built using Streamlit, allowing users to interact with the agents and compare their behavior.

## Agent Frameworks

The following agent frameworks are currently implemented:

* Anthropic (`anthropic_agent.py`) - An implementation built directly on the Anthropic API.  This is the only one that defaults to a non-OpenAI model and requires an Anthropic API key
* OpenAI (`openai_agent.py`) - An implementation built on top of the OpenAI Assistants API.
* Langchain (`langchain_agent.py`) 
* LangGraph (`langgraph_agent.py`) - This uses the LangGraph prebuilt react agent for simplicity.
* CrewAI (`crewai_agent.py`) - Though CrewAI is meant for multi-agent systems, it is still educational to see this single-agent implementation.
* Pydantic (`pydantic_agent.py`) - Uses the relatively new pydantic-ai framework
* Llama-Index (`llama_index_agent.py`) 
* Atomic Agents (`atomic_agent.py`)

## Getting Started

To run the project, follow these steps:

1.  Clone the repository: `git clone https://github.com/rachedblili/AgentExamples`
2.  Go into the project directory: `cd AgentExamples`
3.  Install the required dependencies: `pip install -r requirements.txt`
4.  To actually run the agents, you need a Tavily AI Research API key.  You can get one here: https://tavily.com/
5.  Create a new file named `.env` in the project directory.
6.  Add your API keys to the `.env` file.  Example:
```commandline
TAVILY_API_KEY="put your tavily key in here"
OPENAI_API_KEY="put your OpenAI key in here"
ANTHROPIC_API_KEY="put your Anthropic key in here"
```
7.  Run the Streamlit app: `streamlit run agent-ui.py`

## Using the App

The Streamlit app provides a simple interface for interacting with the agents:

*   Select an agent type from the sidebar dropdown menu.
*   Type a message in the chat input field to send it to the selected agent.
*   The agent's response will be displayed in the chat history.
*   Use the "Clear Chat" button to reset the conversation.

## Contributing

Contributions to the project are welcome. If you'd like to add a new agent implementation or improve an existing one, please follow these guidelines:

*   Create a new Python module for the agent implementation, following the naming convention `XXX_agent.py`.
*   Ensure the agent implementation conforms to the common interface defined in the `agent-ui.py` file.
*   Submit a pull request with your changes, including a brief description of the new agent implementation.

## License

This project is licensed under the MIT license. 

## Acknowledgments

*   [Langchain](https://github.com/langchain-ai/langchain): A framework for building applications that use large language models.
*   [LangGraph](https://github.com/langchain-ai/langgraph): A library used to create agent and multi-agent workflows.
*   [Llama-Index](https://github.com/run-llama/llama_index): A library for building and interacting with large language models.
*   [OpenAI](https://openai.com/): A leading provider of AI research and development, including the development of large language models.
*   [Anthropic](https://www.anthropic.com/): A company focused on developing and applying AI technology, including large language models.
*   [Pydantic](https://github.com/pydantic/pydantic-ai): A framework and shim to use Pydantic with LLMs.
*   [Atomic Agents](https://github.com/atomic-ai/atomic-agents): A framework for building and interacting with autonomous agents.
*   [CrewAI](https://github.com/crewAIInc/crewAI): A popular framework for building multi-agent systems.
