import os
from dotenv import load_dotenv
from datetime import date
from tavily import TavilyClient
import json

# Llama-Index imports
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import PromptTemplate


from prompts import role, goal, instructions, knowledge, llama_index_react_prompt

# Load environment variables
load_dotenv()

# Initialize Tavily client
tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_api_key)


class Agent:
    def __init__(self, model="gpt-4o-mini"):
        """
        Initialize the Llama-Index agent.

        Args:
            model (str): The language model to use
        """
        self.name = "Llama-Index Agent"
        # Initialize the language model
        self.llm = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=model
        )

        # Create tools
        self.tools = self._create_tools()

        # Initialize the memory
        chat_memory = ChatMemoryBuffer.from_defaults(
            token_limit=4096
        )

        # Create the agent
        self.agent = ReActAgent.from_tools(
            tools=self.tools,
            llm=self.llm,
            verbose=False,
            memory=chat_memory
        )

        # Customize the system prompt with our own instructions.
        updated_system_prompt = PromptTemplate("\n".join([role, goal, instructions, knowledge, llama_index_react_prompt]))
        self.agent.update_prompts({"agent_worker:system_prompt": updated_system_prompt})
        self.agent.reset()


    @staticmethod
    def date_tool():
        """
        Function to get the current date.
        """
        today = date.today()
        return today.strftime("%B %d, %Y")

    @staticmethod
    def web_search(query):
        """
        This function searches the web for the given query and returns the results.
        """
        # Call Tavily's search and dump the results as a JSON string
        search_response = tavily_client.search(query)
        results = json.dumps(search_response.get('results', []))
        print(f"Web Search Results for '{query}':")
        print(results)
        return results

    def _create_tools(self):
        """
        Create tools for the agent.

        Returns:
            List of tools
        """
        return [
            FunctionTool.from_defaults(
                fn=self.date_tool,
                name="date",
                description="Useful for getting the current date"
            ),
            FunctionTool.from_defaults(
                fn=self.web_search,
                name="web_search",
                description="Useful for searching the web for information"
            )
        ]

    def chat(self, message):
        """
        Send a message and get a response.

        Args:
            message (str): User's input message

        Returns:
            str: Assistant's response
        """
        try:
            # Send message to the agent
            response = self.agent.chat(message)

            return str(response)

        except Exception as e:
            print(f"Error in chat: {e}")
            return "Sorry, I encountered an error processing your request."

    def clear_chat(self):
        """
        Reset the conversation context.

        Returns:
            bool: True if reset was successful
        """
        try:
            # Reset the agent's chat history
            self.agent.reset()
            return True
        except Exception as e:
            print(f"Error clearing chat: {e}")
            return False


def main():
    """
    Example usage demonstrating the agent interface.
    """
    agent = Agent()

    while True:
        query = input("You: ")
        if query.lower() in ['exit', 'quit']:
            break

        response = agent.chat(query)
        print(f"Assistant: {response}")


if __name__ == "__main__":
    main()
