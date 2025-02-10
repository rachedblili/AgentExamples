import os
from dotenv import load_dotenv
from datetime import date
from tavily import TavilyClient
import json

# Pydantic AI imports
from pydantic_ai import Agent as PydanticAgent, RunContext
from prompts import role, goal, instructions, knowledge

# Load environment variables
load_dotenv()

# Initialize Tavily client
tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_api_key)

class Agent:
    def __init__(self, model="gpt-4o-mini"):
        """
        Initialize the Pydantic AI agent.

        Args:
            model (str): The language model to use
        """
        # Create the agent with a comprehensive system prompt
        self.agent = PydanticAgent(
            f'openai:{model}',
            system_prompt="\n".join([
                role,
                goal,
                instructions,
                "You have access to two primary tools: date and web_search.",
                knowledge
            ]),
            deps_type=str,
            result_type=str
        )

        # Create tools
        self._create_tools()

        # Conversation history
        self.messages = []

    def _create_tools(self):
        """
        Create and register tools for the agent.
        """
        @self.agent.tool
        async def date_tool(ctx: RunContext[str]) -> str:
            """Get the current date"""
            today = date.today()
            return today.strftime("%B %d, %Y")

        @self.agent.tool
        async def web_search(ctx: RunContext[str], query: str) -> str:
            """Search the web for information"""
            # Call Tavily's search and dump the results as a JSON string
            search_response = tavily_client.search(query)
            results = json.dumps(search_response.get('results', []))
            print(f"Web Search Results for '{query}':")
            print(results)
            return results

    def chat(self, message):
        """
        Send a message and get a response.

        Args:
            message (str): User's input message

        Returns:
            str: Assistant's response
        """
        try:
            # If we have previous messages, pass them as message history

            result = self.agent.run_sync(message,deps=message, message_history=self.messages)

            # Maintain conversation history
            self.messages.extend(result.new_messages())

            return result.data

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
            self.messages = []
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
