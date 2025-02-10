import os
from dotenv import load_dotenv
from datetime import date
from tavily import TavilyClient
import json

# CrewAI imports
from crewai import Agent as CrewAIAgent
from crewai import Task, Crew
from langchain_community.tools import tool
from prompts import role, goal, instructions, knowledge

# Load environment variables
load_dotenv()

# Initialize Tavily client
tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_api_key)


class Agent:
    def __init__(self, model="gpt-4o-mini"):
        """
        Initialize the CrewAI agent.

        Args:
            model (str): The language model to use
        """
        # Create tools
        self.tools = self._create_tools()

        # Create the CrewAI agent
        self.agent = self._create_crewai_agent(model)

        # Create a generic task for the agent
        self.task = Task(
            description=("Answer the user's query comprehensively, using tools when necessary. "
                         "This is the conversation history: {history}"
                         "This is the user's latest query: {query}"),
            expected_output="A clear, well-formatted answer, incorporating tool results when appropriate.",
            agent=self.agent
        )

        # Create the crew
        self.crew = Crew(
            agents=[self.agent],
            tasks=[self.task],
            verbose=False
        )

        # Conversation history
        self.messages = []

    @staticmethod
    def date_tool():
        """
        Function to get the current date. This tool takes no arguments.
        """
        today = date.today()
        return today.strftime("%B %d, %Y")

    @staticmethod
    def web_search(query):
        """
        This function searches the web for the given query and returns the results.
        The tool takes a search string as a parameter.
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

        @tool("Get Current Date")
        def date_tool_wrapper():
            """Tool to get the current date. This tool takes no arguments."""
            return self.date_tool()

        @tool("Web Search")
        def web_search_wrapper(query: str):
            """
            This tool searches the web for the given query and returns the results.
            The tool takes a search string as a parameter.
            """
            return self.web_search(query)

        return [date_tool_wrapper, web_search_wrapper]

    def _create_crewai_agent(self, model):
        """
        Create a CrewAI agent with the specified configuration.

        Args:
            model (str): The language model to use

        Returns:
            CrewAI Agent
        """
        return CrewAIAgent(
            role=role,
            goal="\n".join([goal,instructions]),
            backstory=knowledge,
            tools=self.tools,
            verbose=False,
            llm=model
        )

    def chat(self, message):
        """
        Send a message and get a response.

        Args:
            message (str): User's input message

        Returns:
            str: Assistant's response
        """
        try:
            # Kickoff the crew with the user's query
            response = self.crew.kickoff(inputs={"query": message, "history": self.messages})

            # Maintain conversation history
            self.messages.append({"role": "user", "content": str(message)})
            self.messages.append({"role": "assistant", "content": str(response)})

            return response

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
            # Reset messages
            self.messages = []

            # Recreate the crew to ensure a fresh state
            self.crew = Crew(
                agents=[self.agent],
                tasks=[self.task],
                verbose=False
            )

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
