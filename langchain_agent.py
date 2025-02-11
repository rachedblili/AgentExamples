import os
from dotenv import load_dotenv
from datetime import date
from tavily import TavilyClient
import json

# Langchain imports
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from prompts import role, goal, instructions, knowledge, langchain_react_prompt

# Load environment variables
load_dotenv()

# Initialize Tavily client
tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_api_key)




class Agent:
    def __init__(self, model="gpt-4o-mini"):
        """
        Initialize the Langchain agent.

        Args:
            model (str): The language model to use
        """
        self.name = "Langchain Agent"
        # Create tools
        self.tools = self._create_tools()

        # Pull the ReAct prompt template
        base_react_prompt = hub.pull("hwchase17/react")
        base_input_variables = base_react_prompt.input_variables
        # Modify the prompt with additional instructions
        new_prompt = PromptTemplate(
            input_variables=base_input_variables,
            template="\n".join([role, goal, instructions, knowledge, langchain_react_prompt])
        )
        self.prompt = new_prompt

        # Initialize the language model
        self.llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=model,
            temperature=0
        )

        # Create the agent and executor
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt,
            stop_sequence=True,
        )

        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            verbose=False  # Set to True for debugging
        )

        # Conversation history (optional, for consistency with other agents)
        self.messages = []

    @staticmethod
    def date_tool(tool_input={}):  # Accepts anything since some frameworks must pass something.
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
            Tool(
                name="date",
                func=self.date_tool,
                description="Useful for getting the current date"
            ),
            Tool(
                name="web_search",
                func=self.web_search,
                description="Useful for searching the web for information"
            )
        ]

    def _messages_to_str(self):
        """
        Convert the messages history into a readable string for inclusion in the prompt.

        Returns:
            A string of the form:
                user: hello
                assistant: hello, how may I help you?
        """
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.messages])


    def chat(self, message):
        """
        Send a message and get a response.

        Args:
            message (str): User's input message

        Returns:
            str: Assistant's response
        """
        try:
            # Invoke the agent with the message
            response = self.agent_executor.invoke(
                {"input": message, "chat_history": self._messages_to_str()}
            )
            # Extract the output
            assistant_response = response.get('output', 'Sorry, I could not process your request.')
            # Optionally, maintain conversation history
            self.messages.append({"role": "user", "content": message})
            self.messages.append({"role": "assistant", "content": assistant_response})

            return assistant_response

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
        if query.lower() == 'exit':
            break

        response = agent.chat(query)
        print(f"Assistant: {response}")


if __name__ == "__main__":
    main()
