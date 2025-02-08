import os
import anthropic
import json
from tavily import TavilyClient
from dotenv import load_dotenv
from datetime import date
from prompts import system_prompt

# Load environment variables
load_dotenv()

tavily_api_key = os.getenv("TAVILY_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
tavily_client = TavilyClient(api_key=tavily_api_key)


class Agent:
    def __init__(self, model="claude-3-haiku-20240307"):
        """
        Initialize the Anthropic agent.

        Args:
            max_messages (int): Maximum number of messages to keep in context
            model (str): Anthropic model to use
        """
        self.client = anthropic.Anthropic(api_key=anthropic_api_key)
        self.model = model
        self.messages = []

        # Define system prompt and tools
        self.system_prompt = system_prompt

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

    def _prepare_tools(self):
        """
        Prepare tool definitions for the Anthropic API.
        """
        return [
            {
                "name": "date",
                "description": "Get the current date",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "web_search",
                "description": "Search the web for information",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        }
                    },
                    "required": ["query"]
                }
            }
        ]

    def _call_tool(self, tool_name, tool_input):
        """
        Call the appropriate tool based on the tool name.

        Args:
            tool_name (str): Name of the tool to call
            tool_input (dict): Input parameters for the tool

        Returns:
            str: Tool output
        """
        if tool_name == "date":
            return self.date_tool()
        elif tool_name == "web_search":
            return self.web_search(tool_input.get("query", ""))
        else:
            return "Unsupported tool."

    def chat(self, message):
        """
        Send a message and get a response.

        Args:
            message (str): User's input message

        Returns:
            str: Assistant's response
        """
        # Add user message
        self.messages.append({"role": "user", "content": message})

        # Prepare the API call
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=self.system_prompt,
                messages=self.messages,
                tools=self._prepare_tools()
            )

            # Process tool calls if any
            while response.stop_reason == "tool_use":
                tool_outputs = []
                for tool_use in response.content:
                    if tool_use.type == "tool_use":
                        tool_output = self._call_tool(
                            tool_use.name,
                            tool_use.input
                        )
                        tool_outputs.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": tool_output
                        })

                # Make a follow-up call with tool results
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=self.system_prompt,
                    messages=self.messages + [
                        {"role": "assistant", "content": response.content},
                        {"role": "user", "content": tool_outputs}
                    ],
                    tools=self._prepare_tools()
                )

            # Extract and return the response
            assistant_response = response.content[0].text
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
    agent = Agent()

    query = input("You: ")
    while query != "exit":
        response = agent.chat(query)
        print(f"Assistant: {response}")
        query = input("You: ")


if __name__ == "__main__":
    main()
