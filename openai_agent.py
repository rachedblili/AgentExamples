import os
import openai
import time
import json
from tavily import TavilyClient
from dotenv import load_dotenv
from datetime import date
from prompts import role, goal, instructions, knowledge

# Load environment variables
load_dotenv()

tavily_api_key = os.getenv("TAVILY_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_client = TavilyClient(api_key=tavily_api_key)


class Agent:
    def __init__(self, model="gpt-4o-mini", max_polling_attempts=60, polling_interval=1):
        self.name = "OpenAI Agent"
        self.model = model
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.max_polling_attempts = max_polling_attempts
        self.polling_interval = polling_interval
        self.assistant = self._create_assistant()
        self.thread = self._create_thread()

    ### Tools ###
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
        search_response = tavily_client.search(query)
        results = json.dumps(search_response.get('results', []))
        print(results)
        return results

    ### Create Assistant with tools ###
    def _create_assistant(self, name="Web Search Assistant"):
        """
        Create an assistant with instructions and tool definitions for both the date and web_search functions.
        """
        assistant = self.client.beta.assistants.create(
            name=name,
            instructions="\n".join([role, goal, instructions, knowledge]),
            tools=[
                {"type": "function", "function": {
                    "name": "date",
                    "description": "Get the current date",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }},
                {"type": "function", "function": {
                    "name": "web_search",
                    "description": "Search the web for information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            }
                        },
                        "required": ["query"]
                    }
                }}
            ],
            model=self.model
        )
        return assistant

    ### Internal workings of the agent ###
    def _create_thread(self):
        """Create a new thread (conversation)."""
        thread = self.client.beta.threads.create()
        return thread

    def _add_message(self, thread_id, role, content):
        """Add a message to the specified thread."""
        message = self.client.beta.threads.messages.create(
            thread_id=thread_id,
            role=role,
            content=content
        )
        return message

    def _run_assistant(self, thread_id, assistant_id, instructions=None):
        """
        Start a run by attaching the assistant to the thread.
        Optionally, pass additional instructions for this run.
        """
        run_kwargs = {"assistant_id": assistant_id}
        if instructions:
            run_kwargs["instructions"] = instructions
        run = self.client.beta.threads.runs.create(thread_id=thread_id, **run_kwargs)
        return run

    def _get_response(self, thread_id, run_id):
        """
        Poll for the run status until it completes.
        If the run status is 'requires_action', we handle the tool call based on the tool's name.
        Returns the assistant response or None on failure.
        """
        attempts = 0
        while attempts < self.max_polling_attempts:
            run = self.client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
            status = run.status

            if status == "completed":
                messages = self.client.beta.threads.messages.list(thread_id=thread_id)
                for message in messages.data:
                    if message.role == "assistant":
                        try:
                            return message.content[0].text.value
                        except (IndexError, AttributeError):
                            return None
                return None
            elif status == "requires_action":
                if (run.required_action and
                        run.required_action.submit_tool_outputs and
                        run.required_action.submit_tool_outputs.tool_calls):

                    tool_outputs = self._handle_tool_calls(run)

                    self.client.beta.threads.runs.submit_tool_outputs(
                        thread_id=thread_id,
                        run_id=run_id,
                        tool_outputs=tool_outputs
                    )

            elif status in ["failed", "cancelled", "expired"]:
                print(f"Run ended with status: {status}")
                return None

            time.sleep(self.polling_interval)
            attempts += 1

        print("Polling exceeded maximum attempts.")
        return None

    def _handle_tool_calls(self, run):
        """
        Handles tool function calls required by the assistant during execution.

        Args:
            run (object): The current assistant run instance.

        Returns:
            list: A list of tool outputs, or None if an error occurs.
        """
        tool_outputs = []
        for tool_call in run.required_action.submit_tool_outputs.tool_calls:
            try:
                arguments = json.loads(tool_call.function.arguments or "{}")
                tool_name = tool_call.function.name

                if tool_name == "web_search":
                    query = arguments.get("query", "")
                    result = self.web_search(query)
                elif tool_name == "date":
                    result = self.date_tool()
                else:
                    result = "Unsupported tool."

                tool_outputs.append({
                    "tool_call_id": tool_call.id,
                    "output": result
                })
            except Exception as e:
                print(f"Error processing function call {tool_call.id}: {e}")
                return None

        return tool_outputs

    ### API to the frontend ###
    ### These two methods must be implemented for all agents ###
    def chat(self, message):
        self._add_message(thread_id=self.thread.id, role="user", content=message)
        run = self._run_assistant(thread_id=self.thread.id, assistant_id=self.assistant.id)
        response = self._get_response(thread_id=self.thread.id, run_id=run.id)
        return response

    def clear_chat(self):
        try:
            self.thread = self._create_thread()
            return True
        except Exception as e:
            print(f"Error clearing chat: {e}")
            raise e


# Usage example
def main():
    agent = Agent()

    query = input("You: ")
    while query != "exit":
        response = agent.chat(query)
        print(f"Assistant: {response}")
        query = input("You: ")


if __name__ == "__main__":
    main()