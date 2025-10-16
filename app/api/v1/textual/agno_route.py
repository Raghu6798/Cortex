# from agno.agent import Agent
# from agno.models.openai import OpenAIChat
# from agno.tools.duckduckgo import DuckDuckGoTools

from backend.app.config.settings import settings

# agent = Agent(
#     model=OpenAIChat(id = 'llama-4-scout-17b-16e-instruct',base_url="https://api.cerebras.ai/v1/",api_key=settings.CEREBRAS_API_KEY)
#     markdown=True,
# )
# agent.print_response("Whats happening in France?", stream=True)

from portkey_ai import Portkey

portkey = Portkey(
  api_key =settings.PORTKEY_API_KEY
)

response = portkey.chat.completions.create(
    model = "@cerebras/qwen-3-coder-480b",
    messages = [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is Portkey"}
    ],
    max_tokens = 512
)

print(response.choices[0].message.content)