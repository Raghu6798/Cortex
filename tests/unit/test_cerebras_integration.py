import os
import json
import re
from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv

load_dotenv()

tools = [
    {
        "type": "function",
        "function": {
            "name": "execute_api_call",
            "strict": True,
            "description": "Dynamically executes any HTTP request (GET, POST, PUT, DELETE, etc.) using structured input parameters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "api_method": {
                        "type": "string",
                        "enum": ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
                        "description": "The HTTP method to use for the request."
                    },
                    "api_url": {
                        "type": "string",
                        "description": "The base URL of the API endpoint, optionally containing path placeholders like {userId}."
                    },
                    "api_path_params": {
                        "type": "object",
                        "description": "Path parameters to replace in the URL.",
                        "additionalProperties": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "number"},
                                {"type": "boolean"},
                                {"type": "null"}
                            ]
                        }
                    },
                    "api_query_params": {
                        "type": "object",
                        "description": "Query parameters to append to the API URL.",
                        "additionalProperties": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "number"},
                                {"type": "boolean"},
                                {"type": "null"}
                            ]
                        }
                    },
                    "api_headers": {
                        "type": "object",
                        "description": "HTTP headers to include in the API request.",
                        "additionalProperties": {"type": "string"}
                    },
                    "api_body": {
                        "type": "object",
                        "description": "Request body payload for POST or PUT requests.",
                        "additionalProperties": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "number"},
                                {"type": "boolean"},
                                {"type": "object"},
                                {"type": "array"},
                                {"type": "null"}
                            ]
                        }
                    },
                    "dynamic_variables": {
                        "type": "object",
                        "description": "Optional dynamic variables used to fill request parameters.",
                        "additionalProperties": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "number"},
                                {"type": "boolean"},
                                {"type": "null"}
                            ]
                        }
                    }
                },
                "required": ["api_url", "api_method"]
            }
        }
    }
]

client = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY"))

ACCOUNT_ID = "d35c440bc0aace347ea83c7b5eff253a"
api_url = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/@cf/deepseek-ai/deepseek-r1-distill-qwen-32b"
api_method = "POST"
api_body = {"prompt": "Where did the phrase Hello World come from?"}
api_headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('CLOUDFLARE_API_KEY')}"
}

messages = [
    {"role": "system", "content": "You are a helpful assistant capable of executing dynamic HTTP API calls."},
    {"role": "user", "content": f"Execute the API call to {api_url} with the method {api_method} and the body {api_body} and the headers {api_headers}."},
]

response = client.chat.completions.create(
    model="llama-4-scout-17b-16e-instruct",
    messages=messages,
    tools=tools,
    parallel_tool_calls=False,
    response_format={"type": "json_object"}  # ✅ required for Cerebras validation
)

print(response.choices[0].message.tool_calls)
