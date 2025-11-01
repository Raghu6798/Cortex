import os
import json
from typing import Dict, Any, List
from dotenv import load_dotenv
from pydantic import BaseModel,Field
import pytest

from agno.tools.api import CustomApiTools
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.run.agent import RunOutput

load_dotenv()


class ToolConfigSchema(BaseModel):
    name: str
    description: str
    api_url: str
    api_method: str
    api_headers: Dict[str, str] = {}
    api_query_params: Dict[str, str] = {}
    api_path_params: Dict[str, str] = {}
    dynamic_boolean: bool = False
    dynamic_variables: Dict[str, str] = {}
    request_payload: str = ""


def substitute_placeholders(template: Any, values: Dict[str, Any]) -> Any:
    """Recursively replaces {{key}} placeholders."""
    if isinstance(template, str):
        for k, v in values.items():
            template = template.replace(f"{{{{{k}}}}}", str(v))
        return template
    if isinstance(template, dict):
        return {k: substitute_placeholders(v, values) for k, v in template.items()}
    if isinstance(template, list):
        return [substitute_placeholders(i, values) for i in template]
    return template
def create_tool_function(schema: ToolConfigSchema):
    """
    Creates a tool function that wraps CustomApiTools.make_request with a
    Pydantic model for explicit, strongly-typed arguments.
    """
    # 1. Dynamically create a Pydantic model for the tool's arguments
    # This makes the tool's signature explicit to the agent and LLM.
    fields = {
        param: (str, Field(..., description=f"Value for {param}"))
        for param in schema.dynamic_variables.keys()
    }
    ArgsModel = type(f"{schema.name.capitalize()}Args", (BaseModel,), fields)

    def tool_func(args: ArgsModel) -> str:
        """
        Tool function that uses CustomApiTools.make_request with dynamic placeholder substitution.
        
        Args:
            args: A Pydantic model containing the dynamic parameters required by the API.
        
        Returns:
            str: JSON string response from CustomApiTools.make_request
        """
        # Convert the Pydantic model to a dictionary for substitution
        dynamic_values = args.model_dump()
        
        # (The rest of the logic remains almost identical to your original function)
        
        # Start with schema's base URL and substitute placeholders
        final_url = substitute_placeholders(schema.api_url, dynamic_values)
        
        # Handle headers
        final_headers = substitute_placeholders(schema.api_headers or {}, dynamic_values)
        
        # Handle query parameters
        final_params_template = schema.api_query_params or {}
        final_params = substitute_placeholders(final_params_template, dynamic_values)
        # Clean up any unsubstituted placeholders
        final_params = {k: v for k, v in final_params.items() if not v.startswith('{{')}
        
        # Handle request body
        final_json_data = None
        if schema.request_payload:
            payload_template = json.loads(schema.request_payload)
            final_json_data = substitute_placeholders(payload_template, dynamic_values)

        # Create a CustomApiTools instance for this call
        call_api_tools = CustomApiTools(
            base_url=None,
            headers=final_headers or None,
        )
        
        # Call make_request
        result = call_api_tools.make_request(
            endpoint=final_url,
            method=schema.api_method.upper(),
            params=final_params or None,
            json_data=final_json_data,
        )
        
        return result
    
    tool_func.__name__ = schema.name
    tool_func.__doc__ = schema.description
    
    return tool_func

async def test_dynamic_tool_injection_weather():
    """Test dynamic placeholder injection with CustomApiTools end-to-end."""
    print("🚀 Starting test_dynamic_tool_injection_weather...")
    
    try:
        nvidia_key = os.getenv("NVIDIANIM_API_KEY")
        weather_key = os.getenv("WEATHER_API_KEY")
        
        if not nvidia_key:
            print("❌ ERROR: NVIDIANIM_API_KEY not set")
            return
        if not weather_key:
            print("❌ ERROR: WEATHER_API_KEY not set")
            return
        
        print("✅ API keys loaded successfully")
        
        schema = ToolConfigSchema(
            name="get_weather",
            description="Fetches weather using OpenWeather API. Requires lat and lon as parameters.",
            api_url="https://api.openweathermap.org/data/2.5/weather",
            api_method="GET",
            api_headers={"Content-Type": "application/json"},
            api_query_params={
                "lat": "{{lat}}",
                "lon": "{{lon}}",
                "appid": weather_key,
                "units": "metric",
                "lang": "en"
            },
            dynamic_boolean=True,
            request_payload=""
        )
        
        print("✅ ToolConfigSchema created")
        
        # Create the tool function wrapper using CustomApiTools
        weather_tool = create_tool_function(schema)
        print("✅ Tool function created", weather_tool)
        
        # Create LLM
        print("🔄 Creating LLM...")
        llm = OpenAIChat(
            id="meta/llama-3.1-8b-instruct",
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=nvidia_key
        )
        print("✅ LLM created")
        
        # Create agent with the custom tool
        print("🔄 Creating agent...")
        agent = Agent(
            model=llm,
            tools=[weather_tool],
        )
        print("✅ Agent created")
        
        # Run the agent
        print("🔄 Running agent with input: 'Get weather for Hyderabad, India. Use latitude 17.3850 and longitude 78.4867.'")
        print("⏳ This may take a while...")
        run_output: RunOutput = await agent.arun(
            input="Get weather for Hyderabad, India. Use latitude 17.3850 and longitude 78.4867."
        )
        print("✅ Agent run completed")
        
        # Validation
        if run_output is None:
            print("❌ ERROR: run_output is None")
            return
        
        if run_output.content is None:
            print("❌ ERROR: run_output.content is None")
            return
        
        if not isinstance(run_output.content, str):
            print(f"❌ ERROR: run_output.content is not a string: {type(run_output.content)}")
            return
        
        print(f"\n{'='*80}")
        print("✅ Agent response:")
        print(f"{'='*80}")
        print(run_output)
        print(f"{'='*80}\n")
        
        # Additional validation: check if the response contains weather-related content
        # (The response might be JSON string from CustomApiTools)
        try:
            # The tool returns a JSON string, which is then passed to the LLM.
            # The LLM's final `run_output.content` will be a natural language summary.
            # Let's inspect the tool call result directly for a more robust test.
            tool_result_content = ""
            if run_output.tools and run_output.tools[0].result:
                tool_result_content = run_output.tools[0].result
            
            if tool_result_content:
                response_data = json.loads(tool_result_content)
                if "data" in response_data and "name" in response_data["data"]:
                    city_name = response_data["data"]["name"]
                    print(f"✅ API call successful for city: {city_name}")
                    assert city_name.lower() == "hyderabad"
                else:
                    print(f"⚠️  API response format unexpected: {response_data}")
            
            # Check for weather keywords in the final LLM response
            assert any(keyword in run_output.content.lower() for keyword in ["weather", "hyderabad", "temperature", "°c"])

        except (json.JSONDecodeError, AssertionError) as e:
            print(f"❌ Validation failed: {e}")
            raise
        
        print("\n✅ Dynamic tool injection test with CustomApiTools passed successfully!")
        return run_output
        
    except Exception as e:
        print(f"\n❌ ERROR occurred: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    import asyncio
    print("\n" + "="*80)
    print("Starting test_dynamic_tool_injection_weather")
    print("="*80 + "\n")
    try:
        result = asyncio.run(test_dynamic_tool_injection_weather())
        print("\n" + "="*80)
        print("Test completed")
        print("="*80)
    except KeyboardInterrupt:
        print("\n❌ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()