"""
Framework models and registry for dynamic agent framework support.
"""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum

class FrameworkStatus(str, Enum):
    """Framework availability status."""
    ENABLED = "enabled"
    DISABLED = "disabled"
    COMING_SOON = "coming_soon"
    BETA = "beta"

class FrameworkEndpoint(BaseModel):
    """Endpoint configuration for a framework."""
    method: str = "POST"
    path: str
    description: str
    required_fields: List[str] = []
    optional_fields: List[str] = []

class FrameworkConfig(BaseModel):
    """Configuration for an agent framework."""
    name: str
    description: str
    logo_url: str
    status: FrameworkStatus = FrameworkStatus.ENABLED
    endpoint: FrameworkEndpoint
    supported_providers: List[str] = ["openai"]
    default_model: str = "gpt-4o-mini"
    documentation_url: Optional[str] = None
    features: List[str] = []

class FrameworkRegistry:
    """Registry for managing available agent frameworks."""
    
    _frameworks: Dict[str, FrameworkConfig] = {}
    
    @classmethod
    def register(cls, framework_id: str, config: FrameworkConfig) -> None:
        """Register a new framework."""
        cls._frameworks[framework_id] = config
    
    @classmethod
    def get_framework(cls, framework_id: str) -> Optional[FrameworkConfig]:
        """Get a framework by ID."""
        return cls._frameworks.get(framework_id)
    
    @classmethod
    def get_all_frameworks(cls) -> Dict[str, FrameworkConfig]:
        """Get all registered frameworks."""
        return cls._frameworks.copy()
    
    @classmethod
    def get_enabled_frameworks(cls) -> Dict[str, FrameworkConfig]:
        """Get only enabled frameworks."""
        return {
            f_id: config 
            for f_id, config in cls._frameworks.items() 
            if config.status == FrameworkStatus.ENABLED
        }
    
    @classmethod
    def get_framework_list(cls) -> List[Dict[str, Any]]:
        """Get frameworks as a list for API responses."""
        return [
            {
                "id": f_id,
                "name": config.name,
                "description": config.description,
                "logo_url": config.logo_url,
                "status": config.status.value,
                "supported_providers": config.supported_providers,
                "default_model": config.default_model,
                "features": config.features
            }
            for f_id, config in cls._frameworks.items()
        ]

# Initialize default frameworks
def initialize_default_frameworks():
    """Initialize the default framework configurations."""
    
    # LangChain Framework
    FrameworkRegistry.register("langchain", FrameworkConfig(
        name="LangChain Agent",
        description="Use the powerful and flexible LangChain agent framework with tools and memory.",
        logo_url="https://registry.npmmirror.com/@lobehub/icons-static-png/latest/files/dark/langchain-color.png",
        status=FrameworkStatus.ENABLED,
        endpoint=FrameworkEndpoint(
            path="/chat/invoke",
            description="Invoke LangChain agent with tools and memory",
            required_fields=["api_key", "model_name", "message"],
            optional_fields=["system_prompt", "temperature", "top_p", "base_url"]
        ),
        supported_providers=["openai", "google", "groq", "ollama", "llama_cpp"],
        default_model="gpt-4o-mini",
        features=["tools", "memory", "streaming", "multi-modal"]
    ))
    
    # LlamaIndex Framework
    FrameworkRegistry.register("llama_index", FrameworkConfig(
        name="LlamaIndex Workflow",
        description="Build with LlamaIndex's event-driven ReAct agent for complex workflows.",
        logo_url="https://registry.npmmirror.com/@lobehub/icons-static-png/latest/files/dark/llamaindex-color.png",
        status=FrameworkStatus.ENABLED,
        endpoint=FrameworkEndpoint(
            path="/llama-index-workflows/react-agent",
            description="LlamaIndex ReAct agent workflow",
            required_fields=["openai_api_key", "message"],
            optional_fields=["system_prompt", "temperature"]
        ),
        supported_providers=["openai"],
        default_model="gpt-4o-mini",
        features=["workflows", "react", "reasoning", "tools"]
    ))
    
    # Pydantic AI Framework (Coming Soon)
    FrameworkRegistry.register("pydantic_ai", FrameworkConfig(
        name="Pydantic AI",
        description="Leverage Pydantic for structured AI outputs with type safety.",
        logo_url="https://pbs.twimg.com/profile_images/1884966723746435073/x0p8ngPD_400x400.jpg",
        status=FrameworkStatus.COMING_SOON,
        endpoint=FrameworkEndpoint(
            path="/pydantic-ai/invoke",
            description="Pydantic AI structured outputs",
            required_fields=["api_key", "model_name", "message"],
            optional_fields=["system_prompt", "schema"]
        ),
        supported_providers=["openai", "anthropic"],
        default_model="gpt-4o-mini",
        features=["structured_outputs", "type_safety", "validation"]
    ))
    
    # LangGraph Framework (Coming Soon)
    FrameworkRegistry.register("langgraph", FrameworkConfig(
        name="LangGraph",
        description="Build stateful, multi-actor applications with complex workflows.",
        logo_url="https://registry.npmmirror.com/@lobehub/icons-static-png/latest/files/dark/langgraph-color.png",
        status=FrameworkStatus.COMING_SOON,
        endpoint=FrameworkEndpoint(
            path="/langgraph/invoke",
            description="LangGraph multi-actor workflow",
            required_fields=["api_key", "model_name", "message"],
            optional_fields=["system_prompt", "graph_config"]
        ),
        supported_providers=["openai", "anthropic"],
        default_model="gpt-4o-mini",
        features=["stateful", "multi_actor", "workflows", "conditionals"]
    ))
    
    # Google ADK Framework (Coming Soon)
    FrameworkRegistry.register("adk", FrameworkConfig(
        name="Google ADK",
        description="Explore Google's Agent Development Kit for enterprise-grade agents.",
        logo_url="https://google.github.io/adk-docs/assets/agent-development-kit.png",
        status=FrameworkStatus.COMING_SOON,
        endpoint=FrameworkEndpoint(
            path="/google-adk/invoke",
            description="Google Agent Development Kit",
            required_fields=["api_key", "model_name", "message"],
            optional_fields=["system_prompt", "adk_config"]
        ),
        supported_providers=["google"],
        default_model="gemini-pro",
        features=["enterprise", "google_integration", "advanced_reasoning"]
    ))

# Initialize frameworks on module import
initialize_default_frameworks()

