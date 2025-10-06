from llama_index.core.llms import ChatMessage
from llama_index.core.tools import ToolSelection
from llama_index.core.workflow import Event

class PrepEvent(Event):
    pass

class InputEvent(Event):
    input: list[ChatMessage]

class StreamEvent(Event):
    delta: str

class ToolCallEvent(Event):
    tool_calls: list[ToolSelection]


