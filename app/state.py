from typing import TypedDict, Annotated, Optional, List
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[List, add_messages]
    country: Optional[str]
    needs_clarification: bool
    retrieved_context: str
    is_general_query: bool