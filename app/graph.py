from langgraph.graph import StateGraph, END
from app.state import AgentState
from app.nodes import check_country, clarify_node, retrieve_node, generate_node


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("check_country", check_country)
    workflow.add_node("clarify", clarify_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    workflow.set_entry_point("check_country")

    workflow.add_conditional_edges(
        "check_country",
        lambda s: "clarify" if s.get("needs_clarification") else "retrieve",
        {"clarify": "clarify", "retrieve": "retrieve"}
    )

    workflow.add_edge("clarify", END)
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()