# ===============================
# Imports
# ===============================
from typing import TypedDict, Literal

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

from IPython.display import Image, display


# ===============================
# Gemini 2.5 Flash LLM
# ===============================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0, google_api_key="YOUR_API_KEY")


# ===============================
# Graph State
# ===============================
class State(TypedDict):
    topic: str
    content: str
    feedback: str
    quality: str


# ===============================
# Structured Output Schema
# ===============================
class Feedback(BaseModel):
    grade: Literal["acceptable", "needs_improvement"] = Field(
        description="Decide if the content meets quality standards."
    )
    feedback: str = Field(
        description="If the content needs improvement, provide clear, actionable feedback."
    )


# Augment LLM with structured output
evaluator = llm.with_structured_output(Feedback)


# ===============================
# Nodes
# ===============================
def llm_call_generator(state: State):
    """Generate or improve content"""

    if state.get("feedback"):
        prompt = f"""
        Create high-quality content about the topic below.
        Improve it using this feedback:

        Topic: {state['topic']}
        Feedback: {state['feedback']}
        """
    else:
        prompt = f"""
        Create clear, professional, high-quality content about:
        {state['topic']}
        """

    response = llm.invoke(prompt)

    return {"content": response.content}


def llm_call_evaluator(state: State):
    """Evaluate content quality"""

    result = evaluator.invoke(
        f"""
        Evaluate the quality of the following content.

        Content:
        {state['content']}
        """
    )

    return {
        "quality": result.grade,
        "feedback": result.feedback
    }


# ===============================
# Conditional Routing Logic
# ===============================
def route_content(state: State):
    if state["quality"] == "acceptable":
        return "Accepted"
    else:
        return "Rejected"


# ===============================
# Build LangGraph Workflow
# ===============================
builder = StateGraph(State)

builder.add_node("generator", llm_call_generator)
builder.add_node("evaluator", llm_call_evaluator)

builder.add_edge(START, "generator")
builder.add_edge("generator", "evaluator")

builder.add_conditional_edges(
    "evaluator",
    route_content,
    {
        "Accepted": END,
        "Rejected": "generator"
    }
)

workflow = builder.compile()


# ===============================
# Visualize Graph (Notebook)
# ===============================
display(Image(workflow.get_graph().draw_mermaid_png()))


# ===============================
# Invoke Workflow
# ===============================
final_state = workflow.invoke(
    {"topic": "Benefits of Electric Vehicles"}
)

print("\n===== FINAL CONTENT =====\n")
print(final_state["content"])
