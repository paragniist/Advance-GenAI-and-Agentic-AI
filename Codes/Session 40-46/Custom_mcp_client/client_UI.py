import asyncio
import json
import streamlit as st
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage

# Load env vars
load_dotenv()

# ✅ FIXED SERVER CONFIG
SERVERS = {
    "ExpenseTracker": {
        "command": "uv",   # ✅ instead of full path
        "args": [
            "run",
            "fastmcp",
            "run",
            "main.py"
        ],
        "transport": "stdio",
        "cwd": "E:\\Naresh It course videos\\2. code Patterns\\MCP\\Custom_local_mcp-server"
    }
}

# ✅ Async MCP execution
async def run_query(user_query: str):
    try:
        client = MultiServerMCPClient(SERVERS)

        tools = await client.get_tools()
        named_tools = {tool.name: tool for tool in tools}

        # ✅ Get system prompt
        prompts = await client.get_prompt("ExpenseTracker", "expense-assistant")

        system_content = "You are a helpful expense assistant."  # fallback

        for msg in prompts:
            parsed = json.loads(msg.content)
            for inner_msg in parsed.get("messages", []):
                if inner_msg.get("role") == "system":
                    system_content = inner_msg.get("content")
                    break

        # ✅ LLM setup
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0
        )

        llm_with_tools = llm.bind_tools(tools)

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=user_query)
        ]

        response = await llm_with_tools.ainvoke(messages)

        # ✅ If no tool call
        if not getattr(response, "tool_calls", None):
            return response.content

        tool_messages = []

        for tc in response.tool_calls:
            selected_tool = tc["name"]
            selected_tool_args = tc.get("args") or {}
            selected_tool_id = tc["id"]

            result = await named_tools[selected_tool].ainvoke(selected_tool_args)

            tool_messages.append(
                ToolMessage(
                    tool_call_id=selected_tool_id,
                    content=json.dumps(result)
                )
            )

        final_response = await llm_with_tools.ainvoke([
            SystemMessage(content=system_content),
            HumanMessage(content=user_query),
            response,
            *tool_messages
        ])

        return final_response.content

    except Exception as e:
        return f"❌ Error: {str(e)}"


# ✅ STREAMLIT UI
st.set_page_config(page_title="MCP Expense Assistant", layout="wide")

st.title("💰 MCP Expense Assistant")

user_query = st.text_area(
    "Enter your query:",
    placeholder="e.g. Add expense 500 for food"
)

result = None

if st.button("Run Query"):
    if not user_query.strip():
        st.warning("Please enter a query")
    else:
        with st.spinner("Processing..."):
            # ✅ FIXED ASYNC ISSUE
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(run_query(user_query))

        st.success("Response:")
        st.write(result)

# ✅ Chat History
if "history" not in st.session_state:
    st.session_state.history = []

if result and user_query:
    if st.button("Save to History"):
        st.session_state.history.append((user_query, result))

if st.session_state.history:
    st.subheader("History")
    for q, r in reversed(st.session_state.history):
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {r}")
        st.markdown("---")