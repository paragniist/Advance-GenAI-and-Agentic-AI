import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import ToolMessage
import json
# pip install langchain langchain-mcp-adapters langchain-google-genai python-dotenv
load_dotenv()

SERVERS = {
    "ExpenseTracker": {
        "command": "C:\\Users\\win 10\\AppData\\Local\\Programs\\Python\\Python310\\Scripts\\uv.exe",
        "args": [
            "run",
            "--with",
            "fastmcp",
            "fastmcp",
            "run",
            "E:\\Naresh It course videos\\2. code Patterns\\MCP\\Custom_local_mcp-server\\main.py"
        ],
        "transport": "stdio"
    }
}

async def main():
    
    client = MultiServerMCPClient(SERVERS)
    tools = await client.get_tools()

    named_tools = {tool.name: tool for tool in tools}

    print("Available tools:", named_tools.keys())

    # Gemini 2.5 Flash LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key="YOUR_API_KEY",
        temperature=0
    )

    llm_with_tools = llm.bind_tools(tools)

    prompt = "Could you please list all expenses for the month of March 2026 and summarize the total amount spent in each category?"
    response = await llm_with_tools.ainvoke(prompt)

    if not getattr(response, "tool_calls", None):
        print("\nLLM Reply:", response.content)
        return

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

    final_response = await llm_with_tools.ainvoke([prompt, response, *tool_messages])
    print(f"Final response: {final_response.content}")


if __name__ == '__main__':
    asyncio.run(main())