
from loader import query_data
from langchain.agents import create_agent
from config import CHAT_MODEL
tools = [query_data]

agent = create_agent(
    model=CHAT_MODEL,
    tools=tools
)
async def getresponse(user:str) -> str:
    response = await agent.ainvoke({"messages": [{"role": "user", "content": user}]})
    return response["messages"][-1].content
