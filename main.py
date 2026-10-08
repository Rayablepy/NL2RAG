
from loader import query_data
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from config import CHAT_MODEL
tools = [query_data]

agent = create_agent(
    model=CHAT_MODEL,
    tools=tools,
    #may switch to persistent db backed checkpointer
    checkpointer=InMemorySaver()
)
thread={"configurable":{"thread_id":1}}
async def getresponse(user:str) -> str:
    response = await agent.ainvoke({"messages": [{"role": "user", "content": user}]},thread)
    return response["messages"][-1].content
