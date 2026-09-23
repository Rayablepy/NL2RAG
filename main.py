
from config import CHAT_MODEL_NAME
from loader import query_data
from langchain.agents import create_agent
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

tools = [query_data]

model_id = CHAT_MODEL_NAME
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
hf = HuggingFacePipeline(pipeline=pipe)

model=ChatHuggingFace(llm=hf)

agent = create_agent(
    model=model,
    tools=tools
)
async def getresponse(user:str) -> str:
    response = await agent.ainvoke({"messages": [{"role": "user", "content": user}]})
    return response["messages"][-1].content
