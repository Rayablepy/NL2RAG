import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import asyncio
from functools import partial
load_dotenv()

CHAT_MODEL_NAME=os.getenv("CHAT_MODEL_NAME")
MODEL_PATH="./NL2RAG_models/"
ACTUAL_FILE_PATH="./user_data/"
EMBEDDING_MODEL_NAME=os.getenv("EMBEDDING_MODEL_NAME")
EMBEDDING_MODEL_CONTEXT=int(os.getenv("EMBEDDING_MODEL_CONTEXT", 512))
EMBEDDING_MODEL_CHUNK=int(os.getenv("EMBEDDING_MODEL_CHUNK", 64))

tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL_NAME,cache_dir=MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(CHAT_MODEL_NAME,cache_dir=MODEL_PATH)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
hf = HuggingFacePipeline(pipeline=pipe)

class AsyncHFAdapter:
    def __init__(self,model: HuggingFacePipeline):
        self.model=model
    async def ainvoke(self,*args,**kwargs):
        return await asyncio.get_running_loop().run_in_executor(None,partial(self.model.invoke,*args,**kwargs))

async_hf=AsyncHFAdapter(hf)
CHAT_MODEL=ChatHuggingFace(llm=async_hf)
