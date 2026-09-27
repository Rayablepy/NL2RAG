import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.agents import create_agent
import torch

load_dotenv()

CHAT_MODEL_NAME=os.getenv("CHAT_MODEL_NAME")
MODEL_PATH="./NL2RAG_models/"
ACTUAL_FILE_PATH="./user_data/"
EMBEDDING_MODEL_NAME=os.getenv("EMBEDDING_MODEL_NAME")
EMBEDDING_MODEL_CONTEXT=int(os.getenv("EMBEDDING_MODEL_CONTEXT", 512))
EMBEDDING_MODEL_CHUNK=int(os.getenv("EMBEDDING_MODEL_CHUNK", 64))

if torch.accelerator.is_available():
    DEVICE = torch.accelerator.current_accelerator(check_available=True)
else:
    DEVICE = torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL_NAME,cache_dir=MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(CHAT_MODEL_NAME,cache_dir=MODEL_PATH).to(DEVICE)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
hf = HuggingFacePipeline(pipeline=pipe)

CHAT_MODEL=ChatHuggingFace(llm=hf)

agent=create_agent(model=CHAT_MODEL)
print(agent.invoke({"messages": [{"role": "user", "content": "Reply by saying hello."}]}))
