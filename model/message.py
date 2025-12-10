from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from typing import Any, Dict
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen1.5-1.8B-Chat",
    task="text-generation",
    max_new_tokens=100,
    huggingfacehub_api_token=hf_token,
)
model= ChatHuggingFace(llm=llm)
