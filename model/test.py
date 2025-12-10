import langchain
import os
from dotenv import load_dotenv
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
print("Hugging Face Token:", hf_token)
print("LangChain version:", langchain.__version__) 
