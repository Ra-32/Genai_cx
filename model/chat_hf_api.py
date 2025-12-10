from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os


load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
print(hf_token)
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-Nemo-Base-2407",
    task="text-generation",
    max_new_tokens=100,
    huggingfacehub_api_token=hf_token,
)

# model = ChatHuggingFace(llm=llm)

result = llm.invoke("What is the capital of France?")

print(result)

