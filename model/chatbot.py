from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
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
chat_history = [
    SystemMessage(content="You are a helpful bot please help me ")
]   
while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() in ['exit', 'quit']:
        print("Exiting the chatbot. Goodbye!")
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI:", result.content)

print(chat_history)
