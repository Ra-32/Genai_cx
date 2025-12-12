from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
llm1 = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    
    huggingfacehub_api_token=hf_token,
)

llm2= HuggingFaceEndpoint(
    repo_id="Qwen/Qwen1.5-1.8B-Chat",
    task="text-generation",
    
    huggingfacehub_api_token=hf_token,
)

model1 = ChatHuggingFace(llm=llm1)
model2= ChatHuggingFace(llm=llm2)

prompt1=PromptTemplate(
    template="genrate the detailed report on given topic \n :{topic}",
    input_variables=["topic"],
)
prompt2=PromptTemplate(
    template="give the 5 line summary of given text: {text}",
    input_variables=["text"],
)
parser=StrOutputParser()

chain= prompt1 | model2 | parser | prompt2 | model1 | parser
result = chain.invoke({"topic":"unemployment in india"})
print(result)
