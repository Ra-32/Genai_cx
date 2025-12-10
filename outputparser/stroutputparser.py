from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
load_dotenv()
hf_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
print(hf_token)

llm=HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    huggingfacehub_api_token=hf_token,
)
model=ChatHuggingFace(llm=llm)

parser=StrOutputParser()
# template 1
template1=PromptTemplate(
    template="write a detail report on :{topic}",
    input_variables=["review"],
)
# template 2
template2=PromptTemplate(
    template="write a five line summary on the following text:\n{text}",
    input_variables=["text"],
)
# prompt1=template1.invoke({'topic':'black hole'})

# result=model.invoke(prompt1)

# prompt2=template2.invoke({'text':result.content})

# result1=model.invoke(prompt2)

# print(result1.content)
chain= template1 | model | parser| template2| model |parser

result=chain.invoke({'topic':'virat kohli'})
print(result)