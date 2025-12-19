from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
load_dotenv()

import os 
hf_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
print(hf_token)

loader=PyPDFLoader('document_loader\dl-curriculum.pdf')
parser=StrOutputParser()

llm=HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-genration",
    huggingfacehub_api_token=hf_token,

)

model=ChatHuggingFace(llm=llm)

template1=PromptTemplate(
    template="give me detailed report on given text \n {text}",
    input_variables=['text']
)

doc=loader.load()

chain=template1 | model | parser

print(type(doc))
print(len(doc))
result=chain.invoke({'text':doc[1].page_content})
print(result)
