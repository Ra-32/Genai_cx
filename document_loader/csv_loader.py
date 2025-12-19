from langchain_community.document_loaders import TextLoader,CSVLoader
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel,RunnableSequence,RunnablePassthrough

import os 
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
load_dotenv()

hf_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
print(hf_token)

llm=HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-genration",
    huggingfacehub_api_token=hf_token,
)

model=ChatHuggingFace(llm=llm)

loader=CSVLoader('document_loader\Social_Network_Ads.csv')

template1=PromptTemplate(
    template="answer the following question \n {question} from given csv file {csv}",
    input_variables=['question','csv']
)
parser=StrOutputParser()
chain=template1 | model | parser

doc=loader.load()
print(doc[1].page_content)
print(len(doc))

result=chain.invoke({'question':'what is the  total purchased salary','csv':doc[0].page_content})
print(result)