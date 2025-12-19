from langchain_community.document_loaders import TextLoader
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

loader=TextLoader('document_loader\cricket.txt',encoding='utf-8')
parser=StrOutputParser()

template1=PromptTemplate(
    template="write a funny 2 line joke on the given text \n {poem}",
    input_variables=['text']
)

template2=PromptTemplate(
    template="write a summary for the following poem \n {poem}",
    input_variables=['poem']
)

doc=loader.load()
joke_chain=template1 | model | parser

parallel_chain=RunnableParallel({
    'joke':RunnablePassthrough(),
    'summary':RunnableSequence(template2,model,parser)
}
)

chain= joke_chain | parallel_chain
# print(doc[0].page_content)
# print(doc[0].metadata)
# print(len(doc))
result=chain.invoke({'poem':doc[0].page_content})
print(result)

