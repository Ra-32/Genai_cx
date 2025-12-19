from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv
load_dotenv()

import os
hf_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
print(hf_token)
parser=StrOutputParser()
url='https://www.flipkart.com/apple-macbook-air-m2-16-gb-256-gb-ssd-macos-sequoia-mc7x4hn-a/p/itmdc5308fa78421'

loader=WebBaseLoader(url)
llm=HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-genration",
    huggingfacehub_api_token=hf_token,
)

model=ChatHuggingFace(llm=llm)
template1=PromptTemplate(
    template="give the answer of the following question \n {question} from the following text \n {text}",
    input_variables=['question','text'],
)

doc=loader.load()
print(type(doc))
print(len(doc))

chain=template1 |  model | parser
# print(doc[0].page_content)
result=chain.invoke({"question":"what is the available offer to these product ",'text':doc[0].page_content})

print(result)