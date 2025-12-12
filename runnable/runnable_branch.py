from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch,RunnableParallel,RunnableSequence,RunnablePassthrough
from dotenv import load_dotenv
load_dotenv()
import os
hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
print(hf_token)

llm=HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-genration",
    huggingfacehub_api_token=hf_token,

)

model=ChatHuggingFace(llm=llm)
parser=StrOutputParser()

prompt1=PromptTemplate(
    template="write a detailed report on {topic}",
    input_variables=['topic'],

)

prompt2=PromptTemplate(
    template='summarize the following {text}',
    input_variables=['text'],
)

report_gen_chain= prompt1 | model | parser

branch_chain=RunnableBranch(
    (lambda x: len(x.split())>300, prompt2 | model | parser),
    RunnablePassthrough()
)

final_chain=report_gen_chain | branch_chain

result=final_chain.invoke({'topic':'ai'})
print(result)