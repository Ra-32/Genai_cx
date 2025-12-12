from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
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
    template="write a joke on {topic}:",
    input_variables=['topic'],
)

prompt2=PromptTemplate(
    template="explain the following \n {text}",
    input_variables=['text']
)
chain=RunnableSequence(prompt1,model,parser,prompt2,model,parser)
    

result=chain.invoke({'topic':'AI'})
print(result)