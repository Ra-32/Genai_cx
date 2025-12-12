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
    template="write a joke on {topic}:",
    input_variables=['topic'],
)

prompt2=PromptTemplate(
    template="explain the following \n {text}",
    input_variables=['text']
)

joke_chain=RunnableSequence(prompt1,model,parser)

paralle_chain=RunnableParallel(
    {
        'joke':RunnablePassthrough(),
        'explanation':RunnableSequence(prompt2,model,parser)
    }
)

final_chain=joke_chain| paralle_chain

result=final_chain.invoke({'topic':'cricket'})
print(result)