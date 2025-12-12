from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.runnables import RunnableParallel,RunnableSequence
from dotenv import  load_dotenv
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
    template="write a tweet about {topic} ",
    input_variables=['topic'],

)

prompt2=PromptTemplate(
    template="generate a linkedin post on these {topic}",
    input_variables=['topic'],
)

paralel_chain=RunnableParallel({
    'tweet': RunnableSequence(prompt1,model,parser),
    'linkedin': RunnableSequence(prompt2, model, parser)
    }
)

result=paralel_chain.invoke({'topic':'unemployement of india'})

# print(result['tweet'])
print(result['linkedin'])