from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.runnables import RunnableBranch,RunnableParallel,RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Annotated,Literal
from dotenv import load_dotenv
import os
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen1.5-1.8B-Chat",
    task="text-generation",
    huggingfacehub_api_token=hf_token,
    stop=["}"],
)
model = ChatHuggingFace(llm=llm)
class Feedback(BaseModel):
    sentiment:Literal["positive","negative"]=Field(description="Give the sentiment of the  feedback")
   
# template 1
pydantic_parser = PydanticOutputParser(pydantic_object=Feedback)
prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':pydantic_parser.get_format_instructions()}
)
str_parser = StrOutputParser()

classifier_chain= prompt1 | model | pydantic_parser


# result= classifier_chain.invoke({'feedback':'this is a beautiful phone'}).sentiment
# print(result)
prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)
chain2=prompt2 | model | str_parser
chain3 = prompt3 | model | str_parser

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model | str_parser),
    (lambda x:x.sentiment == 'negative', prompt3 | model | str_parser),
    RunnableLambda(lambda x: "could not find sentiment")
)


chain= classifier_chain | branch_chain

result=chain.invoke({'feedback':'It keeps malfunctioning, and the overall performance is weak'})
print(result)
chain.get_graph().print_ascii()