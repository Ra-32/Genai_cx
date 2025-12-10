from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field

from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
load_dotenv()
hf_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
llm=HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    huggingfacehub_api_token=hf_token,
)
model=ChatHuggingFace(llm=llm)
# create a json schema for structured output
# scheama=[
#     ResponseSchema(name="name",description="the name of the person",type="string"),
#     ResponseSchema(name="age",description="the age of the person",type="number"),
# ]

class Student(BaseModel):
    name: str=Field(description="the name of the student")
    age: int=Field(description="the age of the student")
    subject: str=Field(description="the subject the student is studying")
    degree: str=Field(description="the degree the student is pursuing")
    

parser=PydanticOutputParser(pydantic_object=Student)
# template 1
template=PromptTemplate(
    template="Give me a the name ,age and subjectand degree  of a fictional {place} student\n {format_instructions}\n:",
    input_variable=['place'],
    partial_variables={"format_instructions":parser.get_format_instructions()},
)
# prompt=template.invoke({'place':'sri lankan'})
# result=model.invoke(prompt)
# final_result=parser.parse(result.content)
# print(final_result)

chain=template | model | parser
result=chain.invoke({'place':'indain'})

print(result)

# data validation 