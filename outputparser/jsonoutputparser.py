from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.output_parsers import JsonOutputParser
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
json_parser=JsonOutputParser()

# template 1
template =PromptTemplate(
    template="Give me a the name ,age and city of a fictional person\n {format_instructions}\n:",
    input_variable=[],
    partial_variables={"format_instructions":json_parser.get_format_instructions()},

)
# prompt=template.invoke({})
# result=model.invoke(prompt)
# final_result=json_parser.parse(result.content)
# print(final_result)
chain=template| model | json_parser # behind the parse method is called
result=chain.invoke({})
print(result)
# disadvantage:do not enfore schema validation"