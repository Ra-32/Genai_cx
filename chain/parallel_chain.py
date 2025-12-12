from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
llm1 = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    huggingfacehub_api_token=hf_token,
)
llm2= HuggingFaceEndpoint(
    repo_id="Qwen/Qwen1.5-1.8B-Chat",
    task="text-generation",
    huggingfacehub_api_token=hf_token,
)
model1= ChatHuggingFace(llm=llm1)
model2= ChatHuggingFace(llm=llm2)
parser = StrOutputParser()

# template 1
prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 short question answers from the following text \n {text}',
    input_variables=['text']
)

# template 3
prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)

paralle_chain = RunnableParallel(
    {
        'notes': prompt1 | model2 | parser,
        'quiz': prompt2 | model1 | parser,
    }
)
merge_chain = prompt3 | model1 | parser
chain= paralle_chain | merge_chain

result = chain.invoke({
    'text': 'LangChain is a framework for developing applications powered by language models. It '
            'offers a standard interface for all language models, as well as tools to connect '
            'them with other sources of data and computation. With LangChain, developers can '
            'easily build applications that integrate multiple language models and data sources, '
            'enabling more complex and powerful use cases.'
})
print(result)
print(chain.get_graph().print_ascii())