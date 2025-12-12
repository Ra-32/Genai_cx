from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch,RunnableParallel,RunnableSequence,RunnablePassthrough,RunnableLambda
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

def word_count(text):
    return len(text.split())

prompt1=PromptTemplate(
    template='write a joke on {topic}',
    input_variables=['topic'],

)



joke_gen_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(word_count)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = final_chain.invoke({'topic':'AI'})

final_result = """{} \n word count - {}""".format(result['joke'], result['word_count'])

print(final_result)
