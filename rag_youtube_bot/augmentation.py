from langchain_core.prompts import PromptTemplate
from chunks_embedding import retriever
from youtube import model

question="is the topic of nuclear fusion discussed in this video? if yes then what was discussed"
retrieved_doc=retriever.invoke(question)
retrieved_doc


context="\n\n".join(doc.page_content for doc in retrieved_doc)
prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

prompt_result=prompt.invoke({'context':context,'question':question})
answer=model.invoke(prompt_result)
