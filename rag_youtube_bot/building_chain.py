from langchain_core.runnables import RunnableLambda,RunnableParallel,RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from youtube import model
from chunks_embedding import retriever
from augmentation import retrieved_doc
from augmentation import prompt
parser=StrOutputParser()

def format_doc(doc):
    result="\n\n".join(doc.page_content for  doc in retrieved_doc)
    return result

parallel_chain=RunnableParallel({
    'context':retriever |RunnableLambda(format_doc),
    'question':RunnablePassthrough(),
})

main_chain= parallel_chain | prompt | model | parser

answer=main_chain.invoke("what is the amount of bill of daya?")
print(answer)