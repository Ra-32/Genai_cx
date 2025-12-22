from text_splitting import chunks
from langchain_community.vectorstores import FAISS
from youtube import llm_embedding
vector_store=FAISS.from_documents(chunks,llm_embedding)

# print(vector_store.index_to_docstore_id)
# print(vector_store.get_by_ids(['d8a4e824-2e40-41ff-8eeb-82598eaab0e9']))
vector_store.docstore._dict.keys()
doc_id = vector_store.index_to_docstore_id[0]
doc = vector_store.docstore.search(doc_id)


retriever=vector_store.as_retriever(search_type="similarity",search_kwargs={"k":4})



# retriever=vector_store.as_retriever(search_type='similarity',search_kwargs={'k':4})
query="what is deepmind"
result=retriever.invoke(query)
