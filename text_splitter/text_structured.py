from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader('text_splitter/dl-curriculum.pdf')

doc=loader.load()

text="""Space exploration has led to incredible scientific discoveries. From landing on the Moon to exploring Mars, humanity continues to push the boundaries of what’s possible beyond our planet.

These missions have not only expanded our knowledge of the universe but have also contributed to advancements in technology here on Earth. Satellite communications, GPS, and even certain medical imaging techniques trace their roots back to innovations driven by space programs.
"""


splitter=RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=2,
    separators=','
)

# result=splitter.split_text(text)
result=splitter.split_documents(doc)
print(len(doc))
print(type(result))
print(result[0].page_content)

print(len(result))

# most widely used

# \n\n para
# \n -line/sentence
# "-"=word
# '.'-charcter