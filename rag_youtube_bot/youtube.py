from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpointEmbeddings,HuggingFaceEndpoint,ChatHuggingFace
from langchain_community.document_loaders import YoutubeLoader,PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled
from dotenv import load_dotenv
import os

load_dotenv()
hf_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")


llm_text=HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-genration",
    huggingfacehub_api_token=hf_token,
)

model=ChatHuggingFace(llm=llm_text)

llm_embedding=HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-mpnet-base-v2",
    task="feature-extraction",
    huggingfacehub_api_token=hf_token,
)

video_id="J_HSCiaOo4A"
yutube_api=YouTubeTranscriptApi()

try:
   transcript_list=yutube_api.fetch(video_id=video_id,languages=['en','hi'])
   
except TranscriptsDisabled as e:
   print(e.ERROR_MESSAGE)


new_transcript_list=[]
for i in transcript_list:
   new_transcript_list.append(i.text)

trancript=" ".join(chunks for chunks in new_transcript_list)


