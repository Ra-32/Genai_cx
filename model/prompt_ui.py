from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate,load_prompt
from dotenv import load_dotenv
import os
import streamlit as st
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen1.5-1.8B-Chat",
    task="text-generation",
    huggingfacehub_api_token=hf_token,
)



model= ChatHuggingFace(llm=llm,temperature=0.7)

st.header("Research Assistant Chatbot")
paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )
style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] )
length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

template=load_prompt("template.json")

# prompt=template.invoke({
#     "paper_input":paper_input,
#     "style_input":style_input,
#     "length_input":length_input
# })


if st.button("summarize"):
    chain= template | model
    result=chain.invoke({
        "paper_input":paper_input,
        "style_input":style_input,
        "length_input":length_input
    })
    
    st.write(result.content)


    
# result = model.invoke("who is virat kohli?")
# print(result.content)
