from langchain_core.prompts  import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader,TextLoader
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter,CharacterTextSplitter

loader=PyPDFLoader('text_splitter\dl-curriculum.pdf')

doc=loader.load()

splitter=CharacterTextSplitter(
    chunk_size=20,
    chunk_overlap=10,
    separator='-'
        
)

text="""
Yes, engineering students, specifically those in Computer Science (CS) or Information Technology (IT), are eligible for certain RBI internships, particularly the Research Internship Scheme, but not typically the main Summer Internship Program. 
Summer Internship Program vs. Research Internship
The RBI primarily offers two types of internships with different eligibility criteria: 
Summer Internship Program (SIP): This program primarily targets students from traditional fields like Management, Statistics, Law, Commerce, Economics, and Finance. Engineering students are generally not eligible for the standard SIP.
Research Internship Scheme: This scheme is open to individuals with strong quantitative skills, including those with B.E./B.Tech. degrees in relevant fields. The goal is to provide exposure to cutting-edge research in central banking. 
You are eligible if you meet the requirements for the Research Internship Scheme or specific departments under that scheme: 
Department of Statistics and Information Management (DSIM): B.E. / B.Tech. in Computer Science/IT or a postgraduate degree in data science/analytics are eligible to apply.
Strategic Research Unit (SRU): B.Tech. or B.E. graduates with expertise in Computer or Data Analytics are encouraged to apply. Strong programming skills are necessary
"""

result=splitter.split_documents(doc)

# result=splitter.split_text(text)
print(type(doc))
print(len(doc))
print((result))
print(len(result))