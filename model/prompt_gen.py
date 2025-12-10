from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

template=PromptTemplate(
    template=''''
    please summarize the reasearch paper titled "{paper_input}" with a following specification :
    Explanation Style: {style_input}
    Explanation Length: {length_input}
    1.Mathematical details:
          include relvant mathematical equation if present in the paper.
          Explain the mathematical concepts using simple ,intuitive code snippets.where applicable.
    2.Analogies:
    -use relatable analogies to simplify complex ideas.
    if certain information is not available in the paper ,respond with :"Insufficient information available"
    instead of guessing 
    Ensure the summary is clear ,accurate, and aligned with the provided style and length
    ''',
    input_variables=["paper_input", "style_input", "length_input"]
    
)

template.save("template.json")