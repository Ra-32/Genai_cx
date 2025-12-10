from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
import streamlit as st
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
load_dotenv()
hf_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
llm=HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    huggingfacehub_api_token=hf_token,
)
model=ChatHuggingFace(llm=llm)
class Animal(BaseModel):
    name: str=Field(description="the name of the animal")
    species: str=Field(description="the species of the animal")
    habitat: str=Field(description="the natural habitat of the animal")
    diet: str=Field(description="the diet of the animal")

parser=PydanticOutputParser(pydantic_object=Animal)


st.header("Animal Information Generator")

animal_data = {
    "Lion": {
        "species": "Mammal",
        "habitat": ["Savannah", "Forests"],
        "diet": "Carnivore"
    },
    "Eagle": {
        "species": "Bird",
        "habitat": ["Mountains", "Forests"],
        "diet": "Carnivore"
    },
    "Crocodile": {
        "species": "Reptile",
        "habitat": ["Rivers", "Deserts"],
        "diet": "Carnivore"
    },
    "Shark": {
        "species": "Fish",
        "habitat": ["Oceans"],
        "diet": "Carnivore"
    },
    "Frog": {
        "species": "Amphibian",
        "habitat": ["Rivers", "Forests"],
        "diet": "Carnivore"
    },
    "Butterfly": {
        "species": "Insect",
        "habitat": ["Forests", "Gardens"],
        "diet": "Herbivore"
    }
}
# template=PromptTemplate(
#     template="give the me the deatiled information  of {name_input} ,  about a their {habitat_input}  and their diet of {diet_input}.\n :",
#     input_variable=['name_input','habitat_input','diet_input'],
    
# )

name_input = st.selectbox("Select Animal:", list(animal_data.keys()))

species_input = animal_data[name_input]["species"]
habitat_input = st.selectbox("Select Habitat:", animal_data[name_input]["habitat"])
diet_input = animal_data[name_input]["diet"]
template = PromptTemplate(
    template=(
        "Write a detailed, well-structured paragraph describing the following animal.\n"
        "Animal Name: {name}\n"
        "Species: {species}\n"
        "Habitat: {habitat}\n"
        "Diet: {diet}\n\n"
        "Your response should be in a single descriptive paragraph."
    ),
    input_variables=["name", "species", "habitat", "diet"]
)

chain=template | model 

if st.button("Generate Animal Info"):
   
    result = chain.invoke({
        "name": name_input,
        "species": species_input,
        "habitat": habitat_input,
        "diet": diet_input
    })
    st.write(result.content)

