from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate, load_prompt,ChatPromptTemplate

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful{domain} assistant.'),
    ('human', 'explain in simple term what is {topic}')
   
])
prompt=chat_template.invoke({
    "domain":"math", 
    "topic":"calculus"
})
print(prompt)