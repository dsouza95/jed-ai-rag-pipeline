from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
    ]
)

model = ChatOllama(model="gemma4:e4b")

chain = prompt | model | StrOutputParser()
