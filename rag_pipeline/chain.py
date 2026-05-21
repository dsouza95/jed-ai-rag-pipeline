from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from rag_pipeline.settings import settings

prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
    ]
)

model = ChatOllama(model=settings.chain_model)

chain = prompt | model | StrOutputParser()
