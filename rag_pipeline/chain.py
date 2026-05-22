from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from rag_pipeline.settings import settings

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert board game rules assistant for {game}. "
            "Answer questions accurately and concisely based solely on the rulebook "
            "excerpts provided. Each excerpt includes its page number — "
            "cite it when referencing a rule, e.g. (p. 4). "
            "If the answer cannot be determined from the provided excerpts, "
            "say so clearly rather than guessing.",
        ),
        (
            "human",
            "{context}\n\n<question>{input}</question>",
        ),
    ]
)

model = ChatOllama(model=settings.chain_model)

chain = prompt | model | StrOutputParser()
