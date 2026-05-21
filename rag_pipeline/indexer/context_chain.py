from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from rag_pipeline.settings import settings

_CONTEXT_SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    "Here is a section from a board game rulebook:\n\n"
    "<section>\n{section}\n</section>\n\n"
    "Here is a specific passage from that section:\n\n"
    "<chunk>\n{chunk}\n</chunk>\n\n"
    "Write a short 1-2 sentence context that situates this passage within the section. "
    "Be concise and focus on what makes this chunk useful "
    "for answering rules questions."
)

chain = (
    _CONTEXT_SUMMARY_PROMPT
    | ChatOllama(model=settings.chunk_context_model)
    | StrOutputParser()
)
