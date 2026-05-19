import chainlit as cl

from rag_pipeline.chain import chain


@cl.on_message
async def on_message(message: cl.Message):
    response = await chain.ainvoke({"input": message.content})
    await cl.Message(content=response).send()
