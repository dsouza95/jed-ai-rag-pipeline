from pathlib import Path

import chainlit as cl

from rag_pipeline.chain import chain
from rag_pipeline.indexer import index_game, is_game_indexed, list_indexed_games


@cl.on_chat_start
async def on_chat_start():
    indexed_games = list_indexed_games()
    if indexed_games:
        shown, rest = indexed_games[:5], indexed_games[5:]
        names = ", ".join(shown) + (f" and {len(rest)} more" if rest else "")
        hint = f" (already indexed: {names})"
    else:
        hint = ""

    res = await cl.AskUserMessage(
        content=f"Which board game do you want to ask about? {hint}",
    ).send()

    game_name = res.get("output", "").strip() if res is not None else ""
    if not game_name:
        await cl.Message(
            content="No game name provided. Please restart the chat."
        ).send()
        return

    cl.user_session.set("game", game_name)
    if is_game_indexed(game_name):
        await cl.Message(
            content=f"**{game_name}** is already indexed. Ask away!"
        ).send()
        return

    file_res = await cl.AskFileMessage(
        content=f"""I don't have **{game_name}** yet.
        Upload the PDF rulebook to continue.""",
        accept=["application/pdf"],
        max_size_mb=100,
    ).send()

    if not file_res:
        await cl.Message(
            content="""No file received. Please restart
            the chat and upload the rulebook PDF."""
        ).send()
        return

    file = file_res[0]

    msg = await cl.Message(content=f"Indexing **{game_name}**...").send()

    async def on_progress(current: int, total: int) -> None:
        msg.content = f"Indexing **{game_name}**... {current}/{total} chunks"
        await msg.update()

    await index_game(Path(file.path), game_name, on_progress=on_progress)

    msg.content = f"Done! Ask me anything about **{game_name}**."
    await msg.update()


@cl.on_message
async def on_message(message: cl.Message):
    game = cl.user_session.get("game")
    if not game:
        await cl.Message(content="Please restart the chat to select a game.").send()
        return

    msg = cl.Message(content="")
    await msg.send()

    async for token in chain.astream({"input": message.content}):
        await msg.stream_token(token)

    await msg.update()
