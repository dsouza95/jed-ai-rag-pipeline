import re
from pathlib import Path

import chainlit as cl
from chromadb.api.types import Metadata

from rag_pipeline import (
    build_context,
    chain,
    index_game,
    is_game_indexed,
    list_indexed_games,
    parse_rulebook,
    retrieve,
)


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

    pages = await parse_rulebook(Path(file.path))
    await index_game(game_name, pages, on_progress=on_progress)

    msg.content = f"Done! Ask me anything about **{game_name}**."
    await msg.update()


def _build_source_elements(chunks: list[tuple[str, Metadata]]) -> list[cl.Text]:
    elements = []

    for i, (doc, meta) in enumerate(chunks):
        name = f"p. {meta.get('page', '')}" if meta.get("page") else f"excerpt {i + 1}"
        context = str(meta.get("context", ""))
        content = doc.removeprefix(f"{context}\n\n")
        parsed_content = _strip_images_from_content(content)
        if parsed_content:
            elements.append(cl.Text(name=name, content=parsed_content, display="page"))

    return elements


def _strip_images_from_content(content: str) -> str:
    return re.sub(
        r"!\[.*?\]\(.*?\.(png|jpe?g|gif|webp|svg|bmp|tiff?)\)",
        "",
        content,
        flags=re.IGNORECASE,
    )


@cl.on_message
async def on_message(message: cl.Message):
    game = cl.user_session.get("game")
    if not game:
        await cl.Message(content="Please restart the chat to select a game.").send()
        return

    chunks = retrieve(game, message.content)
    source_elements = _build_source_elements(chunks)
    context = build_context(chunks)

    msg = cl.Message(content="", elements=source_elements)
    await msg.send()

    async for token in chain.astream(
        {"input": message.content, "context": context, "game": game}
    ):
        await msg.stream_token(token)

    await msg.update()
