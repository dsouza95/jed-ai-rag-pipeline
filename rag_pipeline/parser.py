from pathlib import Path

from llama_cloud import AsyncLlamaCloud
from llama_cloud.types import ParsingGetResponse


async def parse_rulebook(rulebook_file_path: Path) -> ParsingGetResponse:
    client = AsyncLlamaCloud()
    file = await client.files.create(file=rulebook_file_path, purpose="parse")

    parsed_file = await client.parsing.parse(
        file_id=file.id,
        tier="agentic",
        version="latest",
        expand=["markdown_full"],
    )

    return parsed_file
