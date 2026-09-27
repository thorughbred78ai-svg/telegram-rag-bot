import httpx

from qdrant_client import AsyncQdrantClient

from .config import settings


qdrant = AsyncQdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key,
)


async def embed_query(text: str) -> list[float]:
    if not settings.embedding_url:
        raise RuntimeError(
            "EMBEDDING_URL is not configured"
        )

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            settings.embedding_url,
            json={
                "input": text,
            },
        )

        response.raise_for_status()

        data = response.json()

        embedding = data.get("embedding")

        if not isinstance(embedding, list):
            raise RuntimeError(
                "Embedding service returned invalid data"
            )

        return embedding


async def search_qdrant(
    query: str,
) -> list[dict]:

    vector = await embed_query(query)

    result = await qdrant.query_points(
        collection_name=settings.qdrant_collection,
        query=vector,
        limit=settings.top_k,
        score_threshold=settings.score_threshold,
        with_payload=True,
    )

    return [
        {
            "score": point.score,
            "payload": point.payload or {},
        }
        for point in result.points
    ]


def build_context(
    hits: list[dict],
) -> tuple[str, list[str]]:

    used = 0
    blocks = []
    sources = []

    for index, hit in enumerate(hits, start=1):

        payload = hit["payload"]

        filename = payload.get(
            "fileName",
            "未知文件",
        )

        chunk_index = payload.get(
            "chunkIndex",
            "?",
        )

        text = str(
            payload.get(
                "text",
                "",
            )
        )

        text = text.replace(
            "<context>",
            "",
        ).replace(
            "</context>",
            "",
        )

        block = (
            f"[{index}] "
            f"文件：{filename} "
            f"（段落 {chunk_index}）\n"
            f"{text}"
        )

        if (
            used + len(block)
            > settings.max_context_chars
            and blocks
        ):
            break

        blocks.append(block)
        used += len(block)

        if filename not in sources:
            sources.append(filename)

    return "\n\n".join(blocks), sources
