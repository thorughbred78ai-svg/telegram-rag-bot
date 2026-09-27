import httpx

from qdrant_client import AsyncQdrantClient

from .config import settings


qdrant = AsyncQdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key,
)


async def embed_query(text: str) -> list[float]:
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            settings.embedding_url,
            json={
                "input": text,
            },
        )

        response.raise_for_status()

        data = response.json()

        return data["embedding"]


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
