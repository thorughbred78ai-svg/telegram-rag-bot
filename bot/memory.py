import json

from redis.asyncio import Redis

from .config import settings


redis = Redis.from_url(
    settings.redis_url,
    decode_responses=True,
)


def memory_key(chat_id: int) -> str:
    return f"telegram:memory:{chat_id}"


async def get_memory(chat_id: int) -> list[dict]:
    raw = await redis.get(memory_key(chat_id))

    if not raw:
        return []

    try:
        data = json.loads(raw)
    except Exception:
        return []

    if not isinstance(data, list):
        return []

    return [
        x for x in data
        if isinstance(x, dict)
        and x.get("role") in {"user", "assistant"}
        and isinstance(x.get("content"), str)
    ][-(settings.max_turns * 2):]


async def save_memory(
    chat_id: int,
    history: list[dict],
) -> None:

    history = history[-(settings.max_turns * 2):]

    await redis.set(
        memory_key(chat_id),
        json.dumps(history, ensure_ascii=False),
        ex=settings.memory_ttl,
    )


async def clear_memory(chat_id: int) -> None:
    await redis.delete(memory_key(chat_id))
