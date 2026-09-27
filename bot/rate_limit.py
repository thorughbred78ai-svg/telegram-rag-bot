from redis.asyncio import Redis

from .config import settings


redis = Redis.from_url(
    settings.redis_url,
    decode_responses=True,
)


async def check_rate_limit(chat_id: int) -> bool:
    key = f"telegram:rate:{chat_id}"

    count = await redis.incr(key)

    if count == 1:
        await redis.expire(
            key,
            settings.rate_window,
        )

    return count <= settings.rate_limit
