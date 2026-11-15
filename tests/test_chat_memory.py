from redis.asyncio import Redis

from app.services.chat_memory import ChatMemory


def make_memory(redis: Redis, max_turns: int = 10, ttl: int = 60) -> ChatMemory:
    return ChatMemory(redis, max_turns=max_turns, ttl_seconds=ttl)


async def test_new_session_is_empty(redis: Redis) -> None:
    assert await make_memory(redis).get("nobody") == []


async def test_exchange_is_stored_in_order(redis: Redis) -> None:
    memory = make_memory(redis)
    await memory.add_exchange("s1", "hello", "hi there")
    await memory.add_exchange("s1", "refund?", "30 days")
    messages = await memory.get("s1")
    assert [(m.role, m.content) for m in messages] == [
        ("user", "hello"),
        ("assistant", "hi there"),
        ("user", "refund?"),
        ("assistant", "30 days"),
    ]


async def test_history_is_trimmed_to_max_turns(redis: Redis) -> None:
    memory = make_memory(redis, max_turns=2)
    for number in range(1, 5):
        await memory.add_exchange("s1", f"q{number}", f"a{number}")
    contents = [m.content for m in await memory.get("s1")]
    assert contents == ["q3", "a3", "q4", "a4"]


async def test_ttl_is_set(redis: Redis) -> None:
    await make_memory(redis, ttl=60).add_exchange("s1", "q", "a")
    ttl = await redis.ttl("chat:s1")
    assert 0 < ttl <= 60


async def test_sessions_are_isolated(redis: Redis) -> None:
    memory = make_memory(redis)
    await memory.add_exchange("alice", "q-alice", "a-alice")
    await memory.add_exchange("bob", "q-bob", "a-bob")
    assert [m.content for m in await memory.get("alice")] == ["q-alice", "a-alice"]
    assert [m.content for m in await memory.get("bob")] == ["q-bob", "a-bob"]


async def test_clear_removes_only_that_session(redis: Redis) -> None:
    memory = make_memory(redis)
    await memory.add_exchange("alice", "q", "a")
    await memory.add_exchange("bob", "q", "a")
    await memory.clear("alice")
    assert await memory.get("alice") == []
    assert len(await memory.get("bob")) == 2


async def test_clear_unknown_session_is_harmless(redis: Redis) -> None:
    await make_memory(redis).clear("never-existed")


async def test_unicode_and_quotes_survive(redis: Redis) -> None:
    memory = make_memory(redis)
    text = 'नमस्ते 🙂 "quoted" \\ backslash\nnew line'
    await memory.add_exchange("s1", text, text)
    assert [m.content for m in await memory.get("s1")] == [text, text]