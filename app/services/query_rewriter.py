from app.services.chat_memory import ChatMessage
from app.services.llm import LLMClient, LLMMessage

_SYSTEM_PROMPT = (
    "You rewrite the user's last question so it can be understood on its own, "
    "using the conversation for context. Do not answer it. "
    "If it is already clear on its own, return it unchanged. "
    "Reply with the question only."
)


async def rewrite_query(
    llm: LLMClient, history: list[ChatMessage], question: str
) -> str:
    """Turn a follow-up like 'how long does it take?' into a standalone question."""
    if not history:
        return question

    transcript = "\n".join(f"{m.role}: {m.content}" for m in history)
    prompt = (
        f"Conversation so far:\n{transcript}\n\n"
        f"Last question: {question}\n\nStandalone question:"
    )
    rewritten = await llm.complete(
        [
            LLMMessage(role="system", content=_SYSTEM_PROMPT),
            LLMMessage(role="user", content=prompt),
        ],
        temperature=0.0,
    )
    lines = rewritten.strip().splitlines()
    first_line = lines[0].strip() if lines else ""
    return first_line or question
