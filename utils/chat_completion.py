import asyncio
import os
from openai import OpenAI

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-162f233bba5a4bf287ab305abfd45a45")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

DEFAULT_DEEPSEEK_MODEL = os.getenv("DEFAULT_DEEPSEEK_MODEL", "deepseek-reasoner")
DEFAULT_OLLAMA_MODEL = os.getenv("DEFAULT_OLLAMA_MODEL", "deepseek-v2:16b")

DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.14"))

KEYWORDS_MODEL = DEFAULT_DEEPSEEK_MODEL
READER_MODEL   = DEFAULT_DEEPSEEK_MODEL
THINKING_MODEL = DEFAULT_DEEPSEEK_MODEL

def _infer_provider_from_model(model: str) -> str:
    """
    Route provider purely from model name.
    """
    m = (model or "").strip().lower()

    # exact defaults first
    if m == DEFAULT_DEEPSEEK_MODEL.lower():
        return "deepseek"
    if m == DEFAULT_OLLAMA_MODEL.lower():
        return "ollama"

    # heuristic fallback
    # Ollama-local style model names often include tags like :16b, :latest, :8b, etc.
    if ":" in m:
        return "ollama"

    # DeepSeek official API model names are usually plain names like deepseek-reasoner / deepseek-chat
    if m in {"deepseek-reasoner", "deepseek-chat"}:
        return "deepseek"

    # default fallback
    return "deepseek"


def _client_for_model(model: str) -> OpenAI:
    provider = _infer_provider_from_model(model)

    if provider == "ollama":
        return OpenAI(
            api_key=OLLAMA_API_KEY,
            base_url=OLLAMA_BASE_URL,
        )

    return OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    )


def chat_complete(
    model: str,
    sysmsg: str,
    user: str,
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    """
    Sync completion.
    Provider is inferred from model name.
    """
    client = _client_for_model(model)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sysmsg},
            {"role": "user", "content": user},
        ],
        stream=False,
        temperature=temperature,
    )
    msg = response.choices[0].message
    return (msg.content or "").strip()


async def achat_complete(
    model: str,
    sysmsg: str,
    user: str,
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    """
    Async wrapper over sync completion.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: chat_complete(model, sysmsg, user, temperature),
    )