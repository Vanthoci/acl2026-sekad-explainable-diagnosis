import os
from typing import Optional

from openai import OpenAI

from utils.env import load_env


load_env()


DEFAULT_MODEL = "gpt-4o-mini"
MODEL_ALIASES = {
    "gpt_4o": "OPENAI_CHAT_MODEL",
    "qwen14b": "OPENAI_CHAT_MODEL",
    "deepseek_v3": "OPENAI_CHAT_MODEL",
    "deepseek_r1": "OPENAI_CHAT_MODEL",
    "qwen32b_r1": "OPENAI_CHAT_MODEL",
    "llama3": "OPENAI_CHAT_MODEL",
    "llama31": "OPENAI_CHAT_MODEL",
    "medi_r1": "OPENAI_CHAT_MODEL",
    "dsv3_old": "OPENAI_CHAT_MODEL",
    "geminipro": "OPENAI_CHAT_MODEL",
}


def _build_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL".lower())

    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY. Put it in .env before running.")

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url.rstrip("/") + "/"
    return OpenAI(**client_kwargs)


def _resolve_model_name(requested_model: Optional[str]) -> str:
    requested_model = requested_model or os.environ.get("MY_MODEL")
    if not requested_model:
        return os.environ.get("OPENAI_CHAT_MODEL", DEFAULT_MODEL)

    alias_env = MODEL_ALIASES.get(requested_model)
    if alias_env:
        return os.environ.get(alias_env, os.environ.get("OPENAI_CHAT_MODEL", DEFAULT_MODEL))

    return requested_model


def AskChatGPT(input_template, model=None, api_key=None, temperature=0, top_p=1):
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    client = _build_client()
    model_name = _resolve_model_name(model)

    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": input_template,
            }
        ],
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        stop=None,
    )
    return response


def AskGPTAzure(input_template, api_key, azure_endpoint, api_version, model, temperature=0, top_p=1):
    # Keep the old function name for compatibility, but route everything
    # through the unified OpenAI-compatible endpoint configured in .env.
    return AskChatGPT(
        input_template,
        model=model,
        api_key=api_key or None,
        temperature=temperature,
        top_p=top_p,
    )


def one_contact(input_template, model="", log_prefix="main", log_dir="./logs"):
    """Send one prompt to the configured chat model and append a raw log entry."""
    res = AskChatGPT(input_template, model=model if model else None)
    out = res.choices[0].message.content

    import pytz
    from datetime import datetime

    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f"{log_prefix}_log_{datetime.now().strftime('%Y%m%d')}.txt")
    with open(log_path, "a") as f:
        f.write(f"{datetime.now(pytz.timezone('Asia/Shanghai'))}:\n")
        f.write(f">>> User: {input_template}\n\n")
        f.write(f">>> {model or 'GPT'}: {out}\n\n\n\n")
    return out
