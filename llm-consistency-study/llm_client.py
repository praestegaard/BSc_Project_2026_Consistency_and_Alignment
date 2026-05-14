"""
LLM client: Wraps OpenAI, Gemini, Anthropic, and Mistral behind a single call_llm() function.
Temperature is left at each provider's default to measure what a normal consumer would get.
"""

import os
import time
import logging
import json
import httpx

from config import MODELS, MAX_RETRIES, RETRY_DELAY_SECONDS, REQUEST_DELAY_SECONDS

logger = logging.getLogger(__name__)


def _call_openai(model_id, prompt, system_prompt):
    import openai

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set")

    client = openai.OpenAI(api_key=api_key)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(model=model_id, messages=messages)
    return response.choices[0].message.content


def _call_google(model_id, prompt, system_prompt):
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY not set")

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_id}:generateContent?key={api_key}"
    )

    body = {"contents": [{"parts": [{"text": prompt}]}]}
    if system_prompt:
        # Gemini puts system instructions at the top level, not in messages
        body["system_instruction"] = {"parts": [{"text": system_prompt}]}

    resp = httpx.post(url, json=body, timeout=120)
    resp.raise_for_status()

    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as exc:
        logger.error("Unexpected Gemini response: %s", json.dumps(data)[:500])
        raise RuntimeError(f"Could not parse Gemini response: {exc}") from exc


def _call_anthropic(model_id, prompt, system_prompt):
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set")

    # max_retries=0, retries handled elsewhere
    client = anthropic.Anthropic(api_key=api_key, max_retries=0)

    kwargs = {
        "model": model_id,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system_prompt:
        kwargs["system"] = system_prompt

    response = client.messages.create(**kwargs)
    return response.content[0].text


def _call_mistral(model_id, prompt, system_prompt):
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise EnvironmentError("MISTRAL_API_KEY not set")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    body = {"model": model_id, "messages": messages}

    resp = httpx.post("https://api.mistral.ai/v1/chat/completions",
                      headers=headers, json=body, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


_CALLERS = {
    "openai": _call_openai,
    "google": _call_google,
    "anthropic": _call_anthropic,
    "mistral": _call_mistral,
}


def call_llm(model_key, prompt, system_prompt=None):
    """Send prompt to model, return response text. Retries on failure."""
    cfg = MODELS[model_key]
    caller = _CALLERS[cfg["provider"]]

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            text = caller(cfg["model_id"], prompt, system_prompt)
            time.sleep(REQUEST_DELAY_SECONDS)
            return text
        except Exception as exc:
            last_error = exc
            logger.warning("Attempt %d/%d for %s failed: %s",
                           attempt, MAX_RETRIES, model_key, exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS * attempt)

    raise RuntimeError(
        f"All {MAX_RETRIES} attempts failed for {model_key}: {last_error}"
    ) from last_error
