"""Claude, ChatGPT, Gemini and DeepSeek inside Joshua, with your own API key.

All three companies accept the same request format (the one OpenAI made common), so one small
adapter serves them all. Like apple.chat, chat() answers in ollama.chat's shape, so the rest of
Joshua (tools, web research, fact checks, memory) works unchanged.

Your keys and the models you add are kept in providers.json, on this computer only.
A model's value in the menu is "<provider>::<model>", for example "anthropic::claude-sonnet-5-5".
"""
import base64, json, mimetypes, pathlib
import ollama, requests
from ollama._utils import convert_function_to_tool
from i18n import t

FILE = pathlib.Path(__file__).parent / "providers.json"
PROVIDERS = {
    "anthropic": {"name": "Claude", "company": "Anthropic", "url": "https://api.anthropic.com/v1", "keys": "https://console.anthropic.com/settings/keys", "open": False},
    "openai": {"name": "ChatGPT", "company": "OpenAI", "url": "https://api.openai.com/v1", "keys": "https://platform.openai.com/api-keys", "open": False},
    "gemini": {"name": "Gemini", "company": "Google", "url": "https://generativelanguage.googleapis.com/v1beta/openai", "keys": "https://aistudio.google.com/apikey", "open": False},
    "deepseek": {"name": "DeepSeek", "company": "DeepSeek", "url": "https://api.deepseek.com", "keys": "https://platform.deepseek.com/api_keys", "open": True},  # open weights, served by its makers
}
PANEL = "🔑 Claude, ChatGPT, Gemini, DeepSeek"
SEP = "::"


# ---------- keys and models, in providers.json ----------

def settings():
    try:
        return json.loads(FILE.read_text())
    except Exception:
        return {"keys": {}, "models": []}


def save(data):
    FILE.write_text(json.dumps(data, indent=1))
    FILE.chmod(0o600)  # readable only by you


def set_key(provider, key):
    data = settings()
    data["keys"][provider] = key.strip()
    save(data)


def add_model(provider, model):
    data = settings()
    value = provider + SEP + model
    if value not in data["models"]:
        data["models"].append(value)
    save(data)
    return value


def is_api(value):
    return isinstance(value, str) and SEP in value and value.split(SEP)[0] in PROVIDERS


def split(value):
    provider, model = value.split(SEP, 1)
    return provider, model


def label(value):
    provider, model = split(value)
    return f"🔑 {PROVIDERS[provider]['name']} · {model}"


def company(value):
    return PROVIDERS[split(value)[0]]["company"]


def menu(open_weights):
    """The models you added, for the model menu, if their key is still there: open-weight ones, or the others."""
    data = settings()
    return [(label(v), v) for v in data["models"] if is_api(v) and data["keys"].get(split(v)[0]) and PROVIDERS[split(v)[0]]["open"] == open_weights]


def sees_images(value):
    return split(value)[0] != "deepseek"  # DeepSeek's API models read text only; Joshua's local model looks at pictures for them


# ---------- talking to the company's servers ----------

def headers(provider):
    key = settings()["keys"].get(provider)
    if not key:
        raise RuntimeError(t("There is no API key for {name}. Add it in 🔑 Claude, ChatGPT, Gemini, DeepSeek.", name=PROVIDERS[provider]["name"]))
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def catalog(provider):
    """The chat models this key can use, newest names first."""
    r = requests.get(PROVIDERS[provider]["url"] + "/models", headers=headers(provider), timeout=20)
    check(r, provider)
    ids = [m["id"].removeprefix("models/") for m in r.json().get("data", [])]
    skip = ("embedding", "tts", "audio", "realtime", "transcribe", "image", "search", "moderation", "whisper", "dall-e", "davinci", "babbage", "live", "aqa", "instruct")
    if provider == "openai":
        ids = [i for i in ids if i.startswith(("gpt-", "o1", "o3", "o4", "o5", "chatgpt"))]
    if provider == "gemini":
        ids = [i for i in ids if i.startswith("gemini")]
    if provider == "deepseek":
        ids = [i for i in ids if i.startswith("deepseek")]
    return sorted({i for i in ids if not any(s in i for s in skip)}, reverse=True)


def check(r, provider):
    """Turns the company's error into a sentence that says what to do."""
    if r.ok:
        return
    name, code = PROVIDERS[provider]["name"], r.status_code
    try:
        detail = r.json().get("error", {})
        detail = detail.get("message", "") if isinstance(detail, dict) else str(detail)
    except Exception:
        detail = r.text[:300]
    if code in (401, 403):
        text = t("{name} refused the API key. Check it in 🔑 Claude, ChatGPT, Gemini, DeepSeek, or make a new one.", name=name)
    elif code in (402, 429) and any(w in detail.lower() for w in ("credit", "billing", "quota", "balance", "payment")):
        text = t("Your {name} account has no credit left for the API. Add some on the company's website: API use is paid separately from any chat subscription.", name=name)
    elif code == 429:
        text = t("{name} says too many requests: wait a minute and try again.", name=name)
    elif code == 404:
        text = t("{name} doesn't know this model, or your key can't use it. Pick another one.", name=name)
    else:
        text = t("{name} answered with an error ({code}).", name=name, code=code)
    raise RuntimeError(text + (" " + t("Message from {name}:", name=name) + " " + detail if detail else ""))


def image_part(path):
    kind = mimetypes.guess_type(str(path))[0] or "image/png"
    data = base64.b64encode(pathlib.Path(path).read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{kind};base64,{data}"}}


def convert(messages):
    """Joshua's messages (Ollama's shape) into the companies' shape. Their tool calls need ids, so we number them."""
    out, pending = [], []
    for n, m in enumerate(messages):
        role, content = m["role"], m.get("content") or ""
        if role == "assistant" and m.get("tool_calls"):
            calls = []
            for k, c in enumerate(m["tool_calls"]):
                f = c["function"] if isinstance(c, dict) else c.function
                name, args = (f["name"], f["arguments"]) if isinstance(f, dict) else (f.name, f.arguments)
                calls.append({"id": f"call_{n}_{k}", "type": "function", "function": {"name": name, "arguments": json.dumps(dict(args), ensure_ascii=False)}})
            pending = [c["id"] for c in calls]
            out.append({"role": "assistant", "content": content or None, "tool_calls": calls})
            if m.get("thinking"):
                out[-1]["reasoning_content"] = m["thinking"]
        elif role == "tool":
            out.append({"role": "tool", "tool_call_id": pending.pop(0) if pending else f"call_{n}", "content": str(content)})
        elif m.get("images"):
            out.append({"role": role, "content": [{"type": "text", "text": content}] + [image_part(p) for p in m["images"]]})
        else:
            out.append({"role": role, "content": content})
    return out


def reply(provider, content="", calls=(), thinking=None):
    tool_calls = [ollama.Message.ToolCall(function=ollama.Message.ToolCall.Function(name=c["name"], arguments=json.loads(c["arguments"] or "{}"))) for c in calls]
    return ollama.ChatResponse(model=provider, done=True, message=ollama.Message(role="assistant", content=content, thinking=thinking or None, tool_calls=tool_calls or None))


def chat(model, messages=(), tools=None, stream=False, format=None, options=None, **ignored):
    """Answers like ollama.chat: whole, or piece by piece with stream=True."""
    provider, name = split(model)
    messages = [dict(m) for m in messages]
    if format:  # ask for JSON in words: it works the same with all three companies
        messages[-1]["content"] = (messages[-1].get("content") or "") + "\n\nReply with JSON only, no other text" + (f", matching this JSON schema: {json.dumps(format)}" if isinstance(format, dict) else "") + "."
    body = {"model": name, "messages": convert(messages), "stream": stream}
    if tools:
        body["tools"] = [{"type": "function", "function": convert_function_to_tool(f).model_dump(exclude_none=True)["function"]} for f in tools]
    temperature = (options or {}).get("temperature")
    if temperature is not None and provider != "anthropic":  # recent Claude models refuse any temperature but the default
        body["temperature"] = temperature
    r = requests.post(PROVIDERS[provider]["url"] + "/chat/completions", headers=headers(provider), json=body, stream=stream, timeout=300)
    check(r, provider)
    if not stream:
        msg = r.json()["choices"][0]["message"]
        text = (msg.get("content") or "").strip()
        if format:
            text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return reply(provider, text, [c["function"] for c in msg.get("tool_calls") or []], msg.get("reasoning_content"))
    return pieces(r, provider)


def pieces(r, provider):
    """The streamed answer: text as it comes, then the tool calls, put together from their fragments."""
    calls = {}
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        choices = json.loads(data).get("choices") or [{}]
        delta = choices[0].get("delta") or {}
        if delta.get("reasoning_content"):
            yield reply(provider, thinking=delta["reasoning_content"])
        if delta.get("content"):
            yield reply(provider, delta["content"])
        for c in delta.get("tool_calls") or []:
            call = calls.setdefault(c.get("index", len(calls)), {"name": "", "arguments": ""})
            call["name"] += (c.get("function") or {}).get("name") or ""
            call["arguments"] += (c.get("function") or {}).get("arguments") or ""
    if calls:
        yield reply(provider, "", [calls[k] for k in sorted(calls)])
