"""Apple Intelligence: the language model built into macOS 26. Free, private, already on the Mac.

Joshua talks to it through a small helper program, mac/joshua-apple (source: mac/AppleModel.swift).
The model is small (about 3 billion parameters) and reads at most about 4,000 tokens, words and
pieces of words, counting its answer. So it gets a short version of the chat, and no tools.
"""
import json, pathlib, subprocess
import ollama
from i18n import t

NAME = "apple"
LABEL = "🍎 Apple Intelligence"
HELPER = pathlib.Path(__file__).parent / "mac" / "joshua-apple"
BUDGET = 7000  # characters of chat it reads (about 2,000 tokens), leaving room for the answer
SYSTEM = "You are Joshua, a helpful assistant running privately on the user's Mac. Today is {today}. Answer clearly and concisely, in the language of the user's last message."
STATUS = {}


def why_not():
    """Empty when the model can be used, otherwise the reason. Asked once per start."""
    if "why" not in STATUS:
        try:
            out = subprocess.run([str(HELPER), "check"], capture_output=True, text=True, timeout=20).stdout.strip()
            STATUS["why"] = "" if out == "available" else out or "needs macOS 26"
        except Exception:  # no helper (Linux), or macOS older than 26: the program can't start
            STATUS["why"] = "needs macOS 26"
    return STATUS["why"]


def shorten(text, limit):
    """Keep the beginning and the end: the question and the instructions usually sit there."""
    return text if len(text) <= limit else text[:limit * 2 // 3] + "\n[…]\n" + text[-limit // 3:]


def example(schema):
    """{"type": "object", "properties": {"search": {"type": "boolean"}}} becomes {"search": true}."""
    kind = schema.get("type")
    if kind == "object":
        return {k: example(v) for k, v in schema.get("properties", {}).items()}
    if kind == "array":
        return [example(schema.get("items", {}))]
    return {"boolean": True, "integer": 0, "number": 0}.get(kind, "...")


def facts(notes):
    """Only the remembered facts: not Joshua's long heading, nor the lines about tools this model can't use."""
    lines = notes.strip().splitlines()[1:]
    return "\n".join(line for line in lines if "read_memory" not in line and "remember or forget" not in line)


def fit(messages, format, budget):
    """Turn Joshua's chat into the instructions and the prompt for one answer, within the budget."""
    import agent
    turns = [m for m in messages if m["role"] in ("user", "assistant") and m.get("content")]
    prompt = str(turns.pop()["content"]) if turns else ""
    if format:  # a filled-in example works better than a schema for a small model
        prompt += "\n\nReply with JSON only, no other text" + (f", shaped like this example: {json.dumps(example(format))}" if isinstance(format, dict) else "") + "."
    prompt = shorten(prompt, budget * 3 // 4)
    system = next((str(m["content"]) for m in messages if m["role"] == "system"), "")
    notes = system[system.find("\n\nWhat you remember about the user"):] if "What you remember about the user" in system else ""  # only if this chat may use them
    instructions = SYSTEM.format(today=agent.get_datetime())
    if notes:
        instructions += "\n\nNotes about the USER you are talking to (not about you, Joshua):" + "\n" + shorten(facts(notes), 1000)
    left, history = budget - len(prompt) - len(instructions), []
    for m in reversed(turns):  # the most recent messages first, until the budget runs out
        line = ("User: " if m["role"] == "user" else "Assistant: ") + shorten(str(m["content"]), 1200)
        if len(line) > left:
            break
        history.insert(0, line)
        left -= len(line)
    if history:
        instructions += "\n\nThe conversation so far:\n\n" + "\n\n".join(history)
    return instructions, prompt


ERRORS = {
    "too_long": "This is too long for Apple's model, which reads only about 4,000 tokens. Start a new chat, or pick a bigger model in the menu.",
    "language": "Apple's model doesn't speak this language yet. Pick another model in the menu.",
    "refused": "Apple's model refused to answer this: its safety filter stopped it. Pick another model in the menu.",
}


def pieces(messages, format, temperature):
    budget, sent = BUDGET, False
    while True:
        instructions, prompt = fit(messages, format, budget)
        request = json.dumps({"instructions": instructions, "prompt": prompt, "temperature": temperature})
        proc = subprocess.Popen([str(HELPER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        proc.stdin.write(request)
        proc.stdin.close()
        error = None
        for line in proc.stdout:
            msg = json.loads(line)
            if "delta" in msg:
                sent = True
                yield msg["delta"]
            error = msg.get("error", error)
        proc.wait()
        if error == "too_long" and not sent and budget > 1500:  # try again with less of the chat
            budget //= 2
            continue
        if error:
            raise RuntimeError(t(ERRORS[error]) if error in ERRORS else t("Apple's model stopped with an error:") + " " + str(msg.get("message", error)))
        return


def chat(model=NAME, messages=(), stream=False, format=None, options=None, **ignored):
    """Answers like ollama.chat, so the rest of Joshua doesn't need to know which model it is."""
    temperature = (options or {}).get("temperature")
    reply = lambda text: ollama.ChatResponse(model=NAME, done=True, message=ollama.Message(role="assistant", content=text))
    if stream:
        return (reply(piece) for piece in pieces(messages, format, temperature))
    text = "".join(pieces(messages, format, temperature)).strip()
    if format:  # small models sometimes wrap JSON in ``` fences
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:  # and sometimes copy the schema around the answer
            data = json.loads(text)
            if isinstance(format, dict) and "properties" in data and "properties" not in format.get("properties", {}):
                text = json.dumps(data["properties"])
        except ValueError:
            pass
    return reply(text)
