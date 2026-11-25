"""Training a small language model from zero, with Apple's MLX. Run by scratch.py as a separate process:

  python scratch_job.py prepare DIR SOURCE STAGE    texts -> numbers. SOURCE is a folder, or wikipedia:LANG:MILLIONS (words).
                                                    Stage 1 also learns the vocabulary (tokenizer.json); stage 2 reuses it.
  python scratch_job.py train DIR SIZE STEPS STAGE MIX
                                                    stage 1: random weights -> trained on train.bin
                                                    stage 2: the stage-1 model -> trained on train2.bin, with MIX (0-1) of
                                                    each batch still from train.bin; a copy is kept at 8 points along the way
  python scratch_job.py try DIR PROMPT WHICH        a version of the model continues a piece of text

Both trainings pick up where they stopped. Progress is printed in mlx-lm's format ("Iter 10: Train loss 1.234"),
so the page reads it like fine-tuning.
"""
import hashlib, json, pathlib, shutil, subprocess, sys, time

import numpy as np

HERE = pathlib.Path(__file__).parent
TYPES = {".txt", ".md", ".pdf", ".epub", ".docx", ".odt", ".html", ".htm", ".rtf"}
END = "<|end|>"  # marks where one document ends and the next begins
VOCAB = 8192
CTX = 512  # tokens the model sees at once
BATCH = 8  # windows per step: 8 × 512 = 4,096 tokens. On a 16 GB M4, 32 used 14 GB and was no faster per token
WIKI = ("wikimedia/wikipedia", "20231101")  # Wikipedia as plain text, one row per article, on Hugging Face
SNAPSHOTS = 8  # copies of the model kept along the second stage

# the three sizes: width of each layer, how many layers, attention heads, width of the middle of each layer
SIZES = {"10M": (256, 8, 4, 1024), "25M": (384, 10, 6, 1280), "50M": (512, 12, 8, 1792)}
LR = {"10M": 1e-3, "25M": 8e-4, "50M": 6e-4}
if __import__("os").environ.get("JOSHUA_SCRATCH_TEST"):  # tiny, for testing the whole path on a computer without Apple's chip
    SIZES, CTX, BATCH = {k: (64, 2, 2, 128) for k in SIZES}, 64, 4


def say(*a):
    print(*a, flush=True)


def phase(**kw):
    """A progress report the page shows step by step: which phase, how far, and which files it reads and writes."""
    say("@@PHASE@@" + json.dumps(kw))


def names(stage):
    """The files of each stage: stage 1 is the first training; stage 2, 3… each continue a version of the one before
    on other texts (Wikipedia, then questions and answers, then an author's style, say)."""
    s = "" if stage == 1 else str(stage)
    return {"train": f"train{s}.bin", "val": f"val{s}.bin", "data": f"data{s}.json", "progress": f"progress{s}.json",
            "model": "model" if stage == 1 else f"stage{stage}/model"}


def stage_of(version):
    """'model' -> 1, 'stage3/step-000200' -> 3"""
    return 1 if version == "model" else int(version.split("/")[0][5:])


QUESTION, ANSWER = "Domanda:", "Risposta:"  # how a question and its answer are laid out, in training and when asking
SEARCH, RESULT = "Cerca:", "Risultato:"  # a model that looks things up: what it searches for, and what the search found


# ---------- 1. the texts ----------

def from_folder(folder):
    """Each file in the folder (and its subfolders) as one document: a list of paragraphs. Repeats are dropped."""
    sys.path.insert(0, str(HERE))
    import reader
    reader.MAX_CHARS = 10**12  # the Reader stops long books early; here we want every word
    files = folder_files(folder)
    seen = set()
    for i, f in enumerate(files, 1):
        try:
            paras = reader.extract(f)[1]
        except Exception as e:
            say(f"skipped {f.name}: {e}")
            continue
        keep = []
        for p in paras:
            h = hashlib.sha1(p.strip().lower().encode()).digest()[:8]
            if len(p) > 1 and h not in seen:  # the same paragraph twice teaches nothing new
                seen.add(h)
                keep.append(p)
        say(f"read {i}/{len(files)} · {f.name}")
        phase(phase="read", i=i, n=len(files), file=f.name, folder=str(folder))
        yield keep


def need(package, module=None):
    try:
        __import__(module or package)
    except ImportError:
        say(f"installing {package}…")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], check=True)


def from_wikipedia(lang, words, qa=None, files=None, make=None):
    """Wikipedia in one language, article by article, until about `words` words (None: all of it). Downloaded a part
    at a time (each a few hundred MB) from Hugging Face into its cache folder, which keeps them, so preparing again
    doesn't download them again. With `files` (what step 1 downloaded) it reads only those, without the network."""
    need("pyarrow")
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.constants import HF_HUB_CACHE
    repo, date = WIKI
    parts = files or wiki_parts(lang)
    if make is search_answer:  # the searches come from the other end of Wikipedia, so they aren't the questions already learned
        parts = parts[::-1]
    got = 0
    for k, part in enumerate(parts, 1):
        if files:
            path = part
        else:
            info = dict(phase="download", part=k, parts=len(parts), mb=round((getattr(part, "size", 0) or 0) / 1e6),
                        src=f"https://huggingface.co/datasets/{repo}/tree/main/{date}.{lang}", to=str(HF_HUB_CACHE))
            say(f"downloading Wikipedia ({lang}), part {k} of {len(parts)}…")
            phase(**info)
            path = hf_hub_download(repo, part.path, repo_type="dataset")
            phase(**info, done=True)
        for b in pq.ParquetFile(path).iter_batches(columns=["title", "text"], batch_size=2000):
            for title, text in zip(b.column(0).to_pylist(), b.column(1).to_pylist()):
                paras = [p.strip() for p in text.split("\n") if len(p.strip()) > 1]
                if qa:
                    pair = (make or question_answer)(title, paras)
                    if not pair:
                        continue
                    paras = pair
                got += sum(len(p.split()) for p in paras) if not qa else 1
                yield paras
                if (qa and got >= qa) or (not qa and words and got >= words):
                    return
        say(f"{got:,} {'questions' if qa else 'words'} so far")


def wiki_parts(lang):
    from huggingface_hub import HfApi
    repo, date = WIKI
    parts = sorted((f for f in HfApi().list_repo_tree(repo, path_in_repo=f"{date}.{lang}", repo_type="dataset") if f.path.endswith(".parquet")), key=lambda f: f.path)
    if not parts:
        raise SystemExit(f"No Wikipedia found for the language '{lang}'.")
    return parts


def folder_files(folder):
    files = sorted(p for p in pathlib.Path(folder).expanduser().rglob("*") if p.suffix.lower() in TYPES and p.is_file() and not p.name.startswith("."))
    if not files:
        raise SystemExit(f"No texts found in {folder} (it reads {', '.join(sorted(TYPES))}).")
    return files


def download(d, source):
    """Step 1: gets the texts, and lists what it got in download.json, so tokenizing (step 2) works only on files
    already on this computer. From Wikipedia: the parts it needs, until about the words asked for, each counted as it
    arrives. From a folder: nothing to download, only the list of files it will read."""
    if source.startswith("wikipedia:"):
        need("pyarrow")
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
        from huggingface_hub.constants import HF_HUB_CACHE
        _, lang, millions = source.split(":")
        target = None if millions == "all" else int(float(millions) * 1e6)
        repo, date = WIKI
        src = f"https://huggingface.co/datasets/{repo}/tree/main/{date}.{lang}"
        parts = wiki_parts(lang)
        files, words, mb = [], 0, 0
        for k, part in enumerate(parts, 1):
            size = round((getattr(part, "size", 0) or 0) / 1e6)
            info = dict(phase="download", part=k, parts=len(parts), mb=size, total_mb=mb, words=words, target=target, src=src, to=str(HF_HUB_CACHE))
            say(f"downloading Wikipedia ({lang}), part {k} of {len(parts)}…")
            phase(**info)
            path = hf_hub_download(repo, part.path, repo_type="dataset")
            for b in pq.ParquetFile(path).iter_batches(columns=["text"], batch_size=5000):
                words += sum(len(x.split()) for x in b.column(0).to_pylist())
            files.append(path)
            mb += size
            phase(**{**info, "words": words, "total_mb": mb})
            if target and words >= target:
                break
        got = {"kind": "wiki", "lang": lang, "files": files, "parts": len(files), "of": len(parts), "mb": mb, "words": words, "target": target, "src": src, "to": str(HF_HUB_CACHE)}
    else:
        files = folder_files(source)
        got = {"kind": "folder", "files": [str(f) for f in files], "n": len(files), "mb": round(sum(f.stat().st_size for f in files) / 1e6, 1), "folder": str(pathlib.Path(source).expanduser())}
        phase(phase="found", **{k: v for k, v in got.items() if k != "files"})
    (d / "download.json").write_text(json.dumps({"source": source, **got}))
    phase(phase="downloaded", done=True, **{k: v for k, v in got.items() if k != "files"})


def question_answer(title, paras, words=120):
    """A Wikipedia article as a question and its answer: 'Che cos'è <title>?' and the article's first paragraph,
    which on Wikipedia is a definition. People get 'Chi era' or 'Chi è', guessed from the years in brackets."""
    import random, re
    if not paras or "disambigua" in title.lower() or title.startswith(("Lista", "Elenco", "Anni ", "Nati ", "Morti ")):
        return None
    lead = paras[0]
    if len(lead.split()) < 12:
        return None
    out, n = [], 0
    for sentence in re.split(r"(?<=[.!?])\s+", lead):
        if n + len(sentence.split()) > words and out:
            break
        out.append(sentence)
        n += len(sentence.split())
    name = re.sub(r"\s*\([^)]*\)$", "", title).strip()  # "Mercurio (astronomia)" -> "Mercurio"
    years = re.search(r"\(([^)]*\b\d{3,4}\b[^)]*)\)", lead[:250])
    if years and re.search(r"\bè stat[oa]\b", lead[:400]):
        q = f"Chi era {name}?"
    elif years and re.search(r"\b\d{4}\)", lead[:250]) and re.search(r"\bè (un|una|uno)\b", lead[:400]):
        q = f"Chi è {name}?"
    else:
        # the subject as the article itself names it: "L'automobile è un veicolo…" -> "l'automobile", not "Automobile"
        m = re.match(r"^(.{1,60}?)\s+(?:\(|è\b|sono\b)", lead)
        if m and name.lower() in m[1].lower() and re.match(r"(?i)(l'|un'|il |lo |la |i |gli |le |un |uno |una )", m[1]):
            name = m[1][0].lower() + m[1][1:]
        q = random.choice([f"Che cos'è {name}?", f"Cos'è {name}?", f"Che cosa è {name}?", f"Mi spieghi che cos'è {name}?"])
    return [f"{QUESTION} {q}", f"{ANSWER} {' '.join(out)}"]


def cut(text, words):
    """The first sentences of a text, up to about `words` words (always at least one sentence)."""
    import re
    out, n = [], 0
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if n + len(sentence.split()) > words and out:
            break
        out.append(sentence)
        n += len(sentence.split())
    return " ".join(out)


FOUND_WORDS = 100  # how much of the article the search gives back, in training and when asking: it has to fit in 512 pieces with the question and the answer
STOP = [END, f"\n{QUESTION}", f"\n{SEARCH}", f"\n{RESULT}"]  # where an answer ends: never a new question, search or result


def asked_as(subject, lead, named=None):
    """The many ways a person asks about something, so the model learns to find the subject in any of them,
    not only in "Che cos'è …?". Some only fit certain things: a place, a machine, a person of the past.
    `subject` is how the article names it ("l'automobile"); `named` the bare name ("Automobile"), used after
    "di" and "su", where Italian would merge the article ("dell'automobile")."""
    import random, re
    named = named or subject
    forms = [f"Che cos'è {subject}?", f"Cos'è {subject}?", f"Parlami di {named}.", f"Cosa sai di {named}?", f"Dimmi qualcosa su {named}.",
             f"Mi spieghi {subject}?", f"Che cosa sai dirmi di {named}?", f"Vorrei sapere di più su {named}.", f"{named}?", f"Informazioni su {named}."]
    head = lead[:300]
    person = re.search(r"\(.{0,80}\b\d{3,4}\b.{0,80}\)", head) and re.search(r"\b(è stat[oa]|è (un|una) .{0,40}(politic|scrittor|attor|cantant|musicist|calciator|giornalist|regist|pittor|scienziat|filosof|imprenditor))", head)
    if person:  # nobody asks "what is" a person
        forms = [f for f in forms if not f.startswith(("Che cos'è", "Cos'è", "Mi spieghi"))]
    if re.search(r"\bè stat[oa]\b", head):
        forms += [f"Chi era {subject}?", f"Chi è stato {subject}?", f"Cosa ha fatto {subject}?", f"Quando è vissuto {subject}?", f"Perché è famoso {subject}?"]
    elif re.search(r"\bè (un|una) .{0,40}(politic|scrittor|attor|cantant|musicist|calciator|giornalist|regist|pittor|scienziat|filosof|imprenditor)", head):
        forms += [f"Chi è {subject}?", f"Chi è {subject}? Cosa fa?", f"Di cosa si occupa {subject}?"]
    if re.search(r"\b(comune|città|paese|frazione|regione|provincia|fiume|lago|monte|isola|capitale|quartiere)\b", head) and not re.search(r"\bè stat[oa]\b", head):
        forms += [f"Dove si trova {subject}?", f"Dov'è {subject}?", f"Com'è {subject}?"]
    if re.search(r"\b(motore|macchina|dispositivo|apparecchio|strumento|sistema|meccanismo|processo|tecnica|metodo|software|algoritmo|reazione)\b", head):
        forms += [f"Come funziona {subject}?", f"A cosa serve {subject}?", f"Come si usa {subject}?"]
    if re.search(r"\b(guerra|battaglia|rivoluzione|trattato|evento|crisi|elezioni|terremoto)\b", head):
        forms += [f"Cosa è successo con {subject}?", f"Quando c'è stata {subject}?", f"Raccontami {subject}."]
    q = random.choice(forms)
    if random.random() < 0.3:  # people don't always use capitals and accents
        q = q.lower()
    return q


def search_answer(title, paras):
    """A question answered by looking it up: after the question the model writes what to search for (the subject),
    the search gives back the start of the article, and the answer restates it. Copying from a text a few lines
    above is much easier for a small model than remembering facts, so it learns to answer from what it found."""
    import re
    pair = question_answer(title, paras)  # same filters as the questions and answers: no lists, no disambiguation pages
    if not pair:
        return None
    subject = re.sub(r"\s*\([^)]*\)$", "", title).strip()
    m = re.match(r"^(.{1,60}?)\s+(?:\(|è\b|sono\b|fu\b)", paras[0])  # "L'automobile è…" -> ask about "l'automobile"
    asked = m[1][0].lower() + m[1][1:] if m and subject.lower() in m[1].lower() and re.match(r"(?i)(l'|il |lo |la |i |gli |le )", m[1]) else subject
    named = subject[0].lower() + subject[1:] if asked != subject else subject  # a common noun: "parlami di automobile", not "di Automobile"
    found = cut(" ".join(p for p in paras[:2] if len(p.split()) >= 8), FOUND_WORDS)  # a short second line is a section heading, which the search never gives back
    return [f"{QUESTION} {asked_as(asked, paras[0], named)}", f"{SEARCH} {subject}", f"{RESULT} {title}. {found}", f"{ANSWER} {found}"]


# what a search doesn't need: the words of the question around the subject ("come funziona un motore a scoppio" -> "motore a scoppio")
QUESTION_WORDS = r"(?i:^(chi|che|cosa|cos'|cos'è|è|che cos'è|che cosa|come|dove|dov'è|quando|perché|quale|quali|quanto|quanti|parlami|dimmi|spiegami|mi spieghi|vorrei sapere|informazioni|mi|sai|dirmi|di più|qualcosa|su|di|del|della|dello|dei|degli|delle|sul|sulla|a|per|funziona|funzionano|serve|servono|si usa|si trova|si trovano|c'è stata|c'è stato|è successo|successo|ha fatto|è vissuto|è vissuta|è famoso|è famosa|era|stato|stata)(?=\s|$|')\s*'?)|^(un|una|uno|un'|il|lo|la|l'|i|gli|le|l)(?=\s|$|')\s*'?"  # articles only in lowercase: "L'Aquila" keeps its own


def lookup(lang, query, words=FOUND_WORDS):
    """What the search finds when a model asks: the start of the Wikipedia article that best matches, as in training.
    Wikipedia's own first result, skipping disambiguation pages. Returns (title, text), or (None, None) if nothing
    is found or Wikipedia can't be reached."""
    import re, urllib.parse, urllib.request
    query = re.sub(r"[?.!]+$", "", query).strip()
    for _ in range(8):  # a whole question searched as it is: take the question words off the front, one by one
        query = re.sub(QUESTION_WORDS, "", query, count=1).strip() or query
    url = f"https://{lang}.wikipedia.org/w/api.php?" + urllib.parse.urlencode({
        "action": "query", "format": "json", "generator": "search", "gsrsearch": query, "gsrlimit": 3,
        "prop": "extracts", "exintro": 1, "explaintext": 1, "exlimit": 3, "redirects": 1})
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AskJoshua/1.0 (https://ask.joshua.earth)"})
        pages = list((json.load(urllib.request.urlopen(req, timeout=15)).get("query") or {}).get("pages", {}).values())
    except Exception:
        return None, None
    pages = [p for p in pages if p.get("extract", "").strip() and "disambigua" not in p.get("extract", "")[:300].lower()]
    if not pages:
        return None, None
    best = min(pages, key=lambda p: p.get("index", 9))  # the pages come back unordered; "index" is Wikipedia's rank
    paras = [p.strip() for p in best["extract"].split("\n") if p.strip()]
    return best["title"], cut(" ".join(paras[:2]), words)


def searched(lang, question, generate):
    """Asks with a search, the way the training did: the model writes what to search for, Wikipedia is searched,
    the result goes into the text, and the model writes the answer. `generate(prompt, stop)` runs the model.
    Returns (the text to continue for the answer, what it searched for, the title found or None)."""
    asked = f"{END}\n{QUESTION} {question.strip()}\n{SEARCH}"
    query = generate(asked, ["\n", END]).strip() or question.strip()
    title, found = lookup(lang, query)
    if not title:  # nothing found: it answers from what it knows
        return f"{END}\n{QUESTION} {question.strip()}\n{ANSWER}", query, None
    return f"{asked} {query}\n{RESULT} {title}. {found}\n{ANSWER}", query, title


def stop_at(text):
    """An answer ends where the model starts a new question, search or result."""
    for s in STOP:
        text = text.split(s)[0]
    return text


def documents(source, files=None):
    if source.startswith("wikipedia:"):
        _, lang, millions = source.split(":")
        return from_wikipedia(lang, None if millions == "all" else float(millions) * 1e6, files=files)
    if source.startswith("wikiqa:"):  # questions and answers made from Wikipedia: wikiqa:LANG:HOW_MANY
        _, lang, n = source.split(":")
        return from_wikipedia(lang, None, qa=int(n))
    if source.startswith("wikisearch:"):  # questions answered by searching Wikipedia: wikisearch:LANG:HOW_MANY
        _, lang, n = source.split(":")
        return from_wikipedia(lang, None, qa=int(n), make=search_answer)
    return from_folder(source)


# ---------- 2. the vocabulary, and the texts as numbers ----------

def prepare(d, source, stage):
    """Three passes, so even a billion words never sit in memory at once:
    the texts go to a plain file (one paragraph per line, an empty line between documents);
    in stage 1 the vocabulary is learned from it; then it's turned into numbers, written to disk as it goes."""
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    f = names(stage)
    corpus = d / f"corpus{stage}.txt"
    words, docs = 0, 0
    wiki = source.startswith(("wikipedia:", "wikiqa:"))
    target = None
    if source.startswith("wikipedia:") and source.split(":")[2] != "all":
        target = int(float(source.split(":")[2]) * 1e6)
    got = json.loads((d / "download.json").read_text()) if stage == 1 and (d / "download.json").exists() else {}
    files = got.get("files") if got.get("source") == source and got.get("kind") == "wiki" else None  # what step 1 downloaded
    with open(corpus, "w", encoding="utf-8") as out:
        for paras in documents(source, files):
            if paras:
                out.write("".join(" ".join(p.split()) + "\n" for p in paras) + "\n")
                words += sum(len(p.split()) for p in paras)
                docs += 1
                if docs % 20000 == 0:
                    say(f"{docs:,} documents · {words:,} words")
                    phase(phase="text", docs=docs, words=words, target=target, file=str(corpus))
    phase(phase="text", docs=docs, words=words, target=target, file=str(corpus), done=True)
    if words < (50_000 if stage == 1 else 5_000):  # a later stage only adjusts a model that already knows the language
        corpus.unlink(missing_ok=True)
        raise SystemExit(f"Only {words:,} words: a model needs at least a few hundred thousand to learn anything.")

    def lines(limit=None):
        n = 0
        with open(corpus, encoding="utf-8") as src:
            for line in src:
                line = line.rstrip("\n")
                if line:
                    yield line
                    n += len(line.split())
                    if limit and n >= limit:
                        return

    if stage == 1:
        say("learning the vocabulary…")
        phase(phase="vocab", file=str(d / "tokenizer.json"), words=min(words, 40_000_000))
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)  # starts from bytes: any character, any language
        tok.decoder = decoders.ByteLevel()
        tok.train_from_iterator(lines(40_000_000), trainers.BpeTrainer(  # 40 million words are plenty to choose 8,192 pieces
            vocab_size=VOCAB, special_tokens=[END], initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
        tok.save(str(d / "tokenizer.json"))
        phase(phase="vocab", file=str(d / "tokenizer.json"), words=min(words, 40_000_000), done=True)
    else:
        tok = Tokenizer.from_file(str(d / "tokenizer.json"))  # the same vocabulary: the model only knows these pieces
    end = np.array([tok.token_to_id(END)], np.uint16)

    say("turning the texts into numbers…")
    n_train = n_val = k = 0
    seen_words = 0  # words turned into numbers so far, to estimate the total number of pieces
    with open(corpus, encoding="utf-8") as src, open(d / f["train"], "wb") as tr, open(d / f["val"], "wb") as va:
        batch = []

        def flush():
            nonlocal n_train, n_val, k
            nonlocal seen_words
            seen_words += sum(len(p.split()) for p in batch)
            for enc in tok.encode_batch(["\n" + p for p in batch]):
                ids = np.array(enc.ids, np.uint16)
                k += 1
                if k % 50 == 0:  # one paragraph in 50 is kept aside, to check the model on text it never studies
                    ids.tofile(va)
                    n_val += len(ids)
                else:
                    ids.tofile(tr)
                    n_train += len(ids)
            batch.clear()

        # an example made of several lines (a question and its answer, a search) stays whole: set aside or studied
        # together, otherwise a question could end up next to someone else's answer
        whole = str(source).startswith(("wikiqa:", "wikisearch:"))
        doc = []
        for line in src:
            line = line.rstrip("\n")
            if not line:  # the end of a document
                if doc:
                    batch.append("\n".join(doc))
                    doc.clear()
                flush()
                end.tofile(tr)
                n_train += 1
                continue
            if whole:
                doc.append(line)
                continue
            batch.append(line)
            if len(batch) >= 5000:
                before = n_train
                flush()
                if n_train // 5_000_000 != before // 5_000_000:
                    say(f"{n_train:,} tokens")
                    phase(phase="numbers", tokens=n_train + n_val, estimate=round((n_train + n_val) / max(seen_words, 1) * words), file=str(d / f["train"]))
        flush()
    phase(phase="numbers", tokens=n_train + n_val, estimate=n_train + n_val, file=str(d / f["train"]), done=True)
    corpus.unlink()
    label = source if wiki else pathlib.Path(source).name
    where = {"data": str(d / f["train"]), "data_mb": round((d / f["train"]).stat().st_size / 1e6), "deleted": str(corpus)}
    if wiki:
        from huggingface_hub.constants import HF_HUB_CACHE
        blobs = pathlib.Path(HF_HUB_CACHE) / ("datasets--" + WIKI[0].replace("/", "--")) / "blobs"
        # newer versions of huggingface_hub keep the files in a shared store and only link to them from here
        size = sum(p.resolve().stat().st_size for p in blobs.iterdir() if p.resolve().is_file()) if blobs.exists() else 0
        where.update(cache=str(HF_HUB_CACHE), cache_gb=round(size / 1e9, 1))
    else:
        where["folder"] = str(pathlib.Path(source).expanduser())
    phase(phase="done", **where)
    info = {"source": label, "docs": docs, "words": words, "train_tokens": n_train, "val_tokens": n_val, "vocab": tok.get_vocab_size(), "where": where}
    (d / f["data"]).write_text(json.dumps(info))
    say(f"ready: {words:,} words → {n_train:,} tokens to learn from, {n_val:,} kept aside to check")


# ---------- 3. the model, and 4. training ----------

def config(size, vocab):
    hidden, layers, heads, inter = SIZES[size]
    return {"architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": hidden, "num_hidden_layers": layers,
            "num_attention_heads": heads, "num_key_value_heads": heads, "intermediate_size": inter, "vocab_size": vocab,
            "max_position_embeddings": CTX, "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "tie_word_embeddings": True,
            "hidden_act": "silu", "torch_dtype": "float32", "bos_token_id": 0, "eos_token_id": 0}


def write_model(d, out, model, cfg):
    """The model in Hugging Face's layout, which mlx-lm, llama.cpp's converter and so Ollama all read."""
    from mlx.utils import tree_flatten
    import mlx.core as mx
    out.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out / "model.safetensors"), dict(tree_flatten(model.parameters())), metadata={"format": "pt"})
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    shutil.copy(d / "tokenizer.json", out / "tokenizer.json")
    (out / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "PreTrainedTokenizerFast", "bos_token": END, "eos_token": END,
                                                           "model_max_length": CTX, "clean_up_tokenization_spaces": False}))


def sample(model, tok, prompt, n=60, temp=0.8):
    import mlx.core as mx
    ids = tok.encode(prompt).ids
    for _ in range(n):
        logits = model(mx.array([ids[-CTX:]]))[0, -1]
        nxt = int(mx.random.categorical(logits / temp))
        if nxt == tok.token_to_id(END):
            break
        ids.append(nxt)
    return tok.decode(ids)


def train(d, size, steps, stage=1, mix=0.0, base="model"):
    import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
    from functools import partial
    from mlx.utils import tree_flatten
    from mlx_lm.models import llama
    from tokenizers import Tokenizer
    f = names(stage)
    tok = Tokenizer.from_file(str(d / "tokenizer.json"))
    cfg = config(size, tok.get_vocab_size())
    model = llama.Model(llama.ModelArgs(**{k: v for k, v in cfg.items() if k in llama.ModelArgs.__dataclass_fields__}))
    keys = ("train", "val", "old", "samples")
    done, meta = 0, {k: [] for k in keys}
    prog, out = d / f["progress"], d / f["model"]
    if prog.exists() and (out / "model.safetensors").exists():
        p = json.loads(prog.read_text())
        if p.get("size") == size and p["step"] < steps and p.get("mix", mix) == mix and p.get("base", base) == base:
            model.load_weights(str(out / "model.safetensors"))
            done, meta = p["step"], {k: p.get(k, []) for k in keys}
            say(f"picking up at step {done} of {steps}")
    if stage > 1 and not done:
        if not (d / base / "model.safetensors").exists():
            raise SystemExit("The version to start from isn't there: train the earlier stage first.")
        model.load_weights(str(d / base / "model.safetensors"))  # where the chosen version stood
        for later in range(stage, 10):  # a new stage replaces the old one, and whatever was built on it
            shutil.rmtree(d / f"stage{later}", ignore_errors=True)
            (d / names(later)["progress"]).unlink(missing_ok=True)
        say(f"starting from {base}")
    mx.eval(model.parameters())
    n_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    say(f"model: {n_params / 1e6:.1f} million parameters · {steps} steps of {BATCH * CTX:,} tokens")

    data = np.memmap(d / f["train"], dtype=np.uint16, mode="r")
    vdata = np.memmap(d / f["val"], dtype=np.uint16, mode="r")
    prev = names(stage_of(base))  # the texts of the stage the starting version came from: mixed in, and watched for forgetting
    first = np.memmap(d / prev["train"], dtype=np.uint16, mode="r") if stage > 1 else None
    vfirst = np.memmap(d / prev["val"], dtype=np.uint16, mode="r") if stage > 1 else None
    if len(data) <= CTX + 1:
        raise SystemExit("Not enough text to fill even one window.")
    rng = np.random.default_rng(done + 1000 * stage)
    old_n = round(BATCH * mix) if stage > 1 else 0  # how many windows of each batch still come from the first texts

    def windows(src, n):
        start = rng.integers(0, len(src) - CTX - 1, n)
        return np.stack([src[s:s + CTX + 1] for s in start]).astype(np.int32)

    def batch(src, n=BATCH, old=0):
        w = windows(src, n - old) if not old else np.concatenate([windows(src, n - old), windows(first, old)])
        return mx.array(w[:, :-1]), mx.array(w[:, 1:])  # the target is the same text shifted by one token

    # the second stage starts from a trained model: a gentler learning rate, so it bends rather than breaks
    lr = LR[size] * (0.3 if stage > 1 else 1)
    warm = min(500 if stage == 1 else 100, steps // 10)
    sched = optim.join_schedules([optim.linear_schedule(lr / 100, lr, warm), optim.cosine_decay(lr, max(1, steps - warm), lr / 10)], [warm])
    opt = optim.AdamW(learning_rate=sched, weight_decay=0.1)
    opt.state["step"] = mx.array(done, mx.uint64)  # so the learning rate continues from the right point

    def loss_fn(m, x, y):
        return nn.losses.cross_entropy(m(x), y, reduction="mean")
    grad = nn.value_and_grad(model, loss_fn)
    state = [model.state, opt.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(x, y):
        loss, g = grad(model, x, y)
        g, _ = optim.clip_grad_norm(g, 1.0)
        opt.update(model, g)
        return loss

    def check(src):
        return sum(loss_fn(model, *batch(src, 16)).item() for _ in range(10)) / 10 if len(src) > CTX + 1 else float("nan")

    # how often to check, save and write a sample: in stage 2, at 8 evenly spaced points, each kept as its own copy
    every = max(10, steps // SNAPSHOTS) if stage > 1 else max(50, min(1000, steps // 20))
    t0, start_step, run = time.time(), done, 0.0
    for i in range(done + 1, steps + 1):
        loss = step(*batch(data, old=old_n))
        mx.eval(state, loss)
        run += loss.item()
        if i % 10 == 0:
            left = (time.time() - t0) / (i - start_step) * (steps - i)
            say(f"Iter {i}: Train loss {run / 10:.3f}, about {int(left // 3600)} h {int(left % 3600 // 60)} min left")
            meta["train"].append((i, round(run / 10, 4)))
            run = 0.0
        if i % every == 0 or i == steps:
            v = check(vdata)
            say(f"Iter {i}: Val loss {v:.3f}")
            meta["val"].append((i, round(v, 4)))
            if stage > 1:  # how well it still knows the earlier texts: rising means it's forgetting them
                o = check(vfirst)
                meta["old"].append((i, round(o, 4)))
                say("@@OLD@@" + json.dumps([i, round(o, 4)]))
            text = sample(model, tok, "\n")
            meta["samples"] = (meta["samples"] + [(i, text)])[-30:]
            say("@@SAMPLE@@" + json.dumps([i, text]))
            write_model(d, out, model, cfg)
            if stage > 1:
                shutil.copytree(out, d / f"stage{stage}" / f"step-{i:06d}", dirs_exist_ok=True)
            prog.write_text(json.dumps({"step": i, "size": size, "steps": steps, "mix": mix, "base": base, **meta}))
    say("finished")


def try_it(d, prompt, which="model", question=False, lang=None):
    """Continues a text; or, for a question, frames it the way the questions-and-answers training did:
    a fresh document (END), the question, then 'Risposta:' for the model to carry on. It stops at END."""
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    model, tok = load(str(d / which))
    framed = f"{END}\n{QUESTION} {prompt.strip()}\n{ANSWER}" if question else prompt
    head = ""
    if question and lang:  # a version that learned to search: it looks the question up first
        def run(text, stop):
            out = generate(model, tok, prompt=text, max_tokens=24, sampler=make_sampler(temp=0.3))
            for s in stop:
                out = out.split(s)[0]
            return out
        framed, query, title = searched(lang, prompt, run)
        head = f"🔎 {query} → " + (f"{title}\n\n" if title else "nothing found\n\n")
    text = generate(model, tok, prompt=framed, max_tokens=240, sampler=make_sampler(temp=0.7))
    say("@@TRY@@" + json.dumps(head + stop_at(text).strip() if question else prompt + text))


if __name__ == "__main__":
    cmd, d = sys.argv[1], pathlib.Path(sys.argv[2])
    if cmd == "download":
        download(d, sys.argv[3])
    elif cmd == "prepare":
        prepare(d, sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 1)
    elif cmd == "train":
        train(d, sys.argv[3], int(sys.argv[4]), int(sys.argv[5]) if len(sys.argv) > 5 else 1, float(sys.argv[6]) if len(sys.argv) > 6 else 0.0,
              sys.argv[7] if len(sys.argv) > 7 else "model")
    elif cmd == "try":
        try_it(d, sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "model", len(sys.argv) > 5 and sys.argv[5] == "question",
               sys.argv[6] if len(sys.argv) > 6 else None)
