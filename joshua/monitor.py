"""Monitors: scheduled tasks that report only what is new since their last run.

Each monitor remembers the links it has already reported (monitors/<name>.json), reads the sites and
RSS feeds you list, and asks Joshua to search the web too. Joshua's code, not the model, decides what
counts as new: an item is new only if its link was never reported before.
"""
import datetime as dt
import email.utils
import json
import pathlib
import re
import urllib.parse
import xml.etree.ElementTree as ET

import requests

DIR = pathlib.Path(__file__).parent / "monitors"
NOTHING = "NOTHING_NEW"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"}
LINK = re.compile(r"https?://[^\s)\]>\"'<]+")


def _path(name):
    return DIR / ((re.sub(r"[^\w-]+", "_", name).strip("_") or "monitor") + ".json")


def state(name):
    try:
        return json.loads(_path(name).read_text(encoding="utf-8"))
    except Exception:
        return {"seen": [], "since": None, "last": None}


def save(name, st):
    DIR.mkdir(exist_ok=True)
    st["seen"] = st["seen"][-3000:]  # the newest links are enough to recognise repeats
    tmp = _path(name).with_suffix(".tmp")
    tmp.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(_path(name))


def forget(name):
    _path(name).unlink(missing_ok=True)


def clean(url):
    """The same article with different tracking tags is the same link."""
    p = urllib.parse.urlsplit(url.strip().rstrip(".,;"))
    q = [(k, v) for k, v in urllib.parse.parse_qsl(p.query) if not k.lower().startswith(("utm_", "fbclid", "gclid", "ref"))]
    return urllib.parse.urlunsplit((p.scheme, p.netloc.lower().removeprefix("www."), p.path.rstrip("/"), urllib.parse.urlencode(q), ""))


def since(task, st):
    """From when to look: the last run that reported something, or one interval back for the first run."""
    if st.get("since"):
        return dt.datetime.fromisoformat(st["since"])
    back = {"hours": dt.timedelta(hours=int(task.get("every", 1))), "weekly": dt.timedelta(days=7)}.get(task["when"], dt.timedelta(days=1))
    return dt.datetime.now() - back


def _date(text):
    if not text:
        return None
    try:
        d = email.utils.parsedate_to_datetime(text)
    except Exception:
        try:
            d = dt.datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
        except Exception:
            return None
    return d.astimezone().replace(tzinfo=None) if d.tzinfo else d


def _feed(xml, base):
    """Items of an RSS or Atom feed: title, link, date."""
    root = ET.fromstring(xml)
    out = []
    for it in root.iter():
        tag = it.tag.split("}")[-1]
        if tag not in ("item", "entry"):
            continue
        get = lambda name: next((c for c in it if c.tag.split("}")[-1] == name), None)
        title, link = get("title"), get("link")
        href = (link.get("href") or link.text or "") if link is not None else ""
        when = next((c.text for c in it if c.tag.split("}")[-1] in ("pubDate", "published", "updated", "date")), None)
        if href:
            out.append({"title": " ".join((title.text or "").split()) if title is not None else "", "link": urllib.parse.urljoin(base, href.strip()), "date": _date(when)})
    return out


def _headlines(page, base):
    """No feed: the page's links that look like headlines (long enough text, same site)."""
    host = urllib.parse.urlsplit(base).netloc.removeprefix("www.")
    out = []
    for href, text in re.findall(r"<a\b[^>]*href=[\"']([^\"'#]+)[\"'][^>]*>(.*?)</a>", page, re.S | re.I):
        text = " ".join(re.sub(r"<[^>]+>", " ", text).split())
        url = urllib.parse.urljoin(base, href)
        if len(text) >= 35 and urllib.parse.urlsplit(url).netloc.removeprefix("www.").endswith(host):
            out.append({"title": text[:200], "link": url, "date": None})
    return out[:40]


def read_source(url):
    """A feed is read as a feed; a web page is searched for its feed, otherwise for its headlines."""
    r = requests.get(url, headers=UA, timeout=20)
    r.raise_for_status()
    body = r.text
    if re.search(r"<(rss|feed|rdf:RDF)\b", body[:2000]):
        return _feed(r.content, url)
    alt = re.search(r"<link\b[^>]*type=[\"']application/(?:rss|atom)\+xml[\"'][^>]*>", body, re.I)
    href = alt and re.search(r"href=[\"']([^\"']+)", alt.group(0))
    if href:
        try:
            f = requests.get(urllib.parse.urljoin(url, href.group(1)), headers=UA, timeout=20)
            f.raise_for_status()
            return _feed(f.content, url)
        except Exception:
            pass
    return _headlines(body, url)


def gather(sources, seen, after):
    """New items from your sources: not reported before, and not older than 'after' (items without a date are kept)."""
    items, failed = [], []
    for src in sources:
        try:
            for it in read_source(src):
                if clean(it["link"]) in seen or (it["date"] and it["date"] < after):
                    continue
                items.append(dict(it, source=urllib.parse.urlsplit(src).netloc.removeprefix("www.")))
        except Exception:
            failed.append(src)
    unique = {clean(i["link"]): i for i in items}
    return list(unique.values())[:60], failed


def prompt(task, items, seen, after):
    """What Joshua is asked at each run. The definition of done is spelled out, and checked again by code afterwards."""
    lines = [f"MONITOR: {task['prompt'].strip()}",
             "",
             f"Find what is NEW about this since {after:%A %d %B %Y, %H:%M}. Today is {dt.datetime.now():%A %d %B %Y, %H:%M}.",
             "Search the web for it, and use the items from my sources below."]
    if items:
        lines += ["", "Items from my sources (headline · date · link):"]
        lines += [f"- {i['title']} · {i['date']:%d/%m %H:%M} · {i['link']}" if i["date"] else f"- {i['title']} · {i['link']}" for i in items]
    if seen:
        lines += ["", "Already reported in earlier runs, do NOT report these again:"] + [f"- {u}" for u in seen[-40:]]
    lines += ["",
              "The report is done when:",
              "- every item is new (published after the date above) and is not in the already-reported list;",
              "- every item has one or two sentences on what happened, its date, and a link to its source as [publisher](URL);",
              "- the most important items come first; several articles about the same event are one item, with all their links.",
              f"If there is nothing new, answer only {NOTHING}. Write in the language of the topic above."]
    return "\n".join(lines)


def new_links(answer, seen):
    """The links in Joshua's report that were never reported before."""
    found = [clean(u) for u in LINK.findall(answer or "")]
    return list(dict.fromkeys(u for u in found if u not in set(seen)))
