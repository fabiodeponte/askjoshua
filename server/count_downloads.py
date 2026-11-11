#!/usr/bin/env python3
"""Counts Joshua's installs and updates from the web server's logs. Runs every hour (cron).

  install = an installation finished on a computer for the first time: install.sh then fetches /installed, once
            (it leaves a marker file, so running the installer again on that computer isn't counted)
  started = install.sh downloaded by the install command: includes repeats and installations given up (private)
  update  = Joshua's update button fetching joshua.tar.gz (python-requests)

Only totals per day are kept: no addresses, nothing about who downloaded.
The logs only go back 14 days, so each day's totals are saved in days.json and kept after the logs are gone.
Public:  /var/www/html/askjoshua/stats.json  -> {"installs": N, "since": "2026-11-06"}
Private: ~/askjoshua-stats/report.txt         -> installs and updates, day by day
"""
import gzip, glob, json, pathlib, re
from collections import defaultdict
from datetime import datetime

HERE = pathlib.Path.home() / "askjoshua-stats"
PUBLIC = pathlib.Path("/var/www/html/askjoshua/stats.json")
LINE = re.compile(r'\[(\d+/\w+/\d+):[^\]]*\] "GET (?:/askjoshua)?/(install\.sh|joshua\.tar\.gz|installed\?done)[ ?][^"]*" (\d{3}) \S+ "[^"]*" "([^"]*)"')  # only the installer asks for installed?done

found = defaultdict(lambda: {"installs": 0, "started": 0, "updates": 0})
for path in glob.glob("/var/log/apache2/other_vhosts_access.log*") + glob.glob("/var/log/apache2/access.log*"):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", errors="replace") as f:
        for line in f:
            m = LINE.search(line)
            if not m or m[3] != "200":
                continue
            day = datetime.strptime(m[1], "%d/%b/%Y").date().isoformat()
            agent = m[4].lower()
            if m[2] == "installed?done":
                found[day]["installs"] += 1
            elif m[2] == "install.sh" and agent.startswith("curl/"):
                found[day]["started"] += 1
            elif m[2] == "joshua.tar.gz" and agent.startswith("python-requests/"):
                found[day]["updates"] += 1

HERE.mkdir(exist_ok=True)
store = HERE / "days.json"
days = json.loads(store.read_text()) if store.exists() else {}
days.update(found)  # days still in the logs are recounted; older days keep their saved totals
for d in days.values():
    d.setdefault("started", 0)
store.write_text(json.dumps(dict(sorted(days.items())), indent=1))

installs = sum(d["installs"] for d in days.values())
updates = sum(d["updates"] for d in days.values())
since = min((k for k, d in days.items() if d["installs"]), default=None)
PUBLIC.write_text(json.dumps({"installs": installs, "since": since}))
started = sum(d["started"] for d in days.values())
report = [f"Joshua: {installs} installs completed, {started} install commands run, {updates} updates (installs counted since {since})", ""] + [f"{k}  installs {d['installs']:4}  started {d['started']:4}  updates {d['updates']:4}" for k, d in sorted(days.items()) if d["installs"] or d["started"] or d["updates"]]
(HERE / "report.txt").write_text("\n".join(report) + "\n")
print(report[0])
