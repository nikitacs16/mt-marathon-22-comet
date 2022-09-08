#!/usr/bin/env python3

import csv
import glob
from pathlib import Path

for f in glob.glob("downloaded/wmt*.csv"):
    langdir = Path(f).stem.split("-")[1]
    lang1 = langdir[0:2]
    lang2 = langdir[2:4]
    with open(f, "r") as f:
        data = list(csv.DictReader(f))
    with open(f"computed/{langdir}.src", "w") as f:
        f.write("\n".join([x["src"] for x in data]))
    with open(f"computed/{langdir}.hyp", "w") as f:
        f.write("\n".join([x["mt"] for x in data]))
    with open(f"computed/{langdir}.ref", "w") as f:
        f.write("\n".join([x["ref"] for x in data]))
    with open(f"computed/{langdir}.score", "w") as f:
        f.write("\n".join([x["score"] for x in data]))
