# Step 1: collect MIDI data
# Saves classical pieces from the music21 corpus as .mid files in data/midi.
# You can also copy your own MIDI files (jazz, classical, ...) into that folder.
import argparse
import os

from music21 import corpus, converter

parser = argparse.ArgumentParser()
parser.add_argument("--composers", nargs="+", default=["bach"], help="composers of the music21 corpus, e.g. bach mozart beethoven")
parser.add_argument("--max-files", type=int, default=500, help="maximum number of files per composer")
parser.add_argument("--out", default="data/midi")
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)
saved = 0
for composer in args.composers:
    paths = list(corpus.getComposer(composer))[: args.max_files]
    print(f"{composer}: {len(paths)} pieces found")
    for i, path in enumerate(paths):
        target = os.path.join(args.out, f"{composer}_{i:03d}.mid")
        if os.path.exists(target):
            continue
        try:
            converter.parse(path).write("midi", fp=target)
            saved += 1
        except Exception as error:
            print("  skipped", os.path.basename(str(path)), "-", error)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(paths)}")

total = len([f for f in os.listdir(args.out) if f.lower().endswith((".mid", ".midi"))])
print(f"Done. {saved} new files saved, {total} MIDI files in {args.out}")
