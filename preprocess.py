# Step 2: preprocess the MIDI files into token sequences for training
# music21 reads every song and turns it into tokens: one token per note (N60) and one for the length (D0.5).
# The songs are cut into pieces of 128 tokens. For every position the model has to guess the next token.
import argparse
import glob
import json
import os
import random

import numpy as np

from music_utils import midi_to_tokens

parser = argparse.ArgumentParser()
parser.add_argument("--midi-dir", default="data/midi")
parser.add_argument("--seq-len", type=int, default=128, help="length of one training piece (in tokens)")
parser.add_argument("--stride", type=int, default=32, help="how far the next piece starts after the previous one")
parser.add_argument("--val-share", type=float, default=0.1, help="share of the songs kept apart to test the model")
parser.add_argument("--max-files", type=int, default=0, help="0 = use all files")
parser.add_argument("--out", default="processed")
args = parser.parse_args()

files = sorted(glob.glob(os.path.join(args.midi_dir, "**", "*.mid*"), recursive=True))
if not files:
    raise SystemExit(f"No MIDI files in {args.midi_dir}. Run collect_data.py first.")
random.seed(0)
if args.max_files:
    files = random.sample(files, min(args.max_files, len(files)))

songs = []
for i, path in enumerate(files, 1):
    tokens = midi_to_tokens(path)
    if len(tokens) > args.seq_len + 1:
        songs.append(tokens)
    if i % 10 == 0 or i == len(files):
        print(f"read {i}/{len(files)} files")

vocab = sorted({t for song in songs for t in song})
index = {t: i for i, t in enumerate(vocab)}
songs = [[index[t] for t in song] for song in songs]

# whole songs are kept apart for validation, so the model is tested on music it has never seen
random.shuffle(songs)
n_val = max(1, int(len(songs) * args.val_share))
val_songs, train_songs = songs[:n_val], songs[n_val:]


def make_pieces(song_list):
    X, Y = [], []
    for song in song_list:
        for i in range(0, len(song) - args.seq_len, args.stride):
            X.append(song[i:i + args.seq_len])
            Y.append(song[i + 1:i + args.seq_len + 1])  # same piece moved by one token = the "next token" for every position
    return np.array(X, dtype=np.int64), np.array(Y, dtype=np.int64)


X_train, Y_train = make_pieces(train_songs)
X_val, Y_val = make_pieces(val_songs)
seeds = np.array([song[:32] for song in train_songs], dtype=np.int64)  # beginnings of real songs, used by generate.py

os.makedirs(args.out, exist_ok=True)
np.savez_compressed(os.path.join(args.out, "dataset.npz"), X_train=X_train, Y_train=Y_train, X_val=X_val, Y_val=Y_val, seeds=seeds)
with open(os.path.join(args.out, "vocab.json"), "w") as f:
    json.dump(vocab, f)

print(f"songs: {len(train_songs)} for training, {len(val_songs)} for validation")
print(f"vocabulary: {len(vocab)} different tokens")
print(f"training pieces: {len(X_train)} | validation pieces: {len(X_val)} (each {args.seq_len} tokens long)")
