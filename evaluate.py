# Step 5: evaluate the generated music with numbers (not only "it sounds nice").
# We compare generated pieces with real pieces the model has never seen (validation set).
import argparse
import json
import os

import numpy as np

from music_utils import tokens_to_events

SCALE = {0, 2, 4, 5, 7, 9, 11}  # all songs are in C major / A minor, so these notes belong to the key
WINDOW = 128  # same number of tokens for real and generated pieces, so the comparison is fair


def analyse(tokens):
    """Music statistics for one piece."""
    tokens = tokens[:WINDOW]
    events = tokens_to_events(tokens)
    sounding = [p for p, _ in events if p]
    notes = [n for p, _ in events for n in p]
    if not notes:
        return None
    total_beats = sum(l for _, l in events)
    rest_beats = sum(l for p, l in events if not p)
    tops = [max(p) for p in sounding]
    grams = [tuple(tokens[i:i + 4]) for i in range(len(tokens) - 3)]
    hist = np.bincount([n % 12 for n in notes], minlength=12) / len(notes)
    return {
        "scale_ratio": float(np.mean([n % 12 in SCALE for n in notes])),
        "distinct_4gram": len(set(grams)) / max(len(grams), 1),
        "rest_ratio": rest_beats / max(total_beats, 1e-9),
        "avg_polyphony": len(notes) / max(len(sounding), 1),
        "pitch_range": max(notes) - min(notes),
        "avg_leap": float(np.mean(np.abs(np.diff(tops)))) if len(tops) > 1 else 0.0,
        "avg_note_length": total_beats / max(len(events), 1),
        "_hist": hist,
    }


def summarise(pieces):
    pieces = [p for p in pieces if p]
    if not pieces:
        raise ValueError("No notes to analyse.")
    out = {k: float(np.mean([p[k] for p in pieces])) for k in pieces[0] if k != "_hist"}
    out["_hist"] = np.mean([p["_hist"] for p in pieces], axis=0)
    return out


def evaluate_setting(model, vocab, seeds, samples, length, temperature, top_k, top_p, no_repeat=0):
    """Generate `samples` pieces with these settings and return their average statistics."""
    from generator import generate_tokens, pick_seed
    return summarise([analyse(generate_tokens(model, vocab, pick_seed(seeds, vocab), length,
                                              temperature, top_k, top_p, no_repeat))
                      for _ in range(samples)])


def pitch_similarity(real, generated):
    return float(np.minimum(real["_hist"], generated["_hist"]).sum())  # 1.0 = identical note usage


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model/music_lstm.pt")
    parser.add_argument("--data", default="processed")
    parser.add_argument("--samples", type=int, default=30, help="how many pieces to generate for the test")
    parser.add_argument("--length", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--no-repeat", type=int, default=0, help="forbid repeating the same run of N tokens (0 = off)")
    parser.add_argument("--sweep", action="store_true", help="compare several temperatures in one table")
    parser.add_argument("--out", default="evaluation")
    args = parser.parse_args()

    from generator import load_model, load_seeds, set_seed

    set_seed(0)
    model, vocab = load_model(args.model)
    seeds = load_seeds(args.data)
    dataset = np.load(os.path.join(args.data, "dataset.npz"))
    if "X_val" not in dataset.files:
        raise SystemExit(f"{args.data}/dataset.npz was made by an older version of the project. Run: python preprocess.py")
    X_val = dataset["X_val"]

    real = summarise([analyse([vocab[i] for i in row]) for row in X_val])
    if args.sweep:
        rows = []
        for temperature in (0.6, 0.8, 1.0, 1.2):
            print(f"temperature {temperature}: generating {args.samples} pieces ...")
            gen = evaluate_setting(model, vocab, seeds, args.samples, args.length, temperature, args.top_k, args.top_p,
                                   args.no_repeat)
            rows.append({"temperature": temperature, "notes_in_key": gen["scale_ratio"], "variety": gen["distinct_4gram"],
                         "avg_melodic_jump": gen["avg_leap"], "pitch_class_similarity": pitch_similarity(real, gen)})
        lines = ["| Temperature | Notes in key | Variety | Melodic jump | Pitch-class similarity |", "|---|---|---|---|---|"]
        lines += [f"| {r['temperature']} | {r['notes_in_key']:.2f} | {r['variety']:.2f} | {r['avg_melodic_jump']:.2f} | "
                  f"{r['pitch_class_similarity']:.2f} |" for r in rows]
        lines.append(f"| *real music* | {real['scale_ratio']:.2f} | {real['distinct_4gram']:.2f} | {real['avg_leap']:.2f} | 1.00 |")
        print("\n" + "\n".join(lines))
        os.makedirs(args.out, exist_ok=True)
        open(os.path.join(args.out, "sweep.md"), "w").write("\n".join(lines) + "\n")
        json.dump(rows, open(os.path.join(args.out, "sweep.json"), "w"), indent=1)
        print(f"\nSaved {args.out}/sweep.md and {args.out}/sweep.json")
        raise SystemExit(0)

    print(f"Generating {args.samples} pieces ...")
    generated = evaluate_setting(model, vocab, seeds, args.samples, args.length, args.temperature, args.top_k, args.top_p,
                                 args.no_repeat)
    similarity = pitch_similarity(real, generated)

    names = {"scale_ratio": "Notes inside the key", "distinct_4gram": "Variety (distinct 4-grams)",
             "rest_ratio": "Share of rests", "avg_polyphony": "Notes per chord", "pitch_range": "Pitch range (semitones)",
             "avg_leap": "Average melodic jump", "avg_note_length": "Average note length (beats)"}
    print(f"\n{'metric':32s}{'real music':>12s}{'generated':>12s}")
    for key, label in names.items():
        print(f"{label:32s}{real[key]:12.2f}{generated[key]:12.2f}")
    print(f"{'Pitch-class similarity':32s}{'':12s}{similarity:12.2f}   (1.00 = same note usage as real music)")

    os.makedirs(args.out, exist_ok=True)
    report = {"real": {k: v for k, v in real.items() if k != "_hist"},
              "generated": {k: v for k, v in generated.items() if k != "_hist"},
              "pitch_class_similarity": similarity, "samples": args.samples,
              "settings": {"temperature": args.temperature, "top_k": args.top_k, "top_p": args.top_p,
                           "no_repeat": args.no_repeat}}
    json.dump(report, open(os.path.join(args.out, "report.json"), "w"), indent=1)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        x = np.arange(12)
        labels = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        axes[0].bar(x - 0.2, real["_hist"], 0.4, label="real music")
        axes[0].bar(x + 0.2, generated["_hist"], 0.4, label="generated")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels)
        axes[0].set_title(f"Which notes are used (similarity {similarity:.2f})")
        axes[0].legend()
        keys = ["scale_ratio", "distinct_4gram", "rest_ratio"]
        y = np.arange(len(keys))
        axes[1].bar(y - 0.2, [real[k] for k in keys], 0.4, label="real music")
        axes[1].bar(y + 0.2, [generated[k] for k in keys], 0.4, label="generated")
        axes[1].set_xticks(y)
        axes[1].set_xticklabels(["in key", "variety", "rests"])
        axes[1].set_title("Music quality metrics")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "evaluation.png"), dpi=150)
        print(f"\nSaved {args.out}/report.json and {args.out}/evaluation.png")
    except ImportError:
        pass