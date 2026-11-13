# Step 6: is the LSTM really better than very simple statistics?
# Every serious model should be compared with simple baselines. Here we measure, on the SAME validation songs
# (never seen in training), how well each method guesses the next token:
#   - random guess        : every token equally likely
#   - n-gram models       : only count which token followed the last 0, 1, 2 or 3 tokens in the training songs
#   - LSTM (our model)    : reads the whole piece so far
# Lower perplexity is better, higher accuracy is better.
#   python baseline.py
import argparse
import json
import os

import numpy as np


class NGram:
    """Interpolated n-gram model with add-k smoothing. order 1 = unigram (no context), order 4 = 3 tokens of context."""

    def __init__(self, vocab_size, order, k=0.1):
        self.v, self.order, self.k = vocab_size, order, k
        self.counts = [dict() for _ in range(order)]  # counts[m][context of m tokens] -> counts of the next token
        total = sum(range(1, order + 1))
        self.weight = [(m + 1) / total for m in range(order)]  # longer context = more weight

    def fit(self, sequences):
        for seq in sequences:
            seq = [int(t) for t in seq]
            for i, token in enumerate(seq):
                for m in range(self.order):
                    if i - m < 0:
                        break
                    table = self.counts[m].setdefault(tuple(seq[i - m:i]), np.zeros(self.v))
                    table[token] += 1
        return self

    def probabilities(self, context):
        """Distribution over the next token given the previous tokens (a list)."""
        p, used = np.zeros(self.v), 0.0
        for m in range(self.order):
            if m > len(context):
                break
            ctx = tuple(context[len(context) - m:]) if m else ()
            counts = self.counts[m].get(ctx)
            if counts is None:
                counts = np.zeros(self.v)
            p += self.weight[m] * (counts + self.k) / (counts.sum() + self.k * self.v)
            used += self.weight[m]
        return p / used


def score_ngram(model, X, Y):
    nll, correct, n = 0.0, 0, 0
    for x_row, y_row in zip(X, Y):
        x_row = [int(t) for t in x_row]
        for t, target in enumerate(y_row):
            p = model.probabilities(x_row[max(0, t + 1 - (model.order - 1)): t + 1])
            nll -= np.log(p[int(target)])
            correct += int(p.argmax() == int(target))
            n += 1
    return nll / n, correct / n


def score_lstm(model_path, X, Y):
    """Loss and accuracy of the trained LSTM on the same validation pieces (None if torch/model is missing)."""
    try:
        import torch
        from generator import load_model
        model, vocab = load_model(model_path)
    except Exception as error:
        print(f"  (could not load the LSTM: {error})")
        return None
    nll, correct, n = 0.0, 0, 0
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for i in range(0, len(X), 64):
            xb, yb = torch.from_numpy(X[i:i + 64]), torch.from_numpy(Y[i:i + 64])
            logits, _ = model(xb)
            nll += loss_fn(logits.reshape(-1, logits.shape[-1]), yb.reshape(-1)).item()
            correct += (logits.argmax(-1) == yb).sum().item()
            n += yb.numel()
    return nll / n, correct / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="processed")
    parser.add_argument("--model", default="model/music_lstm.pt")
    parser.add_argument("--out", default="evaluation")
    args = parser.parse_args()

    data = np.load(os.path.join(args.data, "dataset.npz"))
    if "X_train" not in data.files:
        raise SystemExit(f"{args.data}/dataset.npz was made by an older version of the project. Run: python preprocess.py")
    vocab = json.load(open(os.path.join(args.data, "vocab.json")))
    X_train, Y_train, X_val, Y_val = data["X_train"], data["Y_train"], data["X_val"], data["Y_val"]
    v = len(vocab)
    # a training piece = its inputs plus the very last target (pieces overlap, which does not matter for the probabilities)
    train_sequences = [list(x) + [y[-1]] for x, y in zip(X_train, Y_train)]

    rows = [("Random guess", float(np.log(v)), 1.0 / v)]
    for order, name in ((1, "Unigram (no context)"), (2, "Bigram (1 token of context)"),
                        (3, "Trigram (2 tokens)"), (4, "4-gram (3 tokens)")):
        print(f"fitting {name} ...")
        nll, acc = score_ngram(NGram(v, order).fit(train_sequences), X_val, Y_val)
        rows.append((name, nll, acc))
    print("scoring the LSTM ...")
    lstm = score_lstm(args.model, X_val, Y_val)
    if lstm:
        rows.append(("LSTM (our model)", lstm[0], lstm[1]))

    table = [{"model": name, "val_loss": round(nll, 3), "perplexity": round(float(np.exp(nll)), 2), "accuracy": round(acc, 4)}
             for name, nll, acc in rows]
    lines = ["| Model | Validation loss | Perplexity | Next-token accuracy |", "|---|---|---|---|"]
    lines += [f"| {r['model']} | {r['val_loss']:.3f} | {r['perplexity']:.2f} | {r['accuracy']:.1%} |" for r in table]
    print("\n" + "\n".join(lines))
    if lstm:
        ngrams = [r for r in table if r["model"].startswith(("Unigram", "Bigram", "Trigram", "4-gram"))]
        best = min(ngrams, key=lambda r: r["perplexity"])
        print(f"\nThe LSTM's perplexity is {table[-1]['perplexity']:.2f}; the best simple n-gram model reaches {best['perplexity']:.2f} "
              f"({best['model']}).")

    os.makedirs(args.out, exist_ok=True)
    json.dump(table, open(os.path.join(args.out, "baseline.json"), "w"), indent=1)
    open(os.path.join(args.out, "baseline.md"), "w").write("\n".join(lines) + "\n")
    print(f"Saved {args.out}/baseline.json and {args.out}/baseline.md")


if __name__ == "__main__":
    main()