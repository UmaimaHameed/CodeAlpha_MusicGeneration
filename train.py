# Step 3: build and train the LSTM model (PyTorch)
# The model reads a piece of music token by token and at every position it has to guess the next token.
import argparse
import json
import os
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class MusicLSTM(nn.Module):
    def __init__(self, vocab_size, embed=64, hidden=256, layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed)
        self.lstm = nn.LSTM(embed, hidden, layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, vocab_size)

    def forward(self, x, state=None):
        h, state = self.lstm(self.embedding(x), state)
        return self.out(self.dropout(h)), state  # (batch, length, vocab) scores for the next token


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="processed")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--embed", type=int, default=64, help="size of the note embedding")
    parser.add_argument("--layers", type=int, default=2, help="number of LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0, help="random seed, so a training run can be repeated")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--patience", type=int, default=6, help="stop if the validation loss does not improve for this many epochs")
    parser.add_argument("--out", default="model/music_lstm.pt")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = np.load(os.path.join(args.data, "dataset.npz"))
    vocab = json.load(open(os.path.join(args.data, "vocab.json")))
    train_loader = DataLoader(TensorDataset(torch.from_numpy(data["X_train"]), torch.from_numpy(data["Y_train"])),
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(data["X_val"]), torch.from_numpy(data["Y_val"])),
                            batch_size=args.batch_size)

    model = MusicLSTM(len(vocab), embed=args.embed, hidden=args.hidden, layers=args.layers, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    loss_fn = nn.CrossEntropyLoss()
    n_train, n_val = len(train_loader.dataset), len(val_loader.dataset)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device: {device} | vocabulary: {len(vocab)} | training pieces: {n_train} | validation pieces: {n_val} | "
          f"parameters: {n_params:,}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    history, best, waited = [], float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        total, count = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = loss_fn(logits.reshape(-1, len(vocab)), yb.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * yb.numel()
            count += yb.numel()
        train_loss = total / count

        model.eval()
        val_total, val_count, correct = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits, _ = model(xb)
                val_total += loss_fn(logits.reshape(-1, len(vocab)), yb.reshape(-1)).item() * yb.numel()
                correct += (logits.argmax(-1) == yb).sum().item()
                val_count += yb.numel()
        val_loss, val_acc = val_total / val_count, correct / val_count
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc,
                        "val_perplexity": float(np.exp(val_loss))})
        print(f"epoch {epoch:2d}/{args.epochs}  train loss {train_loss:.3f}  val loss {val_loss:.3f}  "
              f"val accuracy {val_acc:.1%}  perplexity {np.exp(val_loss):.1f}  ({time.time() - start:.0f}s)")

        if val_loss < best:  # keep the best model
            best, waited = val_loss, 0
            torch.save({"state": model.state_dict(), "vocab": vocab, "hidden": args.hidden,
                        "embed": args.embed, "layers": args.layers}, args.out)
        else:
            waited += 1
            if waited >= args.patience:
                print("The validation loss stopped improving, so training stops here (the best model is saved).")
                break

    json.dump(history, open(os.path.join(os.path.dirname(args.out), "history.json"), "w"), indent=1)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4))
        plt.plot([h["train_loss"] for h in history], label="train loss")
        plt.plot([h["val_loss"] for h in history], label="validation loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(args.out), "training_curve.png"), dpi=150)
    except ImportError:
        pass
    print(f"Best validation loss {best:.3f} (perplexity {np.exp(best):.1f}). Model saved in {args.out}")