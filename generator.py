# Reusable generation code. generate.py, evaluate.py, baseline.py and app.py all use these functions.
import os
import random
import re

import numpy as np
import torch

from music_utils import midi_name, midi_to_tokens, parse_notes
from sampling import NoRepeat
from train import MusicLSTM


def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


OLD_MODEL_HELP = (
    "This model/data was made by an older version of the project (its vocabulary is not in the N60 / D0.5 token format). "
    "Copy the matching model/ and processed/ folders into this project, or make them again with:\n"
    "    python collect_data.py\n    python preprocess.py\n    python train.py --epochs 40"
)


def load_model(path="model/music_lstm.pt"):
    """Load a model saved by train.py. The size of the network is read from the saved weights,
    so old checkpoints (without embed/layers) work too."""
    checkpoint = torch.load(path, map_location="cpu")
    vocab, state = checkpoint["vocab"], checkpoint["state"]
    if not all(re.fullmatch(r"[ND]\d+(\.\d+)?", str(t)) for t in vocab):
        raise ValueError(f"{path}: {OLD_MODEL_HELP}")
    layers = sum(1 for k in state if re.fullmatch(r"lstm\.weight_ih_l\d+", k))
    model = MusicLSTM(len(vocab), embed=state["embedding.weight"].shape[1],
                      hidden=state["lstm.weight_hh_l0"].shape[1], layers=layers)
    model.load_state_dict(state)
    model.eval()
    return model, vocab


def load_seeds(data="processed"):
    """Beginnings of real songs. Returns None when the processed data is not there (the app still works)."""
    path = os.path.join(data, "dataset.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path)
    return data["seeds"] if "seeds" in data.files else None


def _trim_to_event_end(ids, vocab):
    """Cut the seed after its last duration token, so we never start in the middle of a chord."""
    for i in range(len(ids) - 1, -1, -1):
        if vocab[ids[i]].startswith("D"):
            return ids[: i + 1]
    return ids


def default_seed(vocab):
    """A plain C major chord, used when there are no real song beginnings available."""
    index = {t: i for i, t in enumerate(vocab)}
    chord = [index[t] for t in ("N48", "N52", "N55", "N60") if t in index]
    if not chord:  # unusual vocabulary: start from a note in the middle of its range
        notes = [i for i, t in enumerate(vocab) if t.startswith("N")]
        chord = [notes[len(notes) // 2]] if notes else []
    duration = next((index[t] for t in ("D1.0", "D0.5") if t in index), None)
    if duration is None:
        duration = next(i for i, t in enumerate(vocab) if t.startswith("D"))
    return chord + [duration]


def pick_seed(seeds, vocab):
    """Beginning of a random real song from the training data (or a simple chord if there is no data)."""
    if seeds is None or len(seeds) == 0:
        return default_seed(vocab)
    return _trim_to_event_end(seeds[random.randrange(len(seeds))].tolist(), vocab)


def primer_from_midi(path, vocab, max_tokens=48):
    """Use the user's own MIDI file as the start of the piece (the model continues it)."""
    index = {t: i for i, t in enumerate(vocab)}
    ids = [index[t] for t in midi_to_tokens(path) if t in index][:max_tokens]
    ids = _trim_to_event_end(ids, vocab)
    if len(ids) < 4:
        raise ValueError("This MIDI file is too short or could not be read.")
    return ids


def primer_from_notes(text, vocab):
    """Notes typed by the user (for example 'C4 E4 G4 C5') as the start of the piece."""
    index = {t: i for i, t in enumerate(vocab)}
    tokens = parse_notes(text)
    unknown = sorted({t for t in tokens if t not in index})
    if unknown:
        pitches = sorted(int(t[1:]) for t in index if t.startswith("N"))
        raise ValueError(f"The model does not know {', '.join(unknown)}. It was trained on notes from "
                         f"{midi_name(pitches[0])} to {midi_name(pitches[-1])}, so please type notes in that range.")
    return [index[t] for t in tokens]


def sample_token(scores, temperature=0.9, top_k=8, top_p=1.0):
    """Temperature + top-k + top-p (nucleus) sampling."""
    scores = scores / max(temperature, 1e-3)
    k = min(top_k, scores.numel()) if top_k > 0 else scores.numel()
    top = torch.topk(scores, k)
    probs = torch.softmax(top.values, dim=0)
    if top_p < 1.0:
        keep = (torch.cumsum(probs, 0) - probs) < top_p  # always keeps the best token
        probs = probs * keep
        probs = probs / probs.sum()
    return top.indices[torch.multinomial(probs, 1)].item()


def generate_tokens(model, vocab, seed_ids, length=150, temperature=0.9, top_k=8, top_p=1.0, no_repeat=0):
    """Continue the seed token by token. `length` = number of chords/notes (events) to create.
    no_repeat = size of the token run that may not appear twice (0 = off, about 16 works well)."""
    tokens, events = [], 0
    guard = NoRepeat(no_repeat, seed_ids)
    with torch.no_grad():
        scores, state = model(torch.tensor([seed_ids]))
        scores = scores[0, -1]
        while events < length and len(tokens) < length * 8:
            banned = guard.banned()
            if banned and len(banned) < scores.numel():
                scores = scores.clone()
                scores[list(banned)] = float("-inf")
            t = sample_token(scores, temperature, top_k, top_p)
            guard.add(t)
            tokens.append(vocab[t])
            events += vocab[t].startswith("D")  # a "D" token ends one chord/note
            scores, state = model(torch.tensor([[t]]), state)
            scores = scores[0, -1]
    return tokens