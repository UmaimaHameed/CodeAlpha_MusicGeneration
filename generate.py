# Step 4: generate new music with the trained model and save it as MIDI + WAV + piano-roll picture
import argparse
import os
import time

from generator import generate_tokens, load_model, load_seeds, pick_seed, primer_from_midi, primer_from_notes, set_seed
from music_utils import INSTRUMENTS, plot_piano_roll, tokens_to_midi, tokens_to_wav

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="model/music_lstm.pt")
parser.add_argument("--data", default="processed")
parser.add_argument("--length", type=int, default=150, help="how many chords/notes (events) to generate")
parser.add_argument("--temperature", type=float, default=0.9, help="low = safe and repeating, high = more surprising")
parser.add_argument("--top-k", type=int, default=8, help="choose only from the k most likely next tokens")
parser.add_argument("--top-p", type=float, default=0.95, help="nucleus sampling: ignore very unlikely tokens (1.0 = off)")
parser.add_argument("--no-repeat", type=int, default=0, help="forbid repeating the same run of N tokens, about 16 avoids loops (0 = off)")
parser.add_argument("--bpm", type=int, default=100)
parser.add_argument("--instrument", default="Piano", choices=list(INSTRUMENTS), help="sound of the WAV file")
parser.add_argument("--primer", default=None, help="your own .mid file: the model continues it")
parser.add_argument("--notes", default=None, help='your own start notes, for example "C4 E4 G4 C5" (+ for chords, C4:0.5 for length)')
parser.add_argument("--count", type=int, default=1, help="how many pieces to generate")
parser.add_argument("--seed", type=int, default=None, help="random seed, to get the same music again")
parser.add_argument("--out", default="output")
args = parser.parse_args()

set_seed(args.seed)
model, vocab = load_model(args.model)
seeds = None if (args.primer or args.notes) else load_seeds(args.data)  # None = simple C chord start
os.makedirs(args.out, exist_ok=True)

for n in range(1, args.count + 1):
    if args.notes:
        start = primer_from_notes(args.notes, vocab)
    elif args.primer:
        start = primer_from_midi(args.primer, vocab)
    else:
        start = pick_seed(seeds, vocab)
    tokens = generate_tokens(model, vocab, start, args.length, args.temperature, args.top_k, args.top_p, args.no_repeat)
    if args.notes or args.primer:  # the piece begins with the user's own notes, then the AI continues
        tokens = [vocab[i] for i in start] + tokens
    name = os.path.join(args.out, f"generated_{time.strftime('%H%M%S')}_{n}")
    tokens_to_midi(tokens, name + ".mid", args.bpm, args.instrument)
    tokens_to_wav(tokens, name + ".wav", args.bpm, instrument=args.instrument)
    plot_piano_roll(tokens, name + ".png")
    print("First tokens:", " ".join(tokens[:12]), "...")
    print(f"Saved {name}.mid, {name}.wav and {name}.png")