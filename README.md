# 🎼 CodeAlpha_MusicGeneration

**AI music generation with an LSTM neural network (PyTorch).** The model learns from classical MIDI music (Bach) and writes
new pieces, note by note. Includes a web app, an evaluation against real music, a comparison with simple baselines and
automated tests.

Task 3 of my CodeAlpha Artificial Intelligence internship.

## Task requirements and where they are done

| Task requirement | Where |
|---|---|
| Collect MIDI music data | `collect_data.py` (music21 Bach corpus, or your own `.mid` files) |
| Preprocess the data into note sequences | `preprocess.py`, `music_utils.py` |
| Build a deep learning model (RNN / LSTM) | `train.py` (`MusicLSTM`) |
| Train the model on the dataset | `train.py` |
| Convert generated sequences to MIDI, play or save them | `generate.py`, `generator.py`, `music_utils.py` (`.mid` and `.wav`) |
| Extras | web app (`app.py`), evaluation (`evaluate.py`), baselines (`baseline.py`), tests (`tests/`) |

## Quick start (no training needed)

The repository contains the trained model (`model/`) and the prepared data (`processed/`).

```bash
pip install -r requirements.txt
python -m streamlit run app.py          # web app
python generate.py --count 3            # or: 3 new pieces in output/ (.mid, .wav and a piano-roll picture)
```

## Web app

`python -m streamlit run app.py`

- **The AI composes live.** Every press on *Compose* makes the LSTM write new notes, one token after the other. Nothing is pre-recorded;
  the synthesizer only turns the notes into sound.
- **Give your own start:** type notes (`C3 E3 G3 C4`, `+` for a chord, `:0.5` for a short note) or upload a MIDI file, and the AI continues it.
- Choose **mood**, **instrument** (piano, organ, music box, strings), **key**, **tempo**, **reverb** and up to 3 **variations**.
- The player shows an **animated piano roll** (notes light up while they play). Download the **MIDI** or **WAV** file.
- A second tab shows the training results and the evaluation.

## Run the whole pipeline yourself

```bash
python collect_data.py                  # add --composers bach mozart beethoven for more data
python preprocess.py
python train.py --epochs 40             # on a laptop CPU use fewer epochs, or a free GPU (Google Colab)
python generate.py
python evaluate.py                      # generated vs real music -> evaluation/
python baseline.py                      # LSTM vs simple n-gram models -> evaluation/
python -m pytest                        # automated tests
```

More options:

```bash
python generate.py --temperature 0.7 --length 200 --bpm 90 --instrument Strings
python generate.py --seed 42                     # the same music again
python generate.py --notes "C3 E3 G3 C4"         # the AI continues the notes you type
python generate.py --primer my_song.mid          # ...or your own MIDI file
python generate.py --no-repeat 16                # forbid exact loops
python evaluate.py --sweep                       # compare several temperatures
python train.py --hidden 512 --layers 3 --seed 1 # a bigger model
```

## How it works

1. **Collect:** MIDI files of classical pieces (about 430 Bach pieces by default).
2. **Preprocess:** every song is moved to C major / A minor, so the model has fewer different notes to learn. A chord becomes one token
   per note and one token for its length: three notes for half a beat are `N48 N64 N67 D0.5`; a rest of one beat is `D1.0`.
   The vocabulary is only **70 tokens**. Songs are cut into pieces of 128 tokens. **Whole songs are kept apart for validation**, so the
   scores are measured on music the model has never seen.
3. **Model:** embedding (64) → 2 × LSTM (256 units) → dropout (0.3) → dense layer. At every position it predicts the next token.
   Adam optimizer, learning-rate scheduler, gradient clipping, early stopping, best model saved.
4. **Generate:** the model reads a start (a real song opening, your typed notes or your MIDI) and continues it token by token.
   *Temperature*, *top-k* and *top-p* (nucleus) sampling control how safe or surprising the music is, and an optional
   *no-repeat* rule stops exact loops.
5. **Output:** a standard MIDI file (own small writer, verified in the tests), a WAV file (built-in synthesizer with reverb) and a piano-roll picture.

## Results

My run: 431 songs (388 for training, 43 for validation), 4,595 training pieces, **0.88 M parameters**, trained for 15 epochs (early stopping).

| Validation accuracy (next token) | Validation perplexity |
|---|---|
| **66.5 %** | **2.9** |

![Training curve](images/training_curve.png)

### Generated music vs real Bach music (never seen in training)

`python evaluate.py` generates 30 pieces and compares their statistics with real validation music:

| Metric | Real music | Generated |
|---|---|---|
| Notes inside the key | 0.93 | 0.96 |
| Variety (distinct 4-grams) | 0.90 | 0.83 |
| Share of rests | 0.01 | 0.00 |
| Notes per chord | 3.86 | 3.92 |
| Pitch range (semitones) | 31.36 | 31.13 |
| Average melodic jump | 1.32 | 1.17 |
| Average note length (beats) | 0.64 | 0.59 |

**Pitch-class similarity: 0.96** (1.00 = the notes are used exactly like in real music).

![Evaluation](images/evaluation.png)

### Is the LSTM better than simple statistics?

`python baseline.py` compares the LSTM with a random guess and with n-gram models (they only count what followed the last
few tokens) on the same validation pieces. Lower perplexity is better.

<!-- paste the table from evaluation/baseline.md here -->

### Effect of the temperature

`python evaluate.py --sweep` shows how the setting changes the music. Low temperature is safe but repeats more,
high temperature is more varied but leaves the key sooner.

<!-- paste the table from evaluation/sweep.md here -->

### Listen

The best generated pieces are in [`samples/`](samples/) (`.wav` to listen, `.mid` to open in any music program).

## Tests

```bash
python -m pytest
```

49 automated tests (GitHub Actions runs them on every push): token and MIDI conversion (the MIDI writer is checked against an
independent reader), the audio synthesizer, typed-note parsing, the sampling rules, saving and loading models (also old
checkpoints), and an **end-to-end test** that runs collect → preprocess → train → generate → evaluate → baseline on a few Bach pieces.

## Project structure

```
app.py            web app (Streamlit)
collect_data.py   step 1: get MIDI files
preprocess.py     step 2: MIDI -> token sequences
train.py          step 3: build and train the LSTM
generate.py       step 4: generate music (command line)
evaluate.py       step 5: generated vs real music (+ temperature sweep)
baseline.py       step 6: LSTM vs n-gram baselines
generator.py      generation functions shared by the scripts and the app
sampling.py       no-repeat rule
music_utils.py    MIDI/tokens/WAV helpers
tests/            automated tests
model/            trained model, training history and curve
processed/        prepared dataset and vocabulary
evaluation/       evaluation results
```

## Limitations (honest notes)

- The model is small and trained **only on Bach**, so the result is Bach-style music. It is not a finished song and has no drums, vocals or other styles.
- Every song is flattened into **block chords** (up to 4 notes at once), so the texture is dense and has almost no rests. Separate melody and
  accompaniment tracks would sound more natural.
- Notes are rounded to a few allowed lengths, and dynamics (loud/soft) are not modelled.
- The built-in synthesizer is simple (it is not a sampled piano). Open the `.mid` file in a music program for a richer sound.

**Ideas to improve it:** more composers or jazz MIDI files (`collect_data.py --composers ...` or copy files into `data/midi`),
separate melody/accompaniment tokens, a Transformer model, and training longer on a GPU.

## Data and licence

The Bach pieces come from the [music21](https://web.mit.edu/music21/) corpus (public domain). Only use MIDI files you are allowed to use.
Code: MIT licence (see `LICENSE`).
