# Helpers shared by the scripts: MIDI -> note tokens, tokens -> MIDI, tokens -> audio (WAV) and piano-roll picture.
# (music21 is only imported when a MIDI file has to be READ, so the web app starts faster.)
import re
import wave

import numpy as np

# note lengths (in quarter notes) that we allow, everything is rounded to the closest one
DURATIONS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]

# simple instrument sounds for the built-in synthesizer
#   h = (harmonic number, loudness) pairs, attack = fade-in time, sustain = keeps sounding while the note is held,
#   decay = how fast the sound fades (short notes, long notes), shift = octave shift, detune = extra slightly
#   detuned copies that make the sound warmer, program = instrument number used inside the MIDI file
INSTRUMENTS = {
    "Piano": dict(h=((1, 1.0), (2, 0.5), (3, 0.25), (4, 0.12)), attack=0.01, decay=(3.0, 1.8), sustain=False, shift=0, detune=(), program=0),
    "Organ": dict(h=((1, 1.0), (2, 0.7), (3, 0.5), (4, 0.35), (6, 0.2)), attack=0.03, decay=None, sustain=True, shift=0, detune=(), program=19),
    "Music box": dict(h=((1, 1.0), (3, 0.2), (5, 0.08)), attack=0.003, decay=(5.0, 3.5), sustain=False, shift=12, detune=(), program=10),
    "Strings": dict(h=((1, 1.0), (2, 0.6), (3, 0.4), (4, 0.3), (5, 0.2), (6, 0.12)), attack=0.12, decay=None, sustain=True, shift=0, detune=(1.004, 0.996), program=48),
}


def quantize(length):
    return min(DURATIONS, key=lambda d: abs(d - length))


def move_to_c(score):
    # transpose every song to C major / A minor, so the model has fewer different notes to learn
    try:
        key = score.analyze("key")
        target = 0 if key.mode == "major" else 9
        shift = (target - key.tonic.pitchClass) % 12
        if shift > 6:
            shift -= 12
        return score.transpose(shift)
    except Exception:
        return score


def midi_to_events(path):
    """Read a MIDI file and return a list of (pitches, length). No pitches means a rest."""
    from music21 import chord, converter, note
    try:
        score = move_to_c(converter.parse(path))
        elements = score.chordify().flatten().notesAndRests
    except Exception as error:
        print("  could not read", path, "-", error)
        return []
    events = []
    for el in elements:
        length = quantize(float(el.quarterLength))
        if isinstance(el, note.Rest):
            events.append(([], length))
        elif isinstance(el, chord.Chord):
            events.append((sorted(p.midi for p in el.pitches)[:4], length))  # at most 4 notes at once
    return events


def events_to_tokens(events):
    # a chord becomes one token per note and then one token with its length:
    # (48, 64, 67) for 0.5 beats  ->  N48 N64 N67 D0.5      a rest of 1 beat  ->  D1.0
    tokens = []
    for pitches, length in events:
        tokens += [f"N{p}" for p in pitches] + [f"D{length}"]
    return tokens


def tokens_to_events(tokens):
    events, current = [], []
    for token in tokens:
        if token.startswith("N"):
            if len(current) < 4:
                current.append(int(token[1:]))
        elif token.startswith("D"):
            events.append((sorted(set(current)), float(token[1:])))
            current = []
    return events


def midi_to_tokens(path):
    return events_to_tokens(midi_to_events(path))


# ---------------------------------------------------------------- typing notes by hand
_LETTERS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_name(p):
    return f"{_NAMES[p % 12]}{p // 12 - 1}"


def parse_notes(text):
    """Turn typed notes into tokens.  'C4 E4 G4'  = three notes, 'C4+E4+G4' = one chord,
    'C4:0.5' = a note of half a beat (default length is 1 beat)."""
    tokens = []
    for word in text.replace(",", " ").split():
        length = 1.0
        if ":" in word:
            word, _, value = word.partition(":")
            try:
                length = quantize(float(value))
            except ValueError:
                raise ValueError(f"'{value}' is not a note length. Use for example C4:0.5")
        pitches = []
        for part in word.split("+"):
            m = re.fullmatch(r"([A-Ga-g])([#b]?)(-?\d)", part)
            if not m:
                raise ValueError(f"I do not understand '{part}'. Write notes like C4 E4 G4, or C4+E4+G4 for a chord.")
            pitch_class = _LETTERS[m.group(1).upper()] + {"#": 1, "b": -1, "": 0}[m.group(2)]
            pitches.append(12 * (int(m.group(3)) + 1) + pitch_class)
        tokens += [f"N{p}" for p in sorted(set(pitches))[:4]] + [f"D{length}"]
    return tokens


def transpose_tokens(tokens, semitones):
    """Move the whole piece up or down (the model writes in C major / A minor, this changes the key)."""
    if not semitones:
        return list(tokens)
    return [f"N{int(t[1:]) + semitones}" if t.startswith("N") else t for t in tokens]


def tokens_to_notes(tokens, bpm=100):
    """List of (start second, length in seconds, pitch) for every note + total seconds. Used by the web player."""
    sec_per_beat = 60.0 / bpm
    notes, beat = [], 0.0
    for pitches, length in tokens_to_events(tokens):
        for p in pitches:
            notes.append((round(beat * sec_per_beat, 3), round(length * sec_per_beat, 3), p))
        beat += length
    return notes, beat * sec_per_beat


# ---------------------------------------------------------------- output files
def _vlq(value):
    """MIDI 'variable length' number."""
    out = [value & 0x7F]
    value >>= 7
    while value:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(out))


def tokens_to_midi(tokens, path, bpm=100, instrument="Piano"):
    """Write a standard MIDI file (a tiny writer of our own: much faster than building a music21 stream)."""
    ticks_per_beat = 480
    program = INSTRUMENTS.get(instrument, INSTRUMENTS["Piano"])["program"]
    track = bytearray(b"\x00\xff\x51\x03" + int(60_000_000 / bpm).to_bytes(3, "big"))  # tempo
    track += b"\x00\xc0" + bytes([program])  # instrument
    wait = 0  # ticks of silence before the next note
    for pitches, length in tokens_to_events(tokens):
        ticks = int(round(length * ticks_per_beat))
        pitches = [p for p in pitches if 0 <= p <= 127]
        if not pitches:
            wait += ticks
            continue
        for k, p in enumerate(pitches):
            track += _vlq(wait if k == 0 else 0) + bytes([0x90, p, 80])  # note on
        for k, p in enumerate(pitches):
            track += _vlq(ticks if k == 0 else 0) + bytes([0x80, p, 0])  # note off
        wait = 0
    track += _vlq(wait) + b"\xff\x2f\x00"
    header = b"MThd" + (6).to_bytes(4, "big") + (0).to_bytes(2, "big") + (1).to_bytes(2, "big") + ticks_per_beat.to_bytes(2, "big")
    with open(path, "wb") as f:
        f.write(header + b"MTrk" + len(track).to_bytes(4, "big") + bytes(track))


def add_reverb(audio, sr, mix=0.2, decay=1.3):
    """Small room reverb (noise that fades out) -> the sound is less 'dry' and robotic."""
    n = int(sr * decay)
    rng = np.random.default_rng(0)
    impulse = rng.standard_normal(n) * np.exp(-np.arange(n) / (sr * decay / 4))
    impulse /= np.sqrt((impulse ** 2).sum())
    size = 1 << (len(audio) + n - 1).bit_length()  # power of two = fast FFT
    wet = np.fft.irfft(np.fft.rfft(audio, size) * np.fft.rfft(impulse, size), size)[: len(audio)]
    return (1 - mix) * audio + mix * wet


def _render_note(midi, dur, inst, sr):
    freq = 440.0 * 2 ** ((midi + inst["shift"] - 69) / 12)
    tt = np.arange(int((dur + 0.8) * sr)) / sr
    tone = sum(amp * np.sin(2 * np.pi * freq * h * tt) for h, amp in inst["h"])
    for ratio in inst["detune"]:
        tone = tone + 0.6 * sum(amp * np.sin(2 * np.pi * freq * ratio * h * tt) for h, amp in inst["h"])
    fade_in = np.minimum(tt / inst["attack"], 1.0)
    if inst["sustain"]:
        envelope = fade_in * np.exp(-np.clip(tt - dur, 0, None) * 9.0)
    else:
        envelope = fade_in * np.exp(-tt * (inst["decay"][0] if dur < 1 else inst["decay"][1]))
    return (tone * envelope).astype(np.float32)


def tokens_to_wav(tokens, path, bpm=100, sr=22050, reverb=0.2, instrument="Piano"):
    """Simple synthesizer (piano, organ, music box, strings), so we can listen without any extra software.
    reverb = amount of room reverb from 0.0 (none) to 0.5. (True/False also works.)"""
    inst = INSTRUMENTS.get(instrument, INSTRUMENTS["Piano"])
    reverb = 0.2 if reverb is True else float(reverb or 0.0)
    sec_per_beat = 60.0 / bpm
    events, t = [], 0.0
    for notes, length in tokens_to_events(tokens):
        events.append((t, notes, length * sec_per_beat))
        t += length * sec_per_beat
    audio = np.zeros(int((t + 2.5) * sr))

    cache = {}  # the same note (pitch + length) sounds the same every time, so it is made only once
    for start, notes, dur in events:
        for midi in notes:
            key = (midi, round(dur, 4))
            if key not in cache:
                cache[key] = _render_note(midi, dur, inst, sr)
            i = int(start * sr)
            segment = cache[key][: len(audio) - i]
            audio[i:i + len(segment)] += segment

    if reverb > 0:
        audio = add_reverb(audio, sr, mix=reverb)
    audio = audio / max(np.abs(audio).max(), 1e-9) * 0.8
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes((audio * 32767).astype(np.int16).tobytes())


def plot_piano_roll(tokens, path, title="Generated music (piano roll)"):
    """Save a picture of the music: time on the x-axis, pitch on the y-axis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lefts, pitches_all, widths, colors = [], [], [], []
    t = 0.0
    for pitches, length in tokens_to_events(tokens):
        for p in pitches:
            lefts.append(t)
            pitches_all.append(p)
            widths.append(length * 0.95)
            colors.append(plt.cm.viridis(min(max((p - 30) / 60.0, 0), 1)))
        t += length
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.barh(pitches_all, widths, left=lefts, height=0.85, color=colors)  # one call for all notes
    ax.set_xlabel("time (beats)")
    ax.set_ylabel("pitch (MIDI note number)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)