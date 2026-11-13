# Tests for the note/token/MIDI/WAV helpers (they need only numpy, no PyTorch).
import random
import wave

import numpy as np
import pytest

from music_utils import (DURATIONS, INSTRUMENTS, events_to_tokens, midi_name, parse_notes, quantize, tokens_to_events,
                         tokens_to_midi, tokens_to_notes, tokens_to_wav, transpose_tokens)

TICKS = 480


def random_tokens(seed=0, events=60):
    rng = random.Random(seed)
    tokens = []
    for _ in range(events):
        tokens += [f"N{rng.choice([40, 48, 52, 55, 60, 64, 67, 72])}" for _ in range(rng.choice([0, 1, 2, 3, 4]))]
        tokens.append(f"D{rng.choice(DURATIONS)}")
    return tokens


def read_midi(path):
    """A tiny independent MIDI reader, so the test does not trust our own writer.
    Returns (ticks per beat, tempo in microseconds per beat, instrument number, sorted (start, pitch, end) notes)."""
    data = open(path, "rb").read()
    assert data[:4] == b"MThd" and data[14:18] == b"MTrk"
    ticks_per_beat = int.from_bytes(data[12:14], "big")
    size = int.from_bytes(data[18:22], "big")
    track = data[22:22 + size]
    assert len(track) == size
    i, now, tempo, program, started, notes = 0, 0, None, None, {}, []
    while i < len(track):
        delta = 0
        while True:
            byte = track[i]
            i += 1
            delta = (delta << 7) | (byte & 0x7F)
            if not byte & 0x80:
                break
        now += delta
        status = track[i]
        i += 1
        if status == 0xFF:
            kind, length = track[i], track[i + 1]
            payload = track[i + 2:i + 2 + length]
            i += 2 + length
            if kind == 0x51:
                tempo = int.from_bytes(payload, "big")
            if kind == 0x2F:
                break
        elif status == 0xC0:
            program = track[i]
            i += 1
        elif status == 0x90:
            started[track[i]] = now
            i += 2
        elif status == 0x80:
            notes.append((started.pop(track[i]), track[i], now))
            i += 2
        else:
            raise AssertionError(f"unexpected MIDI status {hex(status)}")
    assert not started, "a note was never switched off"
    return ticks_per_beat, tempo, program, sorted(notes)


def test_tokens_round_trip():
    events = [([48, 64, 67], 0.5), ([], 1.0), ([60], 2.0)]
    tokens = events_to_tokens(events)
    assert tokens == ["N48", "N64", "N67", "D0.5", "D1.0", "N60", "D2.0"]
    assert tokens_to_events(tokens) == events


def test_incomplete_last_chord_is_ignored_and_chords_are_limited_to_four_notes():
    assert tokens_to_events(["N60", "D1.0", "N62"]) == [([60], 1.0)]
    assert tokens_to_events(["N60", "N61", "N62", "N63", "N64", "D1.0"]) == [([60, 61, 62, 63], 1.0)]


def test_quantize_picks_the_closest_allowed_length():
    assert quantize(0.9) == 1.0
    assert quantize(10) == max(DURATIONS)
    assert quantize(0.01) == min(DURATIONS)


def test_parse_notes_single_notes_chords_and_lengths():
    assert parse_notes("C4 E4") == ["N60", "D1.0", "N64", "D1.0"]
    assert parse_notes("C4+E4+G4") == ["N60", "N64", "N67", "D1.0"]
    assert parse_notes("C4:0.5 Bb3 f#5:2") == ["N60", "D0.5", "N58", "D1.0", "N78", "D2.0"]
    assert parse_notes("c4, e4") == ["N60", "D1.0", "N64", "D1.0"]


@pytest.mark.parametrize("bad", ["H4", "C", "C4:abc", "44", "C#"])
def test_parse_notes_rejects_nonsense_with_a_helpful_message(bad):
    with pytest.raises(ValueError):
        parse_notes(bad)


def test_midi_name():
    assert midi_name(60) == "C4"
    assert midi_name(70) == "A#4"


def test_transpose_moves_only_the_notes():
    assert transpose_tokens(["N60", "D1.0", "N64", "D0.5"], 2) == ["N62", "D1.0", "N66", "D0.5"]
    original = ["N60", "D1.0"]
    assert transpose_tokens(original, 0) == original and transpose_tokens(original, 0) is not original


def test_tokens_to_notes_gives_start_length_and_total_time():
    notes, total = tokens_to_notes(["N60", "D1.0", "D1.0", "N62", "N65", "D2.0"], bpm=120)  # 0.5 s per beat
    assert notes == [(0.0, 0.5, 60), (1.0, 1.0, 62), (1.0, 1.0, 65)]
    assert total == pytest.approx(2.0)


@pytest.mark.parametrize("bpm,instrument", [(100, "Piano"), (60, "Strings"), (140, "Organ"), (90, "Music box")])
def test_midi_file_contains_exactly_the_notes_tempo_and_instrument(tmp_path, bpm, instrument):
    tokens = random_tokens()
    path = tmp_path / "x.mid"
    tokens_to_midi(tokens, str(path), bpm, instrument)
    ticks, tempo, program, notes = read_midi(path)
    expected, now = [], 0
    for pitches, length in tokens_to_events(tokens):
        for p in pitches:
            expected.append((now, p, now + round(length * TICKS)))
        now += round(length * TICKS)
    assert ticks == TICKS
    assert tempo == int(60_000_000 / bpm)
    assert program == INSTRUMENTS[instrument]["program"]
    assert notes == sorted(expected)


def test_midi_of_only_rests_is_still_a_valid_file(tmp_path):
    path = tmp_path / "rest.mid"
    tokens_to_midi(["D1.0", "D2.0"], str(path))
    assert read_midi(path)[3] == []


def read_wav(path):
    with wave.open(str(path)) as f:
        rate, frames = f.getframerate(), f.getnframes()
        audio = np.frombuffer(f.readframes(frames), dtype=np.int16)
        return rate, f.getnchannels(), f.getsampwidth(), audio


@pytest.mark.parametrize("instrument", list(INSTRUMENTS))
def test_wav_is_valid_loud_enough_and_the_right_length(tmp_path, instrument):
    tokens = ["N60", "N64", "D1.0", "N62", "D1.0", "N55", "D2.0"]  # 4 beats = 2.4 s at 100 BPM
    path = tmp_path / "x.wav"
    tokens_to_wav(tokens, str(path), bpm=100, instrument=instrument)
    rate, channels, width, audio = read_wav(path)
    assert (channels, width) == (1, 2)
    assert len(audio) == int((2.4 + 2.5) * rate)
    peak = np.abs(audio).max()
    assert 0.75 * 32767 < peak <= 0.81 * 32767            # normalised to 80 percent
    assert np.sqrt((audio[: 2 * rate].astype(float) ** 2).mean()) > 500   # the music part is not silent


@pytest.mark.parametrize("reverb", [0, 0.3, True])
def test_wav_reverb_options_work(tmp_path, reverb):
    path = tmp_path / "x.wav"
    tokens_to_wav(["N60", "D1.0"], str(path), reverb=reverb)
    assert np.abs(read_wav(path)[3]).max() > 0


def test_reverb_changes_the_tail_but_dry_sound_has_silence_after_the_music(tmp_path):
    tokens = ["N60", "D0.5"]
    dry, wet = tmp_path / "dry.wav", tmp_path / "wet.wav"
    tokens_to_wav(tokens, str(dry), reverb=0)
    tokens_to_wav(tokens, str(wet), reverb=0.4)
    rate, _, _, a = read_wav(dry)
    _, _, _, b = read_wav(wet)
    tail = slice(int(2.0 * rate), int(2.4 * rate))
    assert np.abs(b[tail]).max() > np.abs(a[tail]).max()


def test_same_note_sounds_identical_each_time(tmp_path):
    """The synthesizer caches notes: a repeated note must be rendered exactly like the first one."""
    path = tmp_path / "x.wav"
    tokens_to_wav(["N60", "D1.0", "D2.0", "N60", "D1.0"], str(path), bpm=60, reverb=0)  # second note starts at 3 s
    rate, _, _, audio = read_wav(path)
    first, second = audio[:rate], audio[3 * rate:4 * rate]
    assert np.array_equal(first, second)


def test_events_to_tokens_never_loses_rests():
    assert events_to_tokens([([], 0.5), ([], 0.25)]) == ["D0.5", "D0.25"]
