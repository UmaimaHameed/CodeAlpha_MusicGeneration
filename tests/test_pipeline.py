# End-to-end test: the real scripts, one after the other, on a handful of Bach pieces.
# It checks that collect -> preprocess -> train -> generate -> evaluate -> baseline all work together.
import json
import os
import subprocess
import sys
import wave
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("music21")

ROOT = Path(__file__).resolve().parent.parent


def run(args, cwd):
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONIOENCODING="utf-8")
    result = subprocess.run([sys.executable, str(ROOT / args[0]), *args[1:]], cwd=cwd, env=env,
                            capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, f"{args[0]} failed:\n{result.stdout}\n{result.stderr}"
    return result.stdout


def test_the_whole_pipeline_runs(tmp_path):
    run(["collect_data.py", "--composers", "bach", "--max-files", "10"], tmp_path)
    assert len(list((tmp_path / "data" / "midi").glob("*.mid"))) == 10

    out = run(["preprocess.py"], tmp_path)
    assert "vocabulary" in out
    assert (tmp_path / "processed" / "dataset.npz").exists() and (tmp_path / "processed" / "vocab.json").exists()

    out = run(["train.py", "--epochs", "2", "--hidden", "32", "--batch-size", "32", "--seed", "1"], tmp_path)
    assert "Best validation loss" in out
    assert (tmp_path / "model" / "music_lstm.pt").exists() and (tmp_path / "model" / "history.json").exists()
    assert json.load(open(tmp_path / "model" / "history.json"))[0]["epoch"] == 1

    run(["generate.py", "--length", "25", "--seed", "1", "--no-repeat", "12", "--instrument", "Organ"], tmp_path)
    run(["generate.py", "--length", "10", "--notes", "C3 E3 G3", "--out", "output_typed"], tmp_path)
    for folder in ("output", "output_typed"):
        files = {p.suffix for p in (tmp_path / folder).iterdir()}
        assert files == {".mid", ".wav", ".png"}
    wav = next((tmp_path / "output").glob("*.wav"))
    with wave.open(str(wav)) as f:
        assert f.getnframes() / f.getframerate() > 3

    run(["evaluate.py", "--samples", "3", "--length", "20"], tmp_path)
    report = json.load(open(tmp_path / "evaluation" / "report.json"))
    assert 0 <= report["pitch_class_similarity"] <= 1 and report["samples"] == 3

    run(["evaluate.py", "--samples", "2", "--length", "15", "--sweep"], tmp_path)
    assert (tmp_path / "evaluation" / "sweep.md").read_text().count("|") > 10

    out = run(["baseline.py"], tmp_path)
    table = json.load(open(tmp_path / "evaluation" / "baseline.json"))
    assert [row["model"] for row in table][0] == "Random guess" and table[-1]["model"] == "LSTM (our model)"
    assert "Perplexity" in out
