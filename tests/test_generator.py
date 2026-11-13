# Tests for the model + generation code (need PyTorch; skipped when it is not installed).
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from generator import (default_seed, generate_tokens, load_model, load_seeds, pick_seed, primer_from_notes, sample_token,
                       set_seed, _trim_to_event_end)  # noqa: E402
from train import MusicLSTM  # noqa: E402

NOTES = [f"N{p}" for p in range(40, 76)]
VOCAB = NOTES + ["D0.5", "D1.0", "D2.0"]


@pytest.fixture()
def model():
    """A small random model. Its output is nudged towards duration tokens (a trained model learns to end chords)."""
    torch.manual_seed(0)
    net = MusicLSTM(len(VOCAB), embed=16, hidden=32, layers=2).eval()
    with torch.no_grad():
        net.out.bias[[i for i, t in enumerate(VOCAB) if t.startswith("D")]] += 4.0
    return net


def seed_ids():
    return [VOCAB.index(t) for t in ("N48", "N52", "N55", "D1.0")]


def test_model_output_shape(model):
    scores, state = model(torch.zeros(2, 5, dtype=torch.long))
    assert scores.shape == (2, 5, len(VOCAB))
    assert len(state) == 2


def test_top_k_one_and_tiny_top_p_pick_the_best_token():
    scores = torch.tensor([0.1, 3.0, 0.2, 1.0])
    assert sample_token(scores, temperature=1.0, top_k=1) == 1
    assert sample_token(scores, temperature=1.0, top_k=4, top_p=0.01) == 1


def test_sampling_only_uses_the_top_k_tokens():
    scores = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    torch.manual_seed(1)
    drawn = {sample_token(scores, temperature=5.0, top_k=3) for _ in range(300)}
    assert drawn <= {0, 1, 2} and len(drawn) > 1


def test_higher_temperature_gives_more_variety():
    scores = torch.tensor([3.0, 2.0, 1.0, 0.5, 0.0])
    torch.manual_seed(2)
    cold = {sample_token(scores, 0.1, top_k=5) for _ in range(200)}
    hot = {sample_token(scores, 2.0, top_k=5) for _ in range(200)}
    assert len(cold) < len(hot)


def test_generate_creates_the_requested_number_of_events_with_valid_tokens(model):
    tokens = generate_tokens(model, VOCAB, seed_ids(), length=40, temperature=1.0, top_k=8)
    assert sum(t.startswith("D") for t in tokens) == 40
    assert set(tokens) <= set(VOCAB)
    assert tokens[-1].startswith("D")  # ends after a complete chord


def test_the_same_seed_gives_the_same_music(model):
    set_seed(5)
    a = generate_tokens(model, VOCAB, seed_ids(), length=30)
    set_seed(5)
    b = generate_tokens(model, VOCAB, seed_ids(), length=30)
    assert a == b


def test_no_repeat_prevents_loops(model):
    run = 8
    for temperature in (0.3, 1.0):
        set_seed(3)
        tokens = generate_tokens(model, VOCAB, seed_ids(), length=120, temperature=temperature, top_k=6, no_repeat=run)
        grams = [tuple(tokens[i:i + run]) for i in range(len(tokens) - run + 1)]
        assert len(grams) == len(set(grams))


def test_trim_cuts_the_seed_after_the_last_duration_token():
    vocab = ["N60", "N64", "D1.0"]
    assert _trim_to_event_end([0, 2, 1], vocab) == [0, 2]
    assert _trim_to_event_end([0, 1], vocab) == [0, 1]  # no duration token at all: unchanged


def test_default_seed_is_a_valid_chord_when_there_is_no_data():
    ids = default_seed(VOCAB)
    assert [VOCAB[i] for i in ids] == ["N48", "N52", "N55", "N60", "D1.0"]
    assert pick_seed(None, VOCAB) == ids
    assert default_seed(["N60", "D0.25"]) == [0, 1]  # a tiny vocabulary still works


def test_load_seeds_returns_none_when_the_data_is_missing(tmp_path):
    assert load_seeds(str(tmp_path)) is None


def test_pick_seed_ends_on_a_complete_chord():
    seeds = np.array([[VOCAB.index("N48"), VOCAB.index("D1.0"), VOCAB.index("N52")]])
    ids = pick_seed(seeds, VOCAB)
    assert VOCAB[ids[-1]].startswith("D")


def test_typed_notes_become_ids_and_unknown_notes_explain_the_range():
    assert [VOCAB[i] for i in primer_from_notes("C3 E3", VOCAB)] == ["N48", "D1.0", "N52", "D1.0"]
    with pytest.raises(ValueError, match="trained on notes from"):
        primer_from_notes("C7", VOCAB)


def test_checkpoint_round_trip_including_old_checkpoints(tmp_path, model):
    new, old = tmp_path / "new.pt", tmp_path / "old.pt"
    torch.save({"state": model.state_dict(), "vocab": VOCAB, "hidden": 32, "embed": 16, "layers": 2}, new)
    loaded, vocab = load_model(str(new))
    x = torch.tensor([[1, 2, 3]])
    assert vocab == VOCAB and torch.allclose(loaded(x)[0], model(x)[0])

    legacy = MusicLSTM(len(VOCAB), hidden=32).eval()  # default embed 64 and 2 layers, saved the way train.py used to
    torch.save({"state": legacy.state_dict(), "vocab": VOCAB, "hidden": 32}, old)
    loaded_old, _ = load_model(str(old))
    assert torch.allclose(loaded_old(x)[0], legacy(x)[0])


def test_the_network_size_is_read_from_the_weights(tmp_path):
    """A checkpoint with a different embedding size and 3 layers, saved WITHOUT embed/layers keys, still loads."""
    net = MusicLSTM(len(VOCAB), embed=24, hidden=40, layers=3).eval()
    path = tmp_path / "odd.pt"
    torch.save({"state": net.state_dict(), "vocab": VOCAB, "hidden": 40}, path)
    loaded, vocab = load_model(str(path))
    x = torch.tensor([[1, 2, 3]])
    assert vocab == VOCAB and torch.allclose(loaded(x)[0], net(x)[0])


def test_a_model_from_an_old_project_version_gets_a_clear_message(tmp_path):
    old_vocab = ["60.64.67", "62.65.69", "rest"]  # chord names instead of N60 / D0.5 tokens
    net = MusicLSTM(len(old_vocab), embed=96, hidden=32, layers=2)
    path = tmp_path / "old.pt"
    torch.save({"state": net.state_dict(), "vocab": old_vocab, "hidden": 32}, path)
    with pytest.raises(ValueError, match="older version of the project"):
        load_model(str(path))


def test_old_dataset_without_seeds_is_treated_as_no_data(tmp_path):
    np.savez(tmp_path / "dataset.npz", X=np.zeros((2, 3)), y=np.zeros(2))
    assert load_seeds(str(tmp_path)) is None