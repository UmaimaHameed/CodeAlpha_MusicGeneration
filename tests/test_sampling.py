import random

from sampling import NoRepeat


def test_nothing_is_banned_at_first_and_when_switched_off():
    guard = NoRepeat(4)
    assert guard.banned() == ()
    off = NoRepeat(0, [1, 2, 3, 1, 2, 3])
    assert off.banned() == ()


def test_repeating_a_run_is_banned():
    guard = NoRepeat(3, [1, 2, 3, 1, 2])  # the run 1,2,3 already happened, so after 1,2 the token 3 is banned
    assert guard.banned() == {3}
    guard.add(4)
    assert guard.banned() == ()  # context (2, 4) has not been seen


def test_seed_tokens_count_as_history():
    guard = NoRepeat(3, [7, 8, 9, 7, 8])
    assert 9 in guard.banned()


def test_a_random_walk_with_the_guard_never_repeats_a_run():
    rng, guard, seq = random.Random(0), NoRepeat(4), []
    for _ in range(300):
        allowed = [t for t in range(5) if t not in guard.banned()]
        if not allowed:
            break
        token = rng.choice(allowed)
        guard.add(token)
        seq.append(token)
    grams = [tuple(seq[i:i + 4]) for i in range(len(seq) - 3)]
    assert len(seq) > 100
    assert len(grams) == len(set(grams))


def test_without_the_guard_the_same_walk_does_repeat():
    rng, seq = random.Random(0), []
    for _ in range(300):
        seq.append(rng.choice(range(5)))
    grams = [tuple(seq[i:i + 4]) for i in range(len(seq) - 3)]
    assert len(grams) > len(set(grams))
