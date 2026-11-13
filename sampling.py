# Small helper for generation that needs neither PyTorch nor music21, so it is easy to test.


class NoRepeat:
    """Stops the model from writing the same run of `n` tokens twice in one piece (boring loops).

    Usage: ask `banned()` which tokens are not allowed next, pick a token, then call `add(token)`.
    n = 0 (or 1) switches it off.
    """

    def __init__(self, n, history=()):
        self.n = n
        self.history = []
        self._seen = {}  # the n-1 tokens before -> set of tokens that already followed them
        for token in history:
            self.add(token)

    def banned(self):
        if self.n < 2 or len(self.history) < self.n - 1:
            return ()
        return self._seen.get(tuple(self.history[-(self.n - 1):]), ())

    def add(self, token):
        if self.n >= 2 and len(self.history) >= self.n - 1:
            self._seen.setdefault(tuple(self.history[-(self.n - 1):]), set()).add(token)
        self.history.append(token)
