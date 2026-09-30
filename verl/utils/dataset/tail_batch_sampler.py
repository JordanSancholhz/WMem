"""Keep all epoch samples by merging a short tail into the last full batch."""


class TailMergeBatchSampler:
    """For 300 indices and batch_size=8, yield 36 batches of 8, then 12.

    The stateful iterator saves the shuffled order and cursor for exact
    mid-epoch resume, including the larger final batch.
    """

    def __init__(self, sampler, batch_size):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = False

    def __len__(self):
        size = len(self.sampler)
        return max(1, size // self.batch_size) if size else 0

    def __iter__(self):
        return _TailMergeIterator(self.sampler, self.batch_size)


class _TailMergeIterator:
    def __init__(self, sampler, batch_size):
        self.sampler = sampler
        self.batch_size = batch_size
        self.indices = list(sampler)
        if len(self.indices) != len(sampler):
            raise ValueError("Sampler length does not match its yielded indices")
        self.position = 0

    def __iter__(self):
        return self

    def __next__(self):
        remaining = len(self.indices) - self.position
        if remaining == 0:
            raise StopIteration
        size = remaining if remaining < 2 * self.batch_size else self.batch_size
        start = self.position
        self.position += size
        return self.indices[start:self.position]

    def state_dict(self):
        state = {
            "indices": self.indices[:],
            "position": self.position,
            "batch_size": self.batch_size,
        }
        generator = getattr(self.sampler, "generator", None)
        if generator is not None:
            state["generator_state"] = generator.get_state()
        if hasattr(self.sampler, "state_dict"):
            state["sampler_state"] = self.sampler.state_dict()
        return state

    def load_state_dict(self, state):
        if state["batch_size"] != self.batch_size or len(state["indices"]) != len(self.sampler):
            raise ValueError("Cannot resume with a different batch size or dataset length")
        position = state["position"]
        if not 0 <= position <= len(state["indices"]):
            raise ValueError("Invalid sampler checkpoint position")
        self.indices = list(state["indices"])
        self.position = position
        if "generator_state" in state:
            self.sampler.generator.set_state(state["generator_state"])
        if "sampler_state" in state:
            self.sampler.load_state_dict(state["sampler_state"])
