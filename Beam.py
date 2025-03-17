# THIS CODE WAS COPIED AND PASTED FROM THIS FILE FROM THE ORIGINAL PAPER
# https://github.com/chrishokamp/constrained_decoding/blob/master/constrained_decoding/__init__.py

from sortedcontainers import SortedListWithKey

class Beam:
    def __init__(self, size, eos_token, lower_better=True):
        self.hypotheses = SortedListWithKey(key=lambda x: -x.score if lower_better else x.score)
        self.size = size
        self.eos_token = eos_token

    def add(self, hyp, beam_constraints=[]):
        if all(check(hyp, set([self.eos_token])) for check in beam_constraints):
            self.hypotheses.add(hyp)
            if len(self.hypotheses) > self.size:
                assert len(self.hypotheses) == self.size + 1
                del self.hypotheses[-1]  # Remove lowest-scoring hypothesis
        else:
            print("FAILED TO ADD HYP", hyp)

    def __len__(self):
        return len(self.hypotheses)

    def __iter__(self):
        for hyp in self.hypotheses:
            yield hyp