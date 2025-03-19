# THIS CODE WAS COPIED AND PASTED FROM THIS FILE FROM THE ORIGINAL PAPER
# https://github.com/chrishokamp/constrained_decoding/blob/master/constrained_decoding/__init__.py

import numpy as np
import sys

# Utility Functions
def init_coverage(constraints):
    coverage = []
    for c in constraints:
        coverage.append(np.zeros(len(c[0]), dtype='int16'))
    return coverage

class ConstraintHypothesis:
    """A (partial) hypothesis which maintains an additional constraint coverage object

    Args:
        token (unicode): the surface form of this hypothesis
        score (float): the score of this hypothesis (higher is better)
        coverage (list of lists): a representation of the area of the constraints covered by this hypothesis
        constraints (list of lists): the constraints that may be used with this hypothesis
        payload (:obj:): additional data that comes with this hypothesis. Functions may
            require certain data to be present in the payload, such as the previous states, glimpses, etc...
        backpointer (:obj:`ConstraintHypothesis`): a pointer to the hypothesis object which generated this one
        constraint_index (tuple): if this hyp is part of a constraint, the index into `self.constraints` which
            is covered by this hyp `(constraint_idx, token_idx)`
        unfinished_constraint (bool): a flag which indicates whether this hyp is inside an unfinished constraint

    """

    def __init__(self, token, score, coverage, constraints, payload=None, backpointer=None,
                 constraint_index=None, unfinished_constraint=False):
        self.token = token
        self.score = score

        assert len(coverage) == len(constraints), 'constraints and coverage length must match'
        assert all(len(cov) == len(cons[0]) for cov, cons in zip(coverage, constraints)), \
            'each coverage and constraint vector must match'

        self.coverage = coverage
        self.constraints = constraints
        self.backpointer = backpointer
        self.payload = payload
        self.constraint_index = constraint_index
        self.unfinished_constraint = unfinished_constraint

    def __str__(self):
        return u'token: {}, sequence: {}, score: {}, coverage: {}, constraints: {},'.format(
            self.token, self.sequence, self.score, self.coverage, self.constraints)

    def __getitem__(self, key):
        return getattr(self, key)

    @property
    def sequence(self):
        sequence = []
        current_hyp = self
        while current_hyp.backpointer is not None:
            sequence.append(current_hyp.token)
            current_hyp = current_hyp.backpointer
        sequence.append(current_hyp.token)
        return sequence[::-1]

    @property
    def constraint_indices(self):
        """Return the (start, end) indexes of the constraints covered by this hypothesis"""

        # we know a hyp covered a constraint token if a coverage index changed from 0-->1
        # we also know which constraint and which token was covered by looking at the indices in the coverage
        # constraint tracker sequence = [None, (constraint_index, constraint_token_index), ...]
        # for each hyp:
        #     compare current_hyp.coverage with current_hyp.backpointer.coverage:
        #         if there is a difference:
        #             add difference to constraint tracker sequence
        #         else:
        #             add None to constraint tracker sequence

        def _compare_constraint_coverage(coverage_one, coverage_two):
            for constraint_idx, (this_coverage_row,
                                 prev_coverage_row) in enumerate(zip(coverage_one,
                                                                     coverage_two)):

                for token_idx, (this_coverage_bool,
                                prev_coverage_bool) in enumerate(zip(this_coverage_row, prev_coverage_row)):
                    if this_coverage_bool != prev_coverage_bool:
                        return constraint_idx, token_idx

            return None

        constraint_tracker_sequence = []
        current_hyp = self
        while current_hyp.backpointer is not None:
            # compare current_hyp.coverage with previous hyp's coverage
            constraint_tracker_sequence.append(_compare_constraint_coverage(current_hyp.coverage,
                                                                            current_hyp.backpointer.coverage))
            current_hyp = current_hyp.backpointer

        # we need to check if the first hyp covered a constraint
        start_coverage = init_coverage(current_hyp.constraints)
        constraint_tracker_sequence.append(_compare_constraint_coverage(current_hyp.coverage,
                                                                        start_coverage))

        # finally reverse this sequence to put it in order
        return constraint_tracker_sequence[::-1]

    @property
    def alignments(self):
        current_hyp = self
        if current_hyp.payload.get('alignments', None) is not None:
            alignment_weights = []
            while current_hyp.backpointer is not None:
                alignment_weights.append(current_hyp.payload['alignments'])
                current_hyp = current_hyp.backpointer
            return np.squeeze(np.array(alignment_weights[::-1]), axis=1)
        else:
            return None

    def constraint_candidates(self):
        available_constraints = []
        for idx in range(len(self.coverage)):
            if self.coverage[idx][0] == 0:
                available_constraints.append(idx)

        return available_constraints