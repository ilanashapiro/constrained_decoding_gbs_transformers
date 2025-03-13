# THIS CODE WAS COPIED AND PASTED FROM THIS FILE FROM THE ORIGINAL PAPER
# https://github.com/chrishokamp/constrained_decoding/blob/master/constrained_decoding/__init__.py

from collections import OrderedDict
from Beam import Beam

class ConstrainedDecoder:
    def __init__(self, hyp_generation_func, constraint_generation_func, continue_constraint_func, payload, beam_implementation=Beam):
        self.hyp_generation_func = hyp_generation_func
        self.constraint_generation_func = constraint_generation_func
        self.continue_constraint_func = continue_constraint_func
        self.beam_implementation = beam_implementation
        self.payload=payload
        self.beam_constraints = [self.eos_covers_constraints]

    def eos_covers_constraints(self, hyp, eos=set(['<eos>', u'</S>'])):
        """
        Return False if hyp.token is <eos>, and hyp does not cover all constraints, True otherwise

        """
        constraints_remaining = True
        coverage = hyp.coverage
        if sum(covered for cons in coverage for covered in cons) == sum(len(c) for c in coverage):
            constraints_remaining = False
        is_eos = False
        if hyp.token in eos:
            is_eos = True

        if constraints_remaining and is_eos:
            return False
        return True

    def search(self, start_hyp, constraints, max_hyp_len=50, beam_size=10):
        # the total number of constraint tokens determines the height of the grid
        grid_height = sum(len(c) for c in constraints)

        search_grid = OrderedDict()
        search_grid[(0, 0)] = Beam(size=beam_size)
        search_grid[(0, 0)].add(start_hyp)

        for i in range(1, max_hyp_len + 1):
            j_start = max(i - (max_hyp_len - grid_height), 0)
            j_end = min(i, grid_height) + 1
            for j in range(j_start, j_end):
                # create the new beam
                new_beam = self.beam_implementation(size=beam_size)
                # generate hyps from (i-1, j-1), and (i-1, j), and add them to the beam
                # cell to the left generates
                if (i-1, j) in search_grid:
                    generation_hyps = self.get_generation_hyps(search_grid[(i-1, j)], beam_size)
                    for hyp in generation_hyps:
                        new_beam.add(hyp, beam_constraints=self.beam_constraints)
                # lower left diagonal cell adds hyps from constraints
                if (i-1, j-1) in search_grid:
                    new_constraint_hyps = self.get_new_constraint_hyps(search_grid[(i-1, j-1)])
                    continued_constraint_hyps = self.get_continued_constraint_hyps(search_grid[(i-1, j-1)])
                    for hyp in new_constraint_hyps:
                        new_beam.add(hyp, beam_constraints=self.beam_constraints)
                    for hyp in continued_constraint_hyps:
                        new_beam.add(hyp, beam_constraints=self.beam_constraints)

                search_grid[(i,j)] = new_beam

        return search_grid

    def get_generation_hyps(self, beam, beam_size=1):
        """return all hyps which are continuations of the hyps on this beam

        hyp_generation_func maps `(hyp) --> continuations`
          - the coverage vector of the parent hyp is not modified in each child
        """

        continuations = (self.hyp_generation_func(hyp, beam_size) for hyp in beam if not hyp.unfinished_constraint)

        # flatten
        return (new_hyp for hyp_list in continuations for new_hyp in hyp_list)

    def get_new_constraint_hyps(self, beam):
        """return all hyps which start a new constraint from the hyps on this beam

        constraint_hyp_func maps `(hyp) --> continuations`
          - the coverage vector of the parent hyp is modified in each child
        """

        continuations = (self.constraint_generation_func(hyp)
                         for hyp in beam if not hyp.unfinished_constraint)

        # flatten
        return (new_hyp for hyp_list in continuations for new_hyp in hyp_list)

    def get_continued_constraint_hyps(self, beam):
        """return all hyps which continue the unfinished constraints on this beam

        constraint_hyp_func maps `(hyp, constraints) --> forced_continuations`
          - the coverage vector of the parent hyp is modified in each child
        """
        continuations = (self.continue_constraint_func(hyp)
                         for hyp in beam if hyp.unfinished_constraint)

        return continuations
    
    @staticmethod
    def best_n(search_grid, eos_token, n_best=1, cut_off_eos=True, return_model_scores=False, return_alignments=False,
               length_normalization=True, prefer_eos=False):
        top_row = max(k[1] for k in search_grid.keys())

        if top_row > 0:
            output_beams = [search_grid[k] for k in search_grid.keys() if k[1] == top_row]
        else:
            # constraints seq is empty
            # Note this is a very hackish way to get the last beam
            output_beams = [search_grid[search_grid.keys()[-1]]]

        output_hyps = [h for beam in output_beams for h in beam]

        # if at least one hyp ends with eos, drop all the ones that don't (note this makes some big assumptions)
        # IDEA: just take the hyps that were _continued_
        if prefer_eos:
            eos_hyps = [h for h in output_hyps if eos_token in h.sequence]
            if len(eos_hyps) > 0:
               output_hyps = eos_hyps

        # getting the true length of each hypothesis
        true_lens = [h.sequence.index(eos_token) if eos_token in h.sequence else len(h.sequence)
                     for h in output_hyps]
        true_lens = [float(l) for l in true_lens]
        # hack to let us keep true_len info after sorting
        for h, true_len in zip(output_hyps, true_lens):
            h.true_len = true_len

        # normalizing scores by true_len is optional -- Note: length norm param could also be weighted as in GNMT paper
        try:
            if length_normalization:
                output_seqs = [(h.sequence, h.score / true_len, h) for h, true_len in zip(output_hyps, true_lens)]
            else:
                output_seqs = [(h.sequence, h.score, h) for h in output_hyps]
        except:
            # Note: this happens when there is actually no output, just a None
            output_seqs = [([eos_token], float('inf'), None)]

        if cut_off_eos:
            output_seqs = [(seq[:int(t_len)], score, h) for (seq, score, h), t_len in zip(output_seqs, true_lens)]

        # sort by score, lower is better (i.e. cost semantics)
        output_seqs = sorted(output_seqs, key=lambda x: x[1])
        if return_alignments:
            assert output_hyps[0].alignments is not None, 'Cannot return alignments if they are not part of hypothesis payloads'
            # we subtract 1 from true len index because the starting `None` token is not included in the `h.alignments`
            alignments = [h.alignments[:int(h.true_len-1)] for seq, score, h in output_seqs]

        if return_model_scores:
            output_seqs = [(seq, score, h.payload['model_scores'] / true_len)
                           for (seq, score, h), true_len in zip(output_seqs, true_lens)]
        else:
            output_seqs = [(seq, score, h) for seq, score, h in output_seqs]

        if return_alignments:
            if n_best > 1:
                return output_seqs[:n_best], alignments[:n_best]
            else:
                return output_seqs[0], alignments[:1]
        else:
            if n_best > 1:
                return output_seqs[:n_best]
            else:
                # Note in this case we don't return a list
                return output_seqs[0]