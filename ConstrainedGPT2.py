# THIS CODE WAS MODELED AFTER AND HEAVILY MODIFIED BASED ON THIS FILE FROM THE ORIGINAL PAPER
# https://github.com/chrishokamp/constrained_decoding/blob/master/constrained_decoding/translation_model/nmt.py
# BASICALLY WE TOOK THIS CODE FOR THE NEURAL TRANSLATION MACHINE AND REVISED IT TO WORK FOR GPT2/TRANSFORMER ARCHITECTURE

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from collections import defaultdict, OrderedDict
import copy
import numpy as np

from ConstraintHypothesis import ConstraintHypothesis, init_coverage
from ConstrainedDecoder import ConstrainedDecoder

class ConstrainedGPT2(GPT2LMHeadModel):
    def __init__(self, config, tokenizer):
        super().__init__(config)
        self.tokenizer = tokenizer
        self.eos_token = tokenizer.eos_token_id

    def generate(self, hyp, n_best):
        """
        Generates `n_best` next-token hypotheses using GPT-2 with constraint-aware beam search.
        """

        # if EOS token is generated, continue only with the same hypothesis
        if hyp.token == self.eos_token:
            return [ConstraintHypothesis(
                token=self.eos_token,
                score=hyp.score,
                coverage=copy.deepcopy(hyp.coverage),
                constraints=hyp.constraints,
                payload=hyp.payload,
                backpointer=hyp,
                constraint_index=None,
                unfinished_constraint=False
            )]

        # tokenized input seq
        input_ids = torch.tensor(hyp.payload['input_values']).unsqueeze(0).to(self.device)

        # GPT2 forward pass
        with torch.no_grad():
            outputs = self.forward(input_ids) 
            logits = outputs.logits[:, -1, :] # logits for the final token

        log_probs = F.log_softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        # Get top n_best tokens
        n_best_outputs = np.argsort(log_probs)[-n_best:][::-1]
        chosen_costs = log_probs[n_best_outputs]

        new_hyps = []
        for hyp_idx in range(n_best):
            new_token_id = int(n_best_outputs[hyp_idx])
            new_token = self.tokenizer.decode([new_token_id])

            # Update payload with new token
            new_payload = copy.deepcopy(hyp.payload)
            new_payload['input_values'].append(new_token_id)

            next_score = (hyp.score + chosen_costs[hyp_idx]) if hyp.score is not None else chosen_costs[hyp_idx]

            new_hyp = ConstraintHypothesis(
                token=new_token,
                score=next_score,
                coverage=copy.deepcopy(hyp.coverage),
                constraints=hyp.constraints,
                payload=new_payload,
                backpointer=hyp,
                constraint_index=None,
                unfinished_constraint=False
            )

            new_hyps.append(new_hyp)

        return new_hyps

    def generate_constrained(self, hyp):
        """
        Generates constrained hypotheses by enforcing constraints that have not yet been covered.
        """

        assert not hyp.unfinished_constraint, "hyp must not be part of an unfinished constraint"

        new_constraint_hyps = []
        available_constraints = hyp.constraint_candidates()

        input_ids = torch.tensor(hyp.payload['input_values']).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.forward(input_ids)
            logits = outputs.logits[:, -1, :]

        log_probs = F.log_softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        for idx in available_constraints:
            # Start a new constraint
            constraint_token_id = hyp.constraints[idx][0]  # 1st token of constraint
            constraint_token = self.tokenizer.decode([constraint_token_id])  # convert ID to token

            # append constraint token to input seq
            new_input_values = hyp.payload['input_values'] + [constraint_token_id]

            new_payload = copy.deepcopy(hyp.payload)
            new_payload['input_values'] = new_input_values  # update input seq

            if hyp.score is not None:
                next_score = hyp.score + log_probs[constraint_token_id]
            else:
                next_score = log_probs[constraint_token_id]

            # updates constraint coverage
            coverage = copy.deepcopy(hyp.coverage)
            coverage[idx][0] = 1  # mark 1st token as covered

            unfinished_constraint = len(coverage[idx]) > 1

            new_hyp = ConstraintHypothesis(
                token=constraint_token,
                score=next_score,
                coverage=coverage,
                constraints=hyp.constraints,
                payload=new_payload,
                backpointer=hyp,
                constraint_index=(idx, 0),
                unfinished_constraint=unfinished_constraint
            )

            new_constraint_hyps.append(new_hyp)

        return new_constraint_hyps

    def continue_constrained(self, hyp):
        """
        Continues an unfinished constraint by adding the next required token.
        """

        assert hyp.unfinished_constraint, "hyp must be part of an unfinished constraint"

        # det next token in the constraint seq
        constraint_row_index = hyp.constraint_index[0]
        constraint_tok_index = hyp.constraint_index[1] + 1  # go to next token
        constraint_index = (constraint_row_index, constraint_tok_index)

        continued_constraint_token_id = hyp.constraints[constraint_row_index][constraint_tok_index]
        continued_constraint_token = self.tokenizer.decode([continued_constraint_token_id])

        new_input_values = hyp.payload['input_values'] + [continued_constraint_token_id] # append constraint token to input seq

        input_ids = torch.tensor(new_input_values).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.forward(input_ids)
            logits = outputs.logits[:, -1, :]

        log_probs = F.log_softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        if hyp.score is not None:
            next_score = hyp.score + log_probs[continued_constraint_token_id]
        else:
            next_score = log_probs[continued_constraint_token_id]

        coverage = copy.deepcopy(hyp.coverage)
        coverage[constraint_row_index][constraint_tok_index] = 1 

        # see if there's more tokens in the constraint
        unfinished_constraint = constraint_tok_index + 1 < len(hyp.constraints[constraint_row_index])

        new_payload = copy.deepcopy(hyp.payload)
        new_payload['input_values'] = new_input_values

        new_hyp = ConstraintHypothesis(
            token=continued_constraint_token,
            score=next_score,
            coverage=coverage,
            constraints=hyp.constraints,
            payload=new_payload,
            backpointer=hyp,
            constraint_index=constraint_index,
            unfinished_constraint=unfinished_constraint
        )

        return new_hyp