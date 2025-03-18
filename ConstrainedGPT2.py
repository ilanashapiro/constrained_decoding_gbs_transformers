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

class ConstrainedGPT2:
    def __init__(self, model, tokenizer):
        self.tokenizer = tokenizer
        self.eos_token = tokenizer.eos_token_id
        self.model = model

    def generate_unconstrained(self, hyp, n_best):
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
        input_ids = hyp.payload['input_values'].to(self.model.device)
        
        with torch.no_grad():
          outputs = self.model(input_ids) 
          logits = outputs.logits[:, -1, :]  # Extract last token logits
          log_probs = F.log_softmax(logits, dim=-1)

        top_probs, top_token_ids = torch.topk(log_probs, n_best, dim=-1)

        new_hyps = []
        for hyp_idx in range(n_best):
            new_token_id = top_token_ids[0, hyp_idx].unsqueeze(0)
            new_token = self.tokenizer.decode(new_token_id, skip_special_tokens=True)

            # **Explicitly update input sequence** (GPT-2 requires full sequence context)
            new_payload = copy.deepcopy(hyp.payload)
            new_payload['input_values'] = torch.cat([input_ids, new_token_id.unsqueeze(0)], dim=-1)
            
            # Compute the new hypothesis score
            if hyp.score != np.inf:
                next_score = hyp.score + top_probs[0, hyp_idx]
            else:
                # hyp.score is np.inf for the start hyp
                next_score = top_probs[0, hyp_idx]

            # Create a new hypothesis
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

        input_ids = hyp.payload['input_values'].to(self.model.device)

        with torch.no_grad():
            outputs = self.model(input_ids) 
            logits = outputs.logits[:, -1, :]  # Extract last token logits
            log_probs = F.log_softmax(logits, dim=-1)

        for idx in available_constraints:
            # Start a new constraint
            constraint_token_id = hyp.constraints[idx][0][0]  # 1st token of constraint
            constraint_token = self.tokenizer.decode(constraint_token_id, skip_special_tokens=True)  # convert ID to token

            new_payload = copy.deepcopy(hyp.payload)
            new_payload['input_values'] = torch.cat([input_ids, constraint_token_id.unsqueeze(0).unsqueeze(0)], dim=-1)

            if hyp.score != np.inf:
                next_score = hyp.score + log_probs[0, constraint_token_id]
            else:
                next_score = log_probs[0, constraint_token_id]

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
        constraint_row_index = hyp.constraint_index[0][0]
        constraint_tok_index = hyp.constraint_index[0][1] + 1  # go to next token
        constraint_index = (constraint_row_index, constraint_tok_index)

        continued_constraint_token_id = hyp.constraints[constraint_row_index][constraint_tok_index]
        continued_constraint_token = self.tokenizer.decode(continued_constraint_token_id, skip_special_tokens=True)

        input_ids = hyp.payload['input_values'].to(self.model.device)
      
        new_payload = copy.deepcopy(hyp.payload)
        new_payload['input_values'] = torch.cat([input_ids, continued_constraint_token_id.unsqueeze(0)], dim=-1)

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[:, -1, :]
            log_probs = F.log_softmax(logits, dim=-1)

        if hyp.score != np.inf:
            next_score = hyp.score + log_probs[0, continued_constraint_token_id]
        else:
            next_score = log_probs[0, continued_constraint_token_id]

        coverage = copy.deepcopy(hyp.coverage)
        coverage[constraint_row_index][constraint_tok_index] = 1 

        # see if there's more tokens in the constraint
        unfinished_constraint = constraint_tok_index + 1 < len(hyp.constraints[constraint_row_index])

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