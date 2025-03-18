# USING THIS FILE FOR REFERENCE: https://github.com/chrishokamp/constrained_decoding/blob/master/scripts/translate_with_constraints.py
# IT IS HEAVILY MODIFIED, WE JUST TOOK THE RELEVANT PARTS AND MODIFIED THEM FOR OUR USE CASE WITH GPT2

import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2Config
from Beam import Beam
from collections import defaultdict, OrderedDict
import copy, sys
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from ConstraintHypothesis import ConstraintHypothesis, init_coverage
from ConstrainedDecoder import ConstrainedDecoder
from ConstrainedGPT2 import ConstrainedGPT2

def decode_input_gpt2(decoder, tokenizer, prompt, constraints):
  input_values = tokenizer.encode(prompt, return_tensors="pt")
  constraint_tokens = [tokenizer.encode(c, return_tensors="pt") for c in constraints]

  coverage = init_coverage(constraint_tokens)
  payload = {
        "input_values": input_values
    }

  start_hyp = ConstraintHypothesis(
          token='',
          score=-np.inf,
          coverage=coverage,
          constraints=constraint_tokens,
          payload=payload,
          backpointer=None,
          constraint_index=None,
          unfinished_constraint=False
      )

  search_grid = decoder.search(start_hyp=start_hyp, 
                               constraints=constraint_tokens,
                               eos_token=tokenizer.eos_token,
                               max_hyp_len=15,
                               beam_size=5)
  best_output = decoder.best_n(search_grid, tokenizer.eos_token_id, n_best=1)[0]
  return ''.join(best_output)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
model = ConstrainedGPT2(gpt2, tokenizer)
decoder = ConstrainedDecoder(hyp_generation_func=model.generate_unconstrained,
                                 constraint_generation_func=model.generate_constrained,
                                 continue_constraint_func=model.continue_constrained,
                                 beam_implementation=Beam)
constraints = ["princess", "deer ran", "castle"]
prompt = "Once upon a time"
gen_text = decode_input_gpt2(decoder, tokenizer, prompt, constraints)

print("Prompt:", prompt)
print("Generated Text:", gen_text)