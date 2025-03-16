# USING THIS FILE FOR REFERENCE: https://github.com/chrishokamp/constrained_decoding/blob/master/scripts/translate_with_constraints.py
# IT IS HEAVILY MODIFIED, WE JUST TOOK THE RELEVANT PARTS AND MODIFIED THEM FOR OUR USE CASE WITH GPT2

import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2Config
from Beam import Beam
from collections import defaultdict, OrderedDict
import copy
import numpy as np

from ConstraintHypothesis import ConstraintHypothesis, init_coverage
from ConstrainedDecoder import ConstrainedDecoder
from ConstrainedGPT2 import ConstrainedGPT2

def decode_input_gpt2(decoder, tokenizer, prompt, constraints, length_factor=1.3):
  input_values = tokenizer.encode(prompt, return_tensors="pt").squeeze(0)
  constraint_tokens = [tokenizer.encode(c, return_tensors="pt").squeeze(0) for c in constraints]
  
  coverage = init_coverage(constraint_tokens)
  payload = {
        "input_values": input_values
    }

  start_hyp = ConstraintHypothesis(
          token=None,
          score=-np.inf,
          coverage=coverage,
          constraints=constraint_tokens,
          payload=payload,
          backpointer=None,
          constraint_index=None,
          unfinished_constraint=False
      )

  search_grid = decoder.search(start_hyp=start_hyp, constraints=constraints,
                                 max_hyp_len=int(round(len(prompt) * length_factor)),
                                 beam_size=5)
  best_output = decoder.best_n(search_grid, tokenizer.eos_token_id, n_best=1)
  return best_output

config = GPT2Config.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
config.eos_token_id = tokenizer.eos_token_id # make sure EOS token is set correctly
model = ConstrainedGPT2(config, tokenizer).to("cuda")
decoder = ConstrainedDecoder(hyp_generation_func=model.generate,
                                 constraint_generation_func=model.generate_constrained,
                                 continue_constraint_func=model.continue_constrained,
                                 beam_implementation=Beam)
constraints = ["science is continuing", "technology"]
prompt = "Tell me a story about what's to come in education."
gen_text = decode_input_gpt2(decoder, tokenizer, prompt, constraints)

print("Generated Text:", gen_text)
