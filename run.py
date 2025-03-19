# USING THIS FILE FOR REFERENCE: https://github.com/chrishokamp/constrained_decoding/blob/master/scripts/translate_with_constraints.py
# IT IS HEAVILY MODIFIED, WE JUST TOOK THE RELEVANT PARTS AND MODIFIED THEM FOR OUR USE CASE WITH GPT2

import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2Config
from Beam import Beam
from collections import defaultdict, OrderedDict
import copy, sys, os
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from ConstraintHypothesis import ConstraintHypothesis, init_coverage
from ConstrainedDecoder import ConstrainedDecoder
from ConstrainedGPT2 import ConstrainedGPT2

def load_model(model_path=os.getcwd() + "/fine_tuned_gpt2"):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    return model, tokenizer

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
                               max_hyp_len=20,
                               beam_size=5)
  best_output = decoder.best_n(search_grid, tokenizer.eos_token_id, n_best=1)[0]
  return ''.join(best_output)

use_finetuned = False
if use_finetuned:
  gpt2, tokenizer = load_model()
else:
  tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
  gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")

model = ConstrainedGPT2(gpt2, tokenizer)


decoder = ConstrainedDecoder(hyp_generation_func=model.generate_unconstrained,
                                 constraint_generation_func=model.generate_constrained,
                                 continue_constraint_func=model.continue_constrained,
                                 beam_implementation=Beam)

# prompt = "He had not seen the hussars all that day, but had heard about them from an infantry officer"
# constraints = ['The peril of an', 'I could not see', 'Crunchers attention was here']
# constraints = ['The peril', 'Crunchers attention']

# Prompt: He had not seen the hussars all that day, but had heard about them from an infantry officer
# Generated Text: Crunchers attention was hereI could not see the hussars all that day, but had heard 
#                 about them from an infantry officerI could not see the hussars all that day, but had 
#                 seen them from an officerThe peril of an


constraints = ["princess ", "deer ran "]
prompt = "Once upon a time"
gen_text = decode_input_gpt2(decoder, tokenizer, prompt, constraints)

print("Prompt:", prompt)
print("Generated Text:", gen_text)