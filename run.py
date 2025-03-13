import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from collections import defaultdict, OrderedDict
import copy
import numpy as np

from ConstraintHypothesis import ConstraintHypothesis, init_coverage
from ConstrainedDecoder import ConstrainedDecoder

def decode_input_gpt2(model, tokenizer, source, constraints, length_factor=1.3, beam_size=5):
  device = model.device

  # Tokenize input source
  source_tokens = tokenizer.encode(source, return_tensors="pt").to(device)

  # Prepare constraints (convert words/phrases to tokenized form)
  constraint_tokens = [tokenizer.encode(c, add_special_tokens=False) for c in constraints]

  # Set max generation length
  max_length = int(round(len(source_tokens[0]) * length_factor))

  coverage = [np.zeros(len(c), dtype='int16') for c in constraints]

  # Initialize beam search
  start_hyp = ConstraintHypothesis(
          token=None,
          score=None,
          coverage=coverage,
          constraints=constraints,
          payload={},
          backpointer=None,
          constraint_index=None,
          unfinished_constraint=False
      )

        
  search_grid = decoder.search(start_hyp=start_hyp, 
                                constraints=constraint_tokens,
                                max_hyp_len=max_length,
                                beam_size=beam_size)

  best_output = decoder.best_n(search_grid, tokenizer.eos_token_id, n_best=1)

  # Decode generated tokens back to text
  return tokenizer.decode(best_output[0], skip_special_tokens=True)


# Example Usage
constraints = [[tokenizer.encode("science")[0]], [tokenizer.encode("technology")[0]]]
coverage = init_coverage(constraints)
decoder = ConstrainedDecoder(TODO)

generated_text = decoder.generate("The future of", constraints, max_length=20)
print("Generated Text:", generated_text)
