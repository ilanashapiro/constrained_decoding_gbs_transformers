import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load model and tokenizer
model_name = "gpt2"  # You can use "gpt2-medium", "gpt2-large", etc.
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()  # Set the model to evaluation mode

# Initialize the prompt
prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Generation parameters
max_new_tokens = 20  # Adjust as needed
temperature = 1.0  # Sampling temperature
top_k = 5  # Top k candidates to sample

# Generate token-by-token
for _ in range(max_new_tokens):
    with torch.no_grad():
        outputs = model(input_ids)  # Forward pass
        logits = outputs.logits[:, -1, :]  # Get logits for the last token

        # Apply temperature and softmax to get probabilities
        probs = torch.nn.functional.softmax(logits / temperature, dim=-1)

        # Get the top 5 token candidates based on their probability scores
        top_probs, top_token_ids = torch.topk(probs, top_k, dim=-1)
        print("TOP PROBS", top_probs)

        # Print the new_text for each of the top 5 candidates
        for i in range(top_k):
            next_token_id = top_token_ids[0, i].unsqueeze(0)  # Get the i-th token

            # Make sure next_token_id is a 2D tensor before concatenation
            new_input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)

            # Decode and print for each of the top 5 candidates
            new_text = tokenizer.decode(new_input_ids[0], skip_special_tokens=True)
            print(f"Candidate {i + 1}: {new_text} (Score: {top_probs[0, i].item():.4f})")

        # Optionally, you can stop at a specific candidate's choice or continue
        # If you want to pick the highest probability candidate for the next iteration:
        next_token_id = top_token_ids[0, 0].unsqueeze(0)  # Select the top candidate
        input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)

        # Stop if EOS token is generated
        if next_token_id.item() == tokenizer.eos_token_id:
            break
