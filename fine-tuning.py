
import torch, os
from transformers import (
    GPT2Tokenizer, 
    GPT2LMHeadModel, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling, 
    TextDataset
)
from transformers import LogitsProcessorList, LogitsProcessor
dir_path = os.path.dirname(os.path.realpath(__file__))

# Load Tokenizer & Fix Padding Issue
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set PAD token

# Load Pretrained GPT-2 Model
model = GPT2LMHeadModel.from_pretrained(model_name)

#  Load Dataset
data_path = dir_path+"/datasets/scraped_chekhov.txt"

def tokenize_function(text):
    return tokenizer(text, truncation=True, padding="max_length", max_length=512)

train_dataset = TextDataset(
    file_path=data_path,
    tokenizer=tokenizer,
    block_size=512  # GPT-2 context window
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

#  Fine-Tune GPT-2 Model
training_args = TrainingArguments(
    output_dir=dir_path+"/gpt2_chekhov",
    evaluation_strategy="no",  # No eval set since story generation is subjective
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    num_train_epochs=3,
    logging_steps=200,
    learning_rate=5e-5,
    warmup_steps=500,
    weight_decay=0.01,
    fp16=True,  # Mixed precision for speedup
    push_to_hub=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

trainer.train()
model.save_pretrained(dir_path+"/fine_tuned_gpt2")
tokenizer.save_pretrained(dir_path+"/fine_tuned_gpt2")
