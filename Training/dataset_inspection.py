# %% [markdown]
# # Inspecting Data & "Automatic" Target Creation
# This notebook visualizes how raw text becomes training data, and how Hugging Face models
# automatically handle the "next token prediction" logic.

# %%
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Config, PreTrainedTokenizerFast

# --- CONFIGURATION ---
TOKENIZER_PATH = '/workspace/Wiki-To-Go/Models/Tokenizer/wiki_tokenizer.json'

# %% [markdown]
# ## 1. Load Tokenizer & Prepare Data
# We create a dummy sentence to represent a "Chunk" from your text file.

# %%
print("Loading Tokenizer...")
# If you haven't trained it yet, this might fail. 
# Fallback: using a standard GPT2 tokenizer just for demo if needed.
try:
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_PATH)
except:
    print("Custom tokenizer not found. Using standard GPT2 for demonstration.")
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# A sample "Chunk" (Context Window)
text = "WikiToGo is the best project ever"
encoded = tokenizer.encode(text, return_tensors="pt")

print(f"Text: '{text}'")
print(f"IDs:  {encoded[0].tolist()}")

# %% [markdown]
# ## 2. The "Lazy" Dataset Approach
# In your training script, the dataset yields this:
# `{'input_ids': chunk, 'labels': chunk}`
#
# Wait, why are they identical? Shouldn't the label be shifted? 
# Let's look at what we feed the model.

# %%
inputs = encoded
labels = encoded.clone()  # Exact copy!

print("What we pass to the model:")
print(f"Input IDs: {inputs}")
print(f"Labels:    {labels}")

# %% [markdown]
# ## 3. The "Black Box" (Hugging Face Magic)
# We initialize a tiny random GPT model and feed it these identical tensors.
# Does it crash? No. It calculates a Loss automatically.

# %%
# Initialize a tiny baby GPT-2 (Random weights)
config = GPT2Config(
    vocab_size=tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 50257,
    n_layer=2, 
    n_head=2, 
    n_embd=128
)
model = GPT2LMHeadModel(config)

# Forward pass
output = model(input_ids=inputs, labels=labels)

print("--- Model Output ---")
print(f"Loss: {output.loss.item()}")
print(f"Logits Shape: {output.logits.shape}") 
# Shape is [1, Sequence_Length, Vocab_Size]

# %% [markdown]
# ## 4. The "White Box": How did it know?
# You asked: *"How does the model see that automatically?"*
#
# **Answer:** It's hardcoded inside the `forward()` method of `GPT2LMHeadModel`.
#
# When you provide `labels`, the model internally does Python slicing to align them.
# Let's **manually recreate the loss** to prove exactly what is happening under the hood.

# %%
# 1. Get the raw predictions for everything (Logits)
logits = output.logits

# 2. PERFORM THE SHIFT (This is the logic inside the model)
# We remove the LAST prediction, because we have no label for the token AFTER the sequence ends.
shift_logits = logits[..., :-1, :].contiguous()

# We remove the FIRST label, because we don't predict the first word (we are given it).
shift_labels = labels[..., 1:].contiguous()

print(f"Shifted Logits (Predictions) Shape: {shift_logits.shape}")
print(f"Shifted Labels (Targets) Shape:     {shift_labels.shape}")

# 3. Calculate Cross Entropy manually
# Flatten the batches to make it a long list of predictions
loss_fct = torch.nn.CrossEntropyLoss()
manual_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

print("\n--- COMPARISON ---")
print(f"Model's Automatic Loss: {output.loss.item()}")
print(f"Our Manual Shift Loss:  {manual_loss.item()}")

if torch.isclose(output.loss, manual_loss):
    print("\n✅ MATCH! This proves the model internally shifts indices by 1.")
else:
    print("\n❌ Mismatch (Something went wrong).")

# %% [markdown]
# ## 5. Visualizing the Shift
# This table shows exactly which input token is used to predict which label.

# %%
input_tokens = tokenizer.convert_ids_to_tokens(inputs[0])
label_tokens = tokenizer.convert_ids_to_tokens(labels[0])

print(f"{'INPUT (Given this)':<20} | {'TARGET (Predict this)':<20}")
print("-" * 45)

for i in range(len(input_tokens) - 1):
    # Input at index `i` predicts Label at index `i+1`
    curr_input = input_tokens[i]
    curr_target = input_tokens[i+1] # effectively labels[i+1]
    
    print(f"{curr_input:<20} | {curr_target:<20}")

print("-" * 45)
print(f"(The last input '{input_tokens[-1]}' is ignored for loss because there is no target)")