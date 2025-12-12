from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
import os

# --- CONFIGURATION ---
DATA_FILE = '/home/max-tost/Dokumente/Wiki-To-Go/Data/Debugging_Data/wiki_data_debug.txt'
SAVE_PATH = '/home/max-tost/Dokumente/Wiki-To-Go/Models/Tokenizer/wiki_tokenizer.json'
VOCAB_SIZE = 32000  # Standard size (GPT-2 was 50k, Llama is 32k)

# Ensure directory exists
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

print("Initializing Tokenizer...")
# 1. Initialize a BPE model
tokenizer = Tokenizer(models.BPE())

# 2. Pre-tokenization
# Before BPE happens, we split by whitespace to avoid merging words across spaces
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

# 3. Decoding
# How to turn IDs back to text
tokenizer.decoder = decoders.ByteLevel()

# 4. Trainer
# We add special tokens that might be useful later (optional but good practice)
trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=["<|endoftext|>", "<|padding|>"],
    show_progress=True
)

print(f"Training on {DATA_FILE}...")
# 5. Train
tokenizer.train(files=[DATA_FILE], trainer=trainer)

print("Training finished.")

# 6. Save
tokenizer.save(SAVE_PATH)
print(f"Tokenizer saved to {SAVE_PATH}")

# 7. Quick Test
test_sentence = "Hello, this is a wiki tokenizer test."
encoded = tokenizer.encode(test_sentence)
print(f"\n--- Test ---")
print(f"Input: {test_sentence}")
print(f"Tokens: {encoded.tokens}")
print(f"IDs:   {encoded.ids}")