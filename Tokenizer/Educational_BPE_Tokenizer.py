# %% [markdown]
# # Understanding Byte Pair Encoding (BPE) from Scratch
# 
# In this notebook, we will build a BPE tokenizer manually.
# The goal is to understand how a machine goes from "raw text" to "integers" 
# and how it learns to group characters into meaningful subwords.
#
# ## The Core Idea
# 1. Start with a vocabulary of just individual bytes (256 options).
# 2. Look at the text and find the most common pair of adjacent tokens (e.g., "e" and "r").
# 3. "Merge" them into a new token "er".
# 4. Repeat until you reach your desired vocabulary size.

# %%
# Configuration
DATA_PATH = '/home/max-tost/Dokumente/Wiki-To-Go/Data/Debugging_Data/wiki_data_debug.txt'

# Let's read the first 10,000 characters of your debug data to train our toy tokenizer
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    text_data = f.read(10000)

print(f"Loaded {len(text_data)} characters.")
print(f"Sample: {text_data[:100]}...")

# %% [markdown]
# ## Step 1: Input to Integers
# Computers don't see characters, they see bytes. 
# We convert our text into a list of integers (UTF-8 bytes).
# This is our initial state: every byte is a token.

# %%
# Convert string to list of integers (UTF-8)
tokens = list(text_data.encode("utf-8"))

print(f"Original text length: {len(text_data)}")
print(f"Token list length: {len(tokens)}")
print(f"First 10 tokens: {tokens[:10]}")
# Example: 'A' is 65, 'a' is 97 in ASCII/UTF-8

# %% [markdown]
# ## Step 2: Helper Functions
# We need two main functions:
# 1. `get_stats`: Counts which pair of tokens appears most often together.
# 2. `merge`: Replaces that pair with a new, single token ID.

# %%
def get_stats(ids):
    """
    Given a list of integers (ids), count how often each pair of adjacent integers occurs.
    Returns a dictionary {(id1, id2): count}
    """
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    """
    In the list of integers (ids), replace all occurrences of `pair` with `idx`.
    """
    newids = []
    i = 0
    while i < len(ids):
        # Check if we are at the specific pair, but ensure we don't go out of bounds
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

# Test the functions
test_ids = [1, 2, 3, 1, 2]
print(f"Test stats for {test_ids}: {get_stats(test_ids)}") 
# Should see (1, 2): 2

# %% [markdown]
# ## Step 3: The Training Loop
# Now we iterate!
# We start our new token IDs at 256 (since 0-255 are taken by standard bytes).
# We will do 20 merges just to see it work.

# %%
vocab_size = 276 # 256 initial bytes + 20 new merges
num_merges = vocab_size - 256
ids = list(tokens) # Working copy

merges = {} # To store our learned rules: (p0, p1) -> new_idx

print("--- Starting Training ---")
for i in range(num_merges):
    # 1. Find most common pair
    stats = get_stats(ids)
    if not stats:
        break
        
    pair = max(stats, key=stats.get)
    
    # 2. Assign it a new ID
    idx = 256 + i
    
    # 3. Replace in data
    ids = merge(ids, pair, idx)
    
    # Save the rule
    merges[pair] = idx
    
    # Decode the pair to see what we actually learned (for visualization)
    # Note: This simple decode only works if the pair components are simple bytes
    # In a real scenario, this is recursive.
    print(f"Merge {i+1}/{num_merges}: Replaced pair {pair} with new token {idx}. Occurrences: {stats[pair]}")

print("--- Training Done ---")
print(f"Original length: {len(tokens)}")
print(f"Compressed length: {len(ids)}")
print(f"Compression ratio: {len(tokens) / len(ids):.2f}X")

# %% [markdown]
# ## Step 4: Tokenizer in Action
# Let's see how our new tokenizer encodes a new string.
# It applies the merge rules in the exact same order they were learned.

# %%
def encode(text):
    # 1. Start with raw bytes
    tokens = list(text.encode("utf-8"))
    
    # 2. Apply merges strictly in order
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        # Find the pair in our text that has the lowest index in our 'merges' dictionary
        # (This means it was learned earliest, so it has priority)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        
        # If this pair isn't in our learned rules, we are done
        if pair not in merges:
            break
            
        # Merge
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens

def decode(ids):
    # Reverse the process roughly requires keeping a vocabulary map
    # Since we built 'merges' {(p0, p1): idx}, we can reconstruct the vocabulary
    # For this toy example, we'll just show the IDs.
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]
    
    tokens = b"".join(vocab[idx] for idx in ids)
    return tokens.decode("utf-8", errors="replace")

# Test
test_sentence = "the" # 'th' and 'e' are very common in English
encoded = encode(test_sentence)
decoded = decode(encoded)

print(f"Input: '{test_sentence}'")
print(f"IDs: {encoded}")
print(f"Decoded: '{decoded}'")

# Use this notebook to play around and change the number of merges!