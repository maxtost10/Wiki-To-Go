# Tokenization: The Bridge Between Language and Math

## Why do we need a Tokenizer?
Neural networks are mathematical functions—they perform matrix multiplications on numbers, not text. We cannot simply feed the string "Wiki to go" into a transformer. We need to translate it into a sequence of integers.

There are three main approaches to this translation, but only one works well for Large Language Models (LLMs):

1.  **Character Level:** Map 'a' -> 1, 'b' -> 2.
    *   *Problem:* Sequences become incredibly long. A simple sentence like "The quick brown fox" becomes ~20 tokens. The model struggles to remember context over long distances.
2.  **Word Level:** Map "apple" -> 502, "run" -> 503.
    *   *Problem:* The vocabulary becomes massive (millions of words). We lose the relationship between "run", "running", and "runner". Unknown words cause errors.
3.  **Subword Level (BPE - Byte Pair Encoding):** The "Goldilocks" solution.
    *   Common words ("the", "and") become single tokens.
    *   Rare words ("antidisestablishmentarianism") are broken into meaningful chunks ("anti", "dis", "establish"...).
    *   *Benefit:* Efficient sequence length + reasonable vocabulary size (~32k-50k).

## Learning by Doing: The Toy Tokenizer
To understand the "Black Box" of tokenization, we implemented the **Byte Pair Encoding (BPE)** algorithm from scratch in Python.

### The Algorithm
1.  **Initialize:** Start with a vocabulary of 256 individual bytes (ASCII/UTF-8 characters).
2.  **Count:** Scan the text and count the frequency of every adjacent pair of tokens (e.g., `('e', 'r')`).
3.  **Merge:** Find the most frequent pair and merge it into a new, single token (e.g., `er` -> ID 257).
4.  **Repeat:** Keep merging until the vocabulary reaches a target size.

### Key Lessons Learned
We experimented with the number of merges on a small Wikipedia dataset and discovered the **Memorization vs. Generalization** trade-off.

*   **The "Overfitting" Experiment:**
    We allowed the tokenizer to run for too many merges on a small text.
    *   *Result:* The tokenizer eventually merged words into phrases, and phrases into sentences, until the **entire text became a single token**.
    *   *Why this is bad:* A model trained with this would only recognize that specific text block. If shown a new sentence, it would fail completely because it has no tokens for individual words anymore.

*   **The "Sweet Spot":**
    We learned that we need to stop merging when the vocabulary reaches a standard size (typically **32,000 to 50,000** tokens). This balances compression (short sequences) with generalization (reusable subwords).

### Production Results (Rust Implementation)
For the full 27GB dataset, our Python script was too slow. We switched to the Hugging Face `tokenizers` library (Rust backend) for the final run.

**Run Statistics:**
*   **Input:** 35.8 Million text chunks.
*   **Vocabulary Size:** 32,000 tokens.
*   **Training Time:** ~10 minutes (vs hours/days with pure Python).

**Inference Test:**
We tested the tokenizer on a new sentence to verify subword splitting:

> **Input:** "Hello, this is a wiki tokenizer test."  
> **Tokens:** `['Hello', ',', 'Ġthis', 'Ġis', 'Ġa', 'Ġwiki', 'Ġto', 'ken', 'izer', 'Ġtest', '.']`

*   **Observation 1:** The `Ġ` symbol represents a preceding space.
*   **Observation 2:** Common words like "wiki" are single tokens.
*   **Observation 3:** "tokenizer" was split into `to` + `ken` + `izer`, proving the BPE successfully generalizes to complex words using known sub-parts.