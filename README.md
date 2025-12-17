# 🧠 Wiki-To-Go: A Pocket-Sized GPT Journey

> *"The limits of my language mean the limits of my world."* — Ludwig Wittgenstein

Large Language Models often feel like magic—black boxes that live in the cloud and know everything. But what happens when you decide to build one yourself? 

**Wiki-To-Go** is an educational journey to demystify that magic. We didn't just fine-tune an existing model; we started from the raw atoms of knowledge: the compressed Wikipedia XML dumps. We built the parser, trained the tokenizer, and successfully trained a GPT-Small model on the sum of human knowledge (well, the English parts of it).

This project proves that with a bit of patience (and a good GPU), you can compress the world into a mathematical function.

---

## 📂 Repository Structure

Here is how the project is organized. We have separated the pipeline into three distinct logical stages:

```text
/workspace
├── Data/
│   ├── enwiki-latest-pages-articles-multistream.xml.bz2  # The massive source
│   └── wiki_data_cleaned.txt                             # The 27GB clean text file
│
└── Wiki-To-Go/
    ├── Models/
    │   ├── Checkpoints/       # Saved training checkpoints
    │   └── Tokenizer/         # The trained BPE tokenizer.json
    │
    ├── Parse_raw_Wiki/        # 1. THE CLEANER
    │   ├── build_wiki_dataset.py
    │   └── README.md          # <-- Details on parsing logic here
    │
    ├── Tokenizer/             # 2. THE TRANSLATOR
    │   ├── Educational_BPE_Tokenizer.py  # A "from scratch" python implementation
    │   ├── Production_Tokenizer.py       # The fast Rust-backend implementation
    │   └── README.md          # <-- Deep dive into BPE theory here
    │
    └── Training/              # 3. THE BRAIN
        ├── train_gpt.py       # The PyTorch Lightning training loop
        ├── Train_loss.png
        └── GPU_Utilization.png
```

---

## 🛠️ The Pipeline

### 1. Parsing the World
Before a model can learn, it must read. We processed the 25GB compressed Wikipedia dump, stripping away HTML, tables, and markup to produce clean, linear text.
*   **Detailed Documentation:** [Read about the Parser logic](./Wiki-To-Go/Parse_raw_Wiki/README.md)

### 2. Tokenization (BPE)
A neural network doesn't understand "Apple"; it understands numbers. We implemented Byte Pair Encoding (BPE) to create a vocabulary of 32,000 subword tokens, balancing efficiency and generalization.
*   **Detailed Documentation:** [Read about the Tokenizer & Theory](./Wiki-To-Go/Tokenizer/README.md)

---

## 📉 Training Results

We trained a **GPT-Small** architecture (12 layers, 12 heads, 768 embedding dim) from scratch.

### The Learning Curve
The model successfully converged, learning the structure of language, grammar, and facts from the dataset.
![Training Loss](./Wiki-To-Go/Training/Train_loss.png)

### The "Efficiency" Incident
Training took approximately **100 hours** on an RTX 3090. 

You might notice the flat line of maximum effort below. However, there is a funny story behind the duration. I set `num_workers=2` in the dataloader, assuming that "two workers are better than one!" 

Evidently, the computer interpreted this as *"Great, let's do the exact same work twice!"* Instead of halving the data loading time, the overhead essentially doubled the processing required. Lesson learned: concurrency is hard, and sometimes machines take instructions a little *too* literally.