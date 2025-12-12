import torch
from torch.utils.data import DataLoader, IterableDataset
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
import wandb
import os

# --- CONFIGURATION ---
TOKENIZER_PATH = '/workspace/Wiki-To-Go/Models/Tokenizer/wiki_tokenizer.json'
DATA_FILE = '/workspace/Data/wiki_data_cleaned.txt'
SAVE_DIR = '/workspace/Wiki-To-Go/Models/Checkpoints'

# GPT Architecture (GPT-Small sized for speed, fits easily in 24GB VRAM)
# To make it "GPT-3 style", we just ensure we train from scratch with these mechanics.
MODEL_CONFIG = GPT2Config(
    vocab_size=32000,      # Matches your tokenizer
    n_positions=1024,      # Context window
    n_embd=768,            # Embedding dimension
    n_layer=12,            # Number of layers
    n_head=12,             # Attention heads
    activation_function="gelu_new", # GPT-3 uses GeLU
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
)

BATCH_SIZE = 8          # Adjust based on VRAM usage
GRAD_ACCUMULATION = 4   # Effective batch size = 32
LEARNING_RATE = 6e-4    # Standard for this scale

# --- 1. DATASET (Streaming) ---
class WikiIterableDataset(IterableDataset):
    def __init__(self, file_path, tokenizer, block_size=1024):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __iter__(self):
        # We read the file line by line, tokenize, and yield chunks
        # This is memory efficient and handles the 80GB file easily
        with open(self.file_path, 'r', encoding='utf-8') as f:
            buffer = []
            for line in f:
                # Encode line
                tokens = self.tokenizer.encode(line)
                buffer.extend(tokens)
                
                # When buffer is full enough, yield chunks
                while len(buffer) >= self.block_size:
                    chunk = buffer[:self.block_size]
                    buffer = buffer[self.block_size:]
                    
                    # GPT expects inputs and labels. 
                    # HF Model handles the shifting internally, so labels = input_ids
                    yield {
                        "input_ids": torch.tensor(chunk, dtype=torch.long),
                        "labels": torch.tensor(chunk, dtype=torch.long)
                    }

# --- 2. LIGHTNING MODULE ---
class GPTLightningModule(L.LightningModule):
    def __init__(self, config, lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = GPT2LMHeadModel(config) # Initializes random weights (Pre-training)
        
    def forward(self, input_ids, labels=None):
        return self.model(input_ids=input_ids, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], labels=batch["labels"])
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        # Standard AdamW optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer
    
    # --- AUTO-REGRESSIVE INFERENCE ---
    def generate(self, prompt, tokenizer, max_new_tokens=50):
        self.eval() # Switch to eval mode
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # This is the wrapper for the autoregressive loop
        with torch.no_grad():
            outputs = self.model.generate(
                inputs, 
                max_new_tokens=max_new_tokens, 
                do_sample=True, 
                top_k=50, 
                temperature=0.7
            )
        return tokenizer.decode(outputs[0])

# --- 3. MAIN TRAINING LOOP ---
if __name__ == "__main__":
    # 1. Login to W&B (You will be prompted for API key on first run)
    wandb.login()
    wandb_logger = WandbLogger(project="Wiki-To-Go", name="GPT-Small-Run1")

    # 2. Prepare Tokenizer & Data
    print("Loading Tokenizer...")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_PATH)
    tokenizer.add_special_tokens({'pad_token': '<|padding|>'})
    
    print("Initializing Dataset...")
    dataset = WikiIterableDataset(DATA_FILE, tokenizer, block_size=MODEL_CONFIG.n_positions)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)

    # 3. Initialize Model
    print("Initializing GPT Model...")
    model = GPTLightningModule(MODEL_CONFIG, LEARNING_RATE)

    # 4. Trainer Setup
    trainer = L.Trainer(
        logger=wandb_logger,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",  # Use FP16 for massive speedup on RTX 3090
        max_epochs=1,          # 1 Epoch is usually enough for huge data
        accumulate_grad_batches=GRAD_ACCUMULATION,
        log_every_n_steps=50,
        default_root_dir=SAVE_DIR
    )

    # 5. Start Training
    print("Starting Training...")
    trainer.fit(model, dataloader)
    
    # 6. Save Final Model
    print("Saving model...")
    model.model.save_pretrained(f"{SAVE_DIR}/final_model")
    tokenizer.save_pretrained(f"{SAVE_DIR}/final_model")
    
    # 7. Extended Inference Test
    print("\n" + "="*40)
    print("--- EXTENDED INFERENCE TEST ---")
    print("="*40)
    
    test_prompts = [
        "The history of science is",
        "The capital of France is",
        "Physics is the natural science that",
        "The Roman Empire was",
        "Artificial intelligence is",
        "William Shakespeare was a",
        "The Earth revolves around",
        "In mathematics, a function is",
        "The first World War began in",
        "Computer programming is"
    ]

    # Run inference for each prompt
    # Note: We manually move model to GPU for inference if trainer didn't leave it there
    model.to("cuda") 
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n[Test {i+1}] Prompt: {prompt}")
        try:
            generated_text = model.generate(prompt, tokenizer)
            # Print just the new part to keep it clean (optional, here we print full)
            print(f"Generated: {generated_text}")
        except Exception as e:
            print(f"Error generating: {e}")

    print("\n" + "="*40)
    print("Done!")