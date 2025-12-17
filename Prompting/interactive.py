import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
import sys

# --- CONFIGURATION ---
# Path to the folder where 'final_model' was saved
MODEL_PATH = '/workspace/Wiki-To-Go/Models/Checkpoints/final_model'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model():
    print(f"Loading model from {MODEL_PATH}...")
    print(f"Using device: {DEVICE}")
    
    try:
        # Load the model and tokenizer directly from the saved directory
        tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_PATH)
        model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
        
        # Move model to GPU if available
        model.to(DEVICE)
        model.eval() # Set to evaluation mode (disables dropout, etc.)
        
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def generate_response(model, tokenizer, prompt, max_length=100):
    # Encode input
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    
    # Generate
    # We use sampling (do_sample=True) to make the text creative/varied
    # top_k=50 restricts choices to the top 50 probable next tokens
    # temperature=0.7 controls randomness (lower = more focused, higher = more random)
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_length,
            do_sample=True,
            top_k=50,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode and remove the original prompt from the output if desired
    # Here we return the full sequence including the prompt
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Optional: If you only want the new text, slice it:
    # new_text = full_text[len(prompt):]
    
    return full_text

def main():
    model, tokenizer = load_model()
    
    print("\n" + "="*50)
    print(" GPT Interactive Terminal")
    print(" Type 'exit' or 'quit' to stop.")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input("You: ")
            
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            
            if not user_input.strip():
                continue

            print("GPT: ", end="", flush=True)
            
            # Generate response
            response = generate_response(model, tokenizer, user_input)
            print(response)
            print("-" * 20)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    main()