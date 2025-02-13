import torch
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from architecture import GPT2
import tiktoken
from config import GPTConfig

torch.serialization.add_safe_globals(['GPTConfig'])

enc = tiktoken.get_encoding("gpt2")
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

def load_model(checkpoint_path, device):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device , weights_only=False)
    
    # Initialize model with saved config
    model = GPT2(checkpoint['config'])
    model.to(device)
    
    # Load the state dict
    model.load_state_dict(checkpoint['model'])
    
    # Put model in eval mode
    model.eval()
    
    return model


def generate_text(model, prompt, max_tokens=200, temperature=0.8, top_k=50, device="cpu"):
    model.eval()
    
    # Encode the prompt
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0)  # Add batch dimension
    x = tokens.to(device)
    
    # Set up random generator
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42)
    
    with torch.no_grad():
        while x.size(1) < len(tokens) + max_tokens:
            # Forward pass
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, _ = model(x)
            
            # Get logits of the last token and apply temperature
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k sampling
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
            
            # Sample from the top-k distribution
            idx_next = torch.gather(
                topk_indices, 
                -1, 
                torch.multinomial(topk_probs, 1, generator=sample_rng)
            )
            
            # Append to the sequence
            x = torch.cat((x, idx_next), dim=1)
    
    # Decode the generated sequence
    generated_tokens = x[0].tolist()
    generated_text = enc.decode(generated_tokens)
    
    return generated_text

# Load the model
model = load_model("final_gpt2_124m.pt", device) #put the trained model path

# Generate some text
# Generate some text and save to file
prompts = [
    "Once upon a time in a distant galaxy,",
    "The secret to happiness is",
    "In the year 2050, artificial intelligence has",
]

# Open file in write mode
with open('generated_samples.txt', 'w', encoding='utf-8') as f:
    f.write("Generated Samples:\n")
    f.write("-" * 50 + "\n")
    
    for prompt in prompts:
        generated = generate_text(
            model,
            prompt=prompt,
            max_tokens=200,
            temperature=0.8,
            device=device
        )
        
        # Write to both console and file
        print(f"\nPrompt: {prompt}")
        print(f"Generated text:\n{generated}")
        print("-" * 50)
        
        f.write(f"\nPrompt: {prompt}\n")
        f.write(f"Generated text:\n{generated}\n")
        f.write("-" * 50 + "\n")

print("\nSamples have been saved to 'generated_samples.txt'")