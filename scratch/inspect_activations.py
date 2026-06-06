import torch
import os

path = "d:/Projects/ACT-Break/outputs/gemma-3-1b-it/activations/activations.pt"
if not os.path.exists(path):
    print(f"Error: Path {path} does not exist.")
else:
    data = torch.load(path, map_location="cpu")
    print(f"Keys in saved data: {data.keys()}")
    
    prompts = data["prompts"]
    refusal_responses = data["refusal_responses"]
    
    print(f"Total prompts collected: {len(prompts)}")
    print(f"Total refusal responses: {len(refusal_responses)}")
    
    print("\n--- Inspecting first 15 samples ---")
    for i in range(min(15, len(prompts))):
        p = prompts[i].encode('ascii', errors='replace').decode('ascii')
        r = refusal_responses[i].encode('ascii', errors='replace').decode('ascii').replace('\n', ' ')
        print(f"Sample {i+1}:")
        print(f"  Prompt: {p[:120]}")
        print(f"  Refusal: {r[:120]}")
        print("-" * 50)
