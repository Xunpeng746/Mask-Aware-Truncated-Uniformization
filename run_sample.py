import torch
import argparse
import random
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import os
import json
import warnings
import pandas as pd

from load_model import load_model
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
import sampling

warnings.filterwarnings("ignore")


def set_seed(seed=42, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def batch_entropy(token_ids, vocab_size):
    counts = torch.nn.functional.one_hot(token_ids, vocab_size).sum(dim=1)
    probs = counts / token_ids.size(1)
    nz = probs > 0
    return -torch.sum(probs * torch.log2(probs.where(nz, torch.ones_like(probs))), dim=1)


def main():
    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument("--model_path", default="louaaron/sedd-small", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--num", type=int, default=10)
    parser.add_argument("-p", "--predictor", type=str, default="matu", choices=["analytic", "matu", "euler", "none"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", default="./results/")
    parser.add_argument("--eval_model_path", default="gpt2", type=str)
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to run inference on")
    args = parser.parse_args()

    # Select GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using GPU {args.gpu}.")
    else:
        device = torch.device("cpu")
        print("CUDA not available. Falling back to CPU.")

    # Set random seed
    set_seed(args.seed)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save config
    config_path = os.path.join(args.save_dir, f'config_{args.steps}_{args.predictor}.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    model, graph, noise = load_model(args.model_path, device)
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    
    # Determine vocab size
    vocab_size = tokenizer.vocab_size + 1  # +1 for potential mask token

    # Create sampling function
    sampling_fn = sampling.get_pc_sampler(
        graph, noise, (args.batch_size, args.length), args.predictor, args.steps, device=device
    )

    # Generate samples
    print(f"Generating {args.num} samples with {args.predictor} predictor and {args.steps} steps...")
    samples = []
    num_batches = args.num // args.batch_size
    for n in tqdm(range(num_batches)):
        sample = sampling_fn(model)
        samples.append(sample)
    samples = torch.cat(samples, dim=0)

    # Evaluate samples
    res = []
    print("Evaluating samples...")
    with torch.no_grad():
        eval_model = GPT2LMHeadModel.from_pretrained(args.eval_model_path).to(device).eval()
        batches = samples.shape[0] // args.batch_size
        total_perplexitys = []
        total_entropys = []
        
        for i in tqdm(range(batches)):
            try:
                s = samples[i * args.batch_size:(i + 1) * args.batch_size]
                text_samples = tokenizer.batch_decode(s)
                
                # Compute perplexity
                outputs = eval_model(s, labels=s)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[1]
                logits = logits.transpose(-1, -2)
                perplexitys = F.cross_entropy(logits[..., :-1], s[..., 1:], reduction="none").mean(dim=-1).exp()
                
                # Compute entropy
                entropys = batch_entropy(s, vocab_size)
                
                total_perplexitys.append(perplexitys.mean().item())
                total_entropys.append(entropys.mean().item())
                
                for j in range(s.shape[0]):
                    res.append({
                        'text': text_samples[j],
                        'perplexity': perplexitys[j].item(),
                        'entropy': entropys[j].item()
                    })
            except Exception as e:
                print(f"Error evaluating sample {i}: {e}")
                continue
        
        total_perplexity = sum(total_perplexitys) / len(total_perplexitys)
        total_entropy = sum(total_entropys) / len(total_entropys)
        
        print(f"Generative Perplexity: {total_perplexity:.3f}")
        print(f"Generative Entropy: {total_entropy:.3f}")

    # Save results
    res_df = pd.DataFrame(res)
    print("\nResults Summary:")
    print(res_df.describe())
    
    csv_path = os.path.join(args.save_dir, f'res_{args.steps}_{args.predictor}.csv')
    res_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    
    # Update config with results
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
        config_dict['perplexity'] = total_perplexity
        config_dict['entropy'] = total_entropy
    
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=4)
    print(f"Config with metrics saved to {config_path}")


if __name__=="__main__":
    main()