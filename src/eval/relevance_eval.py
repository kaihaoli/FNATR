"""
E-commerce Search Relevance Evaluation Script
This script evaluates the relevance of products for e-commerce search queries using a pre-trained model.

Requirements:
- Python 3.8+
- PyTorch
- Transformers >= 4.51.0 (pip install --upgrade "transformers>=4.51.0")
- CUDA-capable GPU(s) recommended for faster processing

License: MIT License
Copyright (c) 2025 kaihaoli
"""

import os
import sys
import json
import argparse
import logging
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import time

def print_gpu_memory():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            cached = torch.cuda.memory_reserved(i) / 1024**3  # GB
            props = torch.cuda.get_device_properties(i)
            total = props.total_memory / 1024**3  # GB
            usage_pct = (allocated / (props.total_memory / 1024**3)) * 100
            print(f"GPU {i}: {allocated:.2f}GB/{total:.1f}GB ({usage_pct:.1f}%) allocated, {cached:.2f}GB cached")

def format_instruction(instruction, query, doc):
    if instruction is None:
        instruction = 'Given a ecommerce search query, retrieve relevant products that meet the query'
    output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction,query=query, doc=doc)
    return output

def process_inputs(pairs):
    inputs = tokenizer(
        pairs, padding=False, truncation='longest_first',
        return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
    )
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    
    # With DataParallel, the model is on CUDA and we just need to move inputs to CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for key in inputs:
        inputs[key] = inputs[key].to(device)
    return inputs

@torch.no_grad()
def compute_logits(inputs, **kwargs):
    batch_scores = model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores[:, 1].exp().tolist()
    return scores

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='E-commerce Search Relevance Evaluation')
    parser.add_argument('--model_path', type=str, default='Qwen/Qwen3-Reranker-8B',
                      help='Path to the pre-trained model (default: Qwen3-Reranker-8B)')
    parser.add_argument('--input_file', type=str, required=True,
                      help='''Input JSONL file containing queries and passages. Each line should be a JSON object with format:
                      {"query": "user search query", "passage": "product description"}
                      Example: {"query": "red running shoes", "passage": "Nike Air Zoom men's running shoes in crimson red"}''')
    parser.add_argument('--output_file', type=str, required=True,
                      help='''Output JSONL file to write results. Will contain the input data plus relevance scores.
                      Output format: {"query": "...", "passage": "...", "rel_score": 0.95}
                      The rel_score is a float between 0 and 1, where higher values indicate better relevance.''')
    parser.add_argument('--batch_size', type=int, default=256,
                      help='Batch size for processing')
    parser.add_argument('--max_length', type=int, default=8192,
                      help='Maximum sequence length')
    parser.add_argument('--fp16', action='store_true',
                      help='Use FP16 for inference')
    return parser.parse_args()

args = parse_args()

# Initialize model and tokenizer
logger.info(f"Loading model from {args.model_path}")
tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side='left')

# Load model with DataParallel for better batch-level GPU utilization
model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    torch_dtype=torch.float16 if args.fp16 else torch.float32,
    # Remove device_map to allow DataParallel to work properly
).eval()

# Use DataParallel for batch-level parallelism across all GPUs
if torch.cuda.device_count() > 1:
    print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
    model = torch.nn.DataParallel(model)
    model = model.cuda()
else:
    model = model.cuda()

print(f"Model loaded for parallel inference on {torch.cuda.device_count()} GPUs")

token_false_id = tokenizer.convert_tokens_to_ids("no")
token_true_id = tokenizer.convert_tokens_to_ids("yes")
max_length = 8192

#prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
prefix = """<|im_start|>system
You are an e-commerce search relevance judge. Given a customer's search query and a product description, determine if the product is relevant to what the customer is looking for.

Consider:
- Product category match
- Key features alignment  
- Customer intent satisfaction

Answer only "yes" (relevant) or "no" (not relevant).<|im_end|>
<|im_start|>user
"""
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
        
import json

# Read data from JSONL file
logger.info(f"Reading input file: {args.input_file}")
data = []
try:
    with open(args.input_file, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
except FileNotFoundError:
    logger.error(f"Input file not found: {args.input_file}")
    sys.exit(1)
except json.JSONDecodeError:
    logger.error(f"Invalid JSON in input file: {args.input_file}")
    sys.exit(1)

logger.info(f"Loaded {len(data)} records from {args.input_file}")

# Extract queries and passages
queries = [item['query'] for item in data]
passages = [item['passage'] for item in data]

#task = 'Given a ecommerce search query, retrieve relevant products that meet the query'
task = 'Judge if this product is relevant to the customer search query for e-commerce'

# Increased batch size for better multi-GPU utilization with DataParallel
# DataParallel will automatically split this batch across available GPUs
batch_size = 256  
all_scores = []

print(f"Processing {len(queries)} query-passage pairs with batch size {batch_size}")
print(f"Each GPU will process approximately {batch_size // torch.cuda.device_count()} samples per batch")
print_gpu_memory()

start_time = time.time()

for i in range(0, len(queries), batch_size):
    batch_start = time.time()
    batch_queries = queries[i:i + batch_size]
    batch_passages = passages[i:i + batch_size]
    
    pairs = [format_instruction(task, query, doc) for query, doc in zip(batch_queries, batch_passages)]
    
    # Tokenize the input texts
    inputs = process_inputs(pairs)
    batch_scores = compute_logits(inputs)
    all_scores.extend(batch_scores)
    
    batch_time = time.time() - batch_start
    total_time = time.time() - start_time
    avg_time_per_batch = total_time / ((i // batch_size) + 1)
    remaining_batches = (len(queries) - i - len(batch_queries)) // batch_size
    eta = remaining_batches * avg_time_per_batch
    
    throughput = len(batch_queries) / batch_time
    
    print(f"Batch {i//batch_size + 1}/{(len(queries) + batch_size - 1)//batch_size} - "
          f"Size: {len(batch_queries)} - Time: {batch_time:.2f}s - "
          f"Throughput: {throughput:.1f} samples/s - ETA: {eta:.1f}s")
    
    # Clear cache to prevent memory accumulation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

total_time = time.time() - start_time
total_throughput = len(queries) / total_time

print(f"\n=== Processing Complete ===")
print(f"Total processing time: {total_time:.2f}s")
print(f"Average time per sample: {total_time/len(queries):.3f}s")
print(f"Total throughput: {total_throughput:.1f} samples/s")
print_gpu_memory()

# Add rel_score to each record
for i, item in enumerate(data):
    item['rel_score'] = all_scores[i]

# Save the updated data
with open(args.output_file, 'w') as f:
    for item in data:
        f.write(json.dumps(item) + '\n')

logger.info(f"Saved results with rel_scores to {args.output_file}")
def main():
    args = parse_args()
    
    try:
        # Process queries and calculate scores
        process_queries()
        
        # Save results
        logger.info(f"Saving results to {args.output_file}")
        with open(args.output_file, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        
        logger.info(f"Sample scores: {all_scores[:5]}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
"""
python relevance_eval.py \
    --input_file input.jsonl \
    --output_file output.jsonl
"""