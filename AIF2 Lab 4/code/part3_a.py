import re
import json
import random
import torch
import pandas as pd
import time
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from hotpot_eval import update_answer 
from pyserini.search.lucene import LuceneSearcher

# Update the code so we index the questions and the associated answer from wiki_snippets_dev.json as a document for the rag
# add the directory names in and then test it
# ============================================
# Configuration
# ============================================
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DEV_DATA_PATH = "/home/akadur/AIF2Lab1StarterCode/data/dev.jsonl"
TOKEN = "hf_token"
WIKI_INDEX_PATH = "/home/akadur/AIF2Lab3/bm25_wiki_index/"

MAX_NEW_TOKENS = 50
TEMPERATURE = 0.1
TOP_P = 1.0
BATCH_SIZE = 5
SAMPLE_SIZE = 550 # Using 550 based on your Part 1 sample size
N_PASSAGES_RANGE = range(1, 11) # n=1 to 10 passages for this part

# ============================================
# Model Setup (Reused from Part 2)
# ============================================
def initialize_model(model_name: str, token: str):
    """Initializes the LLM and tokenizer."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", token=token)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=token
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer, device


# ============================================
# Data & Utility Handling (Reused/Modified)
# ============================================
def load_sample_data(path: str, sample_size: int):
    """Loads a random sample of data from a JSONL file."""
    print(f"Loading {sample_size} sample data from: {path}")
    df = pd.read_json(path, lines=True)
    # Using a fixed random state for reproducibility within this experiment
    selected = df.sample(sample_size, random_state=42) 
    return list(selected["question"]), list(selected["answer"]) 

def evaluate(predictions, references):
    """Computes EM and F1 metrics using the hotpot_eval utility."""
    metrics = {'em': 0.0, 'f1': 0.0, 'prec': 0.0, 'recall': 0.0}
    # Ensure predictions are strings (empty string if generation failed)
    valid_pairs = [(p, r) for p, r in zip(predictions, references) if p is not None and r is not None]
    
    for pred, ref in valid_pairs:
        update_answer(metrics, pred, ref) 
        
    total = len(valid_pairs)
    if total > 0:
        for k in metrics:
            metrics[k] /= total
    
    return metrics

def call_model_with_retry(model, tokenizer, inputs, max_retries=3):
    """Helper function for generation with exponential backoff."""
    for attempt in range(max_retries):
        try:
            with torch.no_grad():
                return model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    # Stop generation when the model produces the EOT token
                    eos_token_id=tokenizer.convert_tokens_to_ids("<|eot_id|>")
                )
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                # Do not log retries as errors in the console
                time.sleep(wait_time)
            else:
                raise

def generate_responses(model, tokenizer, device, prompts):
    """Generates responses for a list of prompts in batches."""
    responses = []
    
    for i in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[i:i + BATCH_SIZE]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        prompt_len = inputs["input_ids"].shape[1] 

        try:
            output = call_model_with_retry(model, tokenizer, inputs)
        except Exception as e:
            print(f"Fatal error generating batch starting at index {i}. Skipping batch.")
            responses.extend([""] * len(batch))
            continue
            
        for seq in output:
            # Decode the generated part only
            text = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True).strip()
            
            # Post-process: Llama 3.1 often ends with <|eot_id|> or its token, 
            # but sometimes generates artifacts. Clean up anything after the first newline 
            # if we are expecting a concise answer.
            text = text.split('\n')[0].strip()
            responses.append(text)

        if (i + BATCH_SIZE) % 50 == 0:
            print(f"Processed {min(i + BATCH_SIZE, len(prompts))}/{len(prompts)} examples.")

    return responses


# ============================================
# NEW: Wikipedia RAG Components for Part 3(a)
# ============================================

def retrieve_wiki_passages(searcher: LuceneSearcher, query: str, n: int) -> List[str]:
    """
    Retrieves the top n Wikipedia passages using the query.
    
    Note: For a document-level Wikipedia index, the 'contents' field holds the text.
    The document title is often in the 'title' field.
    """
    if n == 0:
        return []

    hits = searcher.search(query, k=n)
    passages = []

    for hit in hits:
        try:
            # We assume the index stores documents with a 'contents' field for the text
            doc = searcher.doc(hit.docid)
            
            # The structure depends on the specific Pyserini index used.
            # For the default LlamaWiki structure, doc.contents() holds the text.
            # We will use doc.raw() and parse the JSON since it's most robust
            
            # Fallback to hit.text if the index is simplified
            if hasattr(hit, 'text') and hit.text:
                 text = hit.text
            else:
                 # Attempt to parse the raw JSON data if available
                 raw_doc = doc.raw()
                 rec = json.loads(raw_doc)
                 # Typically 'text' or 'contents' holds the main passage content
                 text = rec.get('text', rec.get('contents', '')) 
            
            passages.append(text.strip())
            
        except Exception as e:
            print(f"Warning: Failed to retrieve or parse document for hit {hit.docid}. Error: {e}")
            continue
       
    return passages

def format_rag_prompt(question: str, passages: List[str]) -> str:
    """
    Formats the instruct-style RAG prompt (Context Before Question).
    This format is required for Part 3(a).
    """
    # Create the context block
    context_lines = []
    for i, p in enumerate(passages):
        # We include the passage number for clarity
        context_lines.append(f"Passage {i+1}: {p}")
        
    context_block = "\n---\n".join(context_lines)
    
    # Use the Llama 3.1 instruct template
    prompt = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful and accurate question-answering assistant. 
Use ONLY the provided Wikipedia passages below to answer the user's question concisely. 
Do not use any external knowledge. If the passages do not contain the answer, state that you cannot find the answer.
<|eot_id|>

<|start_header_id|>user<|end_header_id|>
CONTEXT:
{context_block}

QUESTION: {question}

Please provide a concise answer based ONLY on the CONTEXT.
<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
"""
    return prompt.strip()


# ============================================
# Experiment Runner for Part 3(a)
# ============================================
def run_rag_experiment(model, tokenizer, device, dev_questions, dev_answers):
    """
    Runs the RAG experiment for n=1 to 10 passages.
    """
    # 1. Initialize the Wikipedia Searcher
    try:
        # Use the provided Pyserini path to the Wikipedia index
        searcher = LuceneSearcher(WIKI_INDEX_PATH)
        if not searcher:
            raise Exception("Wikipedia Searcher could not be initialized.")
        print(f"Wikipedia Searcher initialized using index at: {WIKI_INDEX_PATH}")
    except Exception as e:
        print(f"ERROR: Could not initialize LuceneSearcher for Wikipedia. Check WIKI_INDEX_PATH. Details: {e}")
        return {} # Return empty results

    results = {}

    for n in N_PASSAGES_RANGE:
        print(f"\n--- Running RAG Experiment (Context Before Question) with n={n} passages ---")
        
        prompts = []
        
        # 2. Generate prompts for all dev questions using retrieval
        for q in dev_questions:
            # Retrieve 'n' passages from Wikipedia using the dev question as the query
            passages = retrieve_wiki_passages(searcher, q, n=n)
            
            # Build the RAG Llama-formatted prompt
            prompt = format_rag_prompt(q, passages)
            prompts.append(prompt)
        #count = 0
        #while count < 5:
        #    print(f"random prompts:\n {random.sample(prompts, 1)}\n")    
        # 3. Generate predictions
        preds = generate_responses(model, tokenizer, device, prompts)
        
        # 4. Evaluate
        scores = evaluate(preds, dev_answers)
        results[n] = {'em': scores['em'], 'f1': scores['f1']}
        print(f"n={n} (EM={scores['em']:.4f}, F1={scores['f1']:.4f})")

    return results

# ============================================
# Main
# ============================================
if __name__ == "__main__":
    
    # 1. Initialize Model
    try:
        model, tokenizer, device = initialize_model(MODEL_NAME, TOKEN)
    except Exception as e:
        print(f"ERROR: Could not initialize model. Details: {e}")
        exit()

    # 2. Load Data
    try:
        dev_questions, dev_answers = load_sample_data(DEV_DATA_PATH, SAMPLE_SIZE)
        if not dev_questions:
             print("ERROR: Dev set is empty.")
             exit()
    except Exception as e:
        print(f"ERROR: Could not load dev data from {DEV_DATA_PATH}. Details: {e}")
        exit()
    
    # 3. Run Experiment
    print("\nRunning Part 3 (a): Wikipedia RAG (n=1 to 10)...")
    rag_results = run_rag_experiment(
        model, tokenizer, device, 
        dev_questions, dev_answers
    )

    # 4. Print Final Summary Table
    print("\n\n=== Part 3 (a) Results Summary: Wikipedia RAG ===")
    print(f"{'n':<3} | {'EM':<6} | {'F1':<6}")
    print("-" * 15)
    
    best_em, best_f1 = 0.0, 0.0
    
    for n, scores in sorted(rag_results.items()):
        print(f"{n:<3} | {scores['em']:.4f} | {scores['f1']:.4f}")
        best_em = max(best_em, scores['em'])
        best_f1 = max(best_f1, scores['f1'])

    print("-" * 15)
    print(f"| Best EM Score: {best_em:.4f}")
    print(f"| Best F1 Score: {best_f1:.4f}")
