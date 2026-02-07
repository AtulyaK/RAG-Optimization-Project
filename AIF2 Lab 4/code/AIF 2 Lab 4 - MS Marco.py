import time
import random
import json
import torch
import pandas as pd
from hotpot_eval import update_answer
from transformers import AutoTokenizer, AutoModelForCausalLM
from pyserini.search.lucene import LuceneSearcher
from pyserini.index.lucene  import Document
from sentence_transformers import CrossEncoder
#import torch.nn.functional as F
# ============================================
# CONFIGURATION
# ============================================

# Model and Token (Use your specific token here)
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
HF_TOKEN = "hf_token"

# Paths and Data Settings
DEV_DATA_PATH = "/home/akadur/AIF2Lab1StarterCode/data/dev.jsonl"
WIKI_INDEX_NAME = "bm25_wiki_index" # Assumed path from your indexer script
SAMPLE_SIZE = 555 # Number of dev examples to evaluate
N_SHOTS_RANGE = range(1, 11) # The core experiment: n=1 to 10

# Generation Parameters
MAX_NEW_TOKENS = 50
TEMPERATURE = 0.1
TOP_P = 1.0
BATCH_SIZE = 10

# Updated Prompt Constants
RAG_SYS_PROMPT = """
You are a highly efficient, direct, and factual AI assistant. Your sole job is to answer the user's question using ONLY the provided context.

***STRICT GENERATION RULES***
1.  **NO FLUFF**: Respond with the **EXACT** factual answer only.
2.  **CONCISE**: The answer must be a **single sentence** or the **key phrase**.
3.  **DO NOT** include any introductory phrases (e.g., "The answer is...", "Based on the context,").
"""
PASSAGE_SEPARATOR = "\n---\n" # Separator for individual retrieved passages

# ============================================
# RAG UTILITY FUNCTIONS (MISSING from original code)
# ============================================

def format_rag_prompt(context_list: list[str]) -> str:
    """
    Combines a list of retrieved text passages into a single, structured context string.
    This places the passages in the format expected by the LLM prompt.
    """
    if not context_list:
        return "No relevant context found."
    
    # Prefix the entire block to make its role clear to the LLM
    formatted_context = "CONTEXT:\n"
    
    # Join individual passages with a clear separator
    for i, passage in enumerate(context_list):
        formatted_context += f"Passage {i+1}: {passage.strip()}{PASSAGE_SEPARATOR}"
        
    return formatted_context.strip()


def make_formatted_prompt(current_user_prompt: str, context: str, system_prompt: str) -> str:
    """
    Composes a formatted instruction-style prompt following the LLaMA 3.1 instruct format.
    The context is placed *after* the user's question. (This is Prompt Format 1 for part 3b)
    """
    base = "<|begin_of_text|>"
    segments = []

    # 1. System Instruction (Includes instruction about the context)
    segments.append(f"<|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|>")

    # 2. Context and Target Question in the user turn
    user_content = f"Question: {context}\n\n{current_user_prompt}"
    
    segments.append(f"<|start_header_id|>user<|end_header_id|>\n{user_content}<|eot_id|>")
    
    # 3. Prompt the model for the assistant's response
    segments.append("<|start_header_id|>assistant<|end_header_id|>")
    
    return base + "\n".join(segments)

# ============================================
# MAIN EXPERIMENT LOOP
# ============================================

def run_rag_experiment(model, tokenizer, device, dev_questions, dev_answers):
    """Runs the RAG experiment for n=1 to 10 retrieved passages."""
    
    # 1. Initialize the Searcher
    try:
        # Use the name of the Wikipedia index you built
        searcher = LuceneSearcher(WIKI_INDEX_NAME)
        if not searcher:
            raise Exception("Searcher could not be initialized.")
        print(f"\nBM25 Searcher initialized using index: {WIKI_INDEX_NAME}")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not initialize LuceneSearcher. Check index path and installation. Details: {e}")
        return {}
    
    all_results = {}

    for n in N_SHOTS_RANGE:
        print(f"\n--- Running RAG with n={n} retrieved passages ---")
        
        start_time = time.time()
        
        # --- Stage 1: Retrieval and Prompt Generation ---
        prompts = []
        
        for i, question in enumerate(dev_questions):
            # Retrieve 'n' most relevant passages using the dev question as the query
            hits = searcher.search(question, k=n)
            
            n_shot_context_text = []
            
            for hit in hits:
                try:
                    doc = Document(hit.lucene_document)
                    rec = json.loads(doc.raw())
                    
                    # Assuming the Wikipedia document content is stored in the 'contents' key
                    # NOTE: This assumes 'contents' contains the full text. 
                    # If your indexer used 'text', change 'contents' to 'text' here.
                    n_shot_context_text.append(rec["contents"]) 
                except Exception as e:
                    # Log retrieval failure but continue
                    print(f"Warning: Failed to retrieve document {hit.docid}: {e}")
                    continue
            
            #NOTE: Re-Ranking done here
            if not n_shot_context_text:
                print("Warning: No documents were retrieved by BM25. Skipping re-ranking.")
            else:
                # Create (Question, Document) pairs for the Cross-Encoder
                pairs_to_score = [[question, doc] for doc in n_shot_context_text]

                # Get the scores from the Cross-Encoder & normalize them
                scores = re_ranker.predict(pairs_to_score)
                #scores = F.sigmoid(scores)
                
                # Combine scores and documents, and sort
                scored_documents = list(zip(scores, n_shot_context_text))
                scored_documents.sort(key=lambda x: x[0], reverse=True)
                n_shot_context_text = [text for score, text in scored_documents]   
                  
            # 1. Format the retrieved context string
            context_str = format_rag_prompt(n_shot_context_text)
            
            # 2. Assemble the full LLaMA formatted prompt (Context BEFORE Question)
            full_prompt = make_formatted_prompt(
                current_user_prompt=question, 
                context=context_str, 
                system_prompt=RAG_SYS_PROMPT
            )
            prompts.append(full_prompt)

        # --- Stage 2: Generation and Evaluation ---
        batch_responses = []
        metrics = {'em': 0.0, 'f1': 0.0, 'prec': 0.0, 'recall': 0.0}

        for i in range(0, len(prompts), BATCH_SIZE):
            batch = prompts[i:i + BATCH_SIZE]
            inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)
            prompt_length = inputs["input_ids"].size(1)

            # Simple retry loop for generation robustness
            for attempt in range(3):
                try:
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=MAX_NEW_TOKENS,
                            temperature=TEMPERATURE,
                            top_p=TOP_P,
                            do_sample=True,
                            pad_token_id=tokenizer.pad_token_id
                        )
                    break # Success
                except Exception as e:
                    wait_time = 2 ** attempt
                    if attempt < 2:
                        # Do not log retries as errors in the console per instruction
                        time.sleep(wait_time)
                    else:
                        print(f"FATAL: Generation failed after 3 attempts on batch starting at index {i}. Skipping batch. Error: {e}")
                        outputs = None
                        break # Failure

            if outputs is None:
                # Append empty strings for failed predictions to maintain list length
                batch_responses.extend([""] * len(batch))
                continue
            
            # Decode responses
            current_batch_responses = []
            for seq in outputs:
                generated_tokens = seq[prompt_length:]
                response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                current_batch_responses.append(response)
            
            batch_responses.extend(current_batch_responses)

            # Update metrics immediately
            for j in range(len(current_batch_responses)):
                 # Get the prediction for this item and the corresponding gold truth
                pred = current_batch_responses[j]
                gold_idx = i + j
                if gold_idx < len(dev_answers):
                    update_answer(metrics, pred, dev_answers[gold_idx])

            if (i + BATCH_SIZE) % 50 == 0:
                print(f"Processed {min(i + BATCH_SIZE, len(prompts))}/{len(prompts)} examples.")

        # --- Final Metrics and Reporting for current n ---

        num_responses = len(dev_answers) # Use the total size of the sample
        
        # Re-normalize metrics based on the full sample size
        final_metrics = {'em': metrics['em'] / num_responses, 'f1': metrics['f1'] / num_responses}
        
        all_results[n] = final_metrics
        
        end_time = time.time()
        print(f"n={n} (EM={final_metrics['em']:.4f}, F1={final_metrics['f1']:.4f}) | Time: {end_time - start_time:.2f}s")

    return all_results

# ============================================
# Execution
# ============================================

if __name__ == "__main__":
    # --- 0. Initialize Model ---
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device being used: {device}")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side='left', token=HF_TOKEN)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else "auto", # Use bfloat16 on GPU
            token=HF_TOKEN
        )
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        #NOTE: Added re-ranker
        re_ranker = CrossEncoder('cross-encoder/ms-marco-L12-v6', token=HF_TOKEN)
    except Exception as e:
        print(f"ERROR: Could not initialize model/tokenizer. Details: {e}")
        exit()

    # --- 1. Load Data ---
    try:
        dev_data = pd.read_json(DEV_DATA_PATH, lines=True)
        
        # Sample the data once at the beginning to ensure consistency across n values
        random.seed(42) # Use a fixed seed for reproducible sampling
        choices = random.sample(range(len(dev_data)), SAMPLE_SIZE)
        
        sample_data = dev_data.iloc[choices]
        dev_questions = list(sample_data['question'])
        dev_answers = list(sample_data['answer'])
        
        if len(dev_questions) != SAMPLE_SIZE:
             print(f"Warning: Loaded sample size is {len(dev_questions)}, expected {SAMPLE_SIZE}")

    except Exception as e:
        print(f"ERROR: Could not load data from {DEV_DATA_PATH}. Details: {e}")
        exit()
    
    # --- 2. Run Experiment ---
    print("\nStarting Part 3 (a): Wikipedia RAG (n=1 to 10)...")
    rag_results = run_rag_experiment(
        model, tokenizer, device, 
        dev_questions, dev_answers
    )

    # --- 3. Print Final Summary Table ---
    print("\n\n=== Part 3 (a) Results Summary: RAG Performance vs. n ===")
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
