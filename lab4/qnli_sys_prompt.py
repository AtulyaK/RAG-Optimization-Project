import time
import random
import json
import torch
import os
import pandas as pd
from typing import List, Dict, Any
from hotpot_eval import update_answer # Assumed external evaluation utility
from transformers import AutoTokenizer, AutoModelForCausalLM
from pyserini.search.lucene import LuceneSearcher
from pyserini.index.lucene  import Document
from sentence_transformers import CrossEncoder

# ============================================
# CONFIGURATION
# ============================================

# LLAMA MODEL CONFIG
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
TOKEN = os.getenv("HF_TOKEN")

# Paths and Data Settings
DEV_DATA_PATH = os.path.join(os.path.dirname(__file__), "data/dev.jsonl")
WIKI_INDEX_NAME = os.path.join(os.path.dirname(__file__), "bm25_wiki_index")
SAMPLE_SIZE = 555
N_SHOTS_RANGE = range(1, 11)

# File Saving
OUTPUT_DIR = "ragas_evaluation_data"
OUTPUT_SUB_DIR = "qnli_and_sys"
# NOTE: The final file name will be e.g., ragas_evaluation_data/ragas_data_n_5.jsonl

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
PASSAGE_SEPARATOR = "\n---\n" 

# ============================================
# RAG UTILITY FUNCTIONS
# ============================================

def format_rag_prompt(context_list: List[str]) -> str:
    """
    Combines a list of retrieved text passages into a single, structured context string.
    """
    if not context_list:
        return "No relevant context found."
    
    formatted_context = "CONTEXT:\n"
    
    for i, passage in enumerate(context_list):
        formatted_context += f"Passage {i+1}: {passage.strip()}{PASSAGE_SEPARATOR}"
        
    return formatted_context.strip()


def make_formatted_prompt(current_user_prompt: str, context: str, system_prompt: str) -> str:
    """
    Composes a formatted instruction-style prompt following the LLaMA 3.1 instruct format.
    """
    base = "<|begin_of_text|>"
    segments = []

    # 1. System Instruction 
    segments.append(f"<|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|>")

    # 2. Context and Target Question in the user turn
    user_content = f"Question: {context}\n\n{current_user_prompt}"
    
    segments.append(f"<|start_header_id|>user<|end_header_id|>\n{user_content}<|eot_id|>")
    
    # 3. Prompt the model for the assistant's response
    segments.append("<|start_header_id|>assistant<|end_header_id|>")
    
    return base + "\n".join(segments)


def save_ragas_data(data: List[Dict[str, Any]], n: int):
    """Saves the collected RAGAS data components to a JSON Lines file."""
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_SUB_DIR)
    os.makedirs(output_path, exist_ok=True)  # Create the full directory tree if missing
        
    filename = os.path.join(OUTPUT_DIR, OUTPUT_SUB_DIR,f"ragas_data_n_{n}.jsonl")
    
    with open(filename, 'w') as f:
        for item in data:
            # If it errors out convert ground truth to a list holding string
            # item['ground_truth'] = [item['ground_truth']]
            f.write(json.dumps(item) + '\n')
            
    print(f"\nSuccessfully saved RAGAS evaluation data to: {filename}")
# ============================================
# MAIN EXPERIMENT LOOP
# ============================================

def run_rag_experiment(model, tokenizer, device, dev_questions, dev_answers):
    """Runs the RAG experiment for n=1 to 10 retrieved passages and saves data."""
    
    # RAGAS configuration is skipped as per user request.

    # 1. Initialize the Searcher
    try:
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
        
        # --- DATA COLLECTION LIST ---
        # List of dictionaries, each dict holds one question's results
        collected_data = [] 
        
        # --- Stage 1: Retrieval and Prompt Generation ---
        prompts = []
        
        for i, question in enumerate(dev_questions):
            # Retrieve 'n' most relevant passages
            hits = searcher.search(question, k=n)
            original_context_text = []
            
            for hit in hits:
                try:
                    doc = Document(hit.lucene_document)
                    rec = json.loads(doc.raw())
                    original_context_text.append(rec["contents"]) 
                except Exception as e:
                    print(f"Warning: Failed to retrieve document {hit.docid}: {e}")
                    continue
            
            # NOTE: Re-Ranking and Reordering Logic
            reordered_context = []
            
            if not original_context_text:
                print("Warning: No documents were retrieved by BM25. Skipping re-ranking.")
            else:
                pairs_to_score = [[question, doc] for doc in original_context_text]
                scores = re_ranker.predict(pairs_to_score)
                scored_documents = list(zip(scores, original_context_text))
                scored_documents.sort(key=lambda x: x[0], reverse=True)
                
                top_k_documents = [text for score, text in scored_documents]

                # # Apply Lost-in-the-Middle mitigation -- update the collected_data if using Lost-in-the-Middle mitigation
                # if not top_k_documents:
                #     pass
                # elif len(top_k_documents) < 3:
                #     reordered_context = top_k_documents
                # else:
                #     reordered_context.append(top_k_documents[0]) 
                #     reordered_context.extend(top_k_documents[2:])
                #     reordered_context.append(top_k_documents[1])
                    
                # final_context_list = reordered_context  
            
            # 1. Format the retrieved context string
            context_str = format_rag_prompt(top_k_documents)
            
            # 2. Assemble the full LLaMA formatted prompt
            full_prompt = make_formatted_prompt(
                current_user_prompt=question, 
                context=context_str, 
                system_prompt=RAG_SYS_PROMPT
            )
            prompts.append(full_prompt)
            
            # Temporarily store essential RAGAS components for this question
            # We store the *list* of contexts, not the formatted string
            collected_data.append({
                'question': question,
                'contexts': top_k_documents,
                'ground_truth': dev_answers[i] 
            })


        # --- Stage 2: Generation and EM/F1 Evaluation (collecting generated answers) ---
        batch_responses = []
        metrics = {'em': 0.0, 'f1': 0.0, 'prec': 0.0, 'recall': 0.0}

        for i in range(0, len(prompts), BATCH_SIZE):
            batch = prompts[i:i + BATCH_SIZE]
            inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)
            prompt_length = inputs["input_ids"].size(1)

            # Simple retry loop for generation robustness
            outputs = None
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
                    break 
                except Exception as e:
                    wait_time = 2 ** attempt
                    if attempt < 2:
                        time.sleep(wait_time)
                    else:
                        print(f"FATAL: Generation failed on batch starting at index {i}. Error: {e}")
                        outputs = None
                        break 

            if outputs is None:
                current_batch_responses = [""] * len(batch)
            else:
                current_batch_responses = []
                for seq in outputs:
                    generated_tokens = seq[prompt_length:]
                    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                    current_batch_responses.append(response)

            batch_responses.extend(current_batch_responses)

            # Update traditional EM/F1 metrics
            for j in range(len(current_batch_responses)):
                pred = current_batch_responses[j]
                gold_idx = i + j
                if gold_idx < len(dev_answers):
                    update_answer(metrics, pred, dev_answers[gold_idx])
                    
                # *** Crucial Step: Add the generated answer to the collected data ***
                # The index i + j corresponds to the index in collected_data
                if gold_idx < len(collected_data):
                    collected_data[gold_idx]['answer'] = pred


            if (i + BATCH_SIZE) % 50 == 0:
                print(f"Processed {min(i + BATCH_SIZE, len(prompts))}/{len(prompts)} examples.")
        
        # --- Stage 3: Save Data ---
        save_ragas_data(collected_data, n)

        # --- Final Metrics and Reporting for current n ---
        num_responses = len(dev_answers)
        
        final_metrics = {
            'em': metrics['em'] / num_responses, 
            'f1': metrics['f1'] / num_responses,
        }
        
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
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else "auto",
            token=HF_TOKEN
        )
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        re_ranker = CrossEncoder('cross-encoder/qnli-electra-base', token=HF_TOKEN)
    except Exception as e:
        print(f"ERROR: Could not initialize model/tokenizer. Details: {e}")
        exit()

    # --- 1. Load Data ---
    try:
        dev_data = pd.read_json(DEV_DATA_PATH, lines=True)
        random.seed(42) 
        choices = random.sample(range(len(dev_data)), SAMPLE_SIZE)
        sample_data = dev_data.iloc[choices]
        dev_questions = list(sample_data['question'])
        # Ensure ground_truth is a list, as RAGAS expects it
        dev_answers = list(sample_data['answer']) 
        
        if len(dev_questions) != SAMPLE_SIZE:
             print(f"Warning: Loaded sample size is {len(dev_questions)}, expected {SAMPLE_SIZE}")
    except Exception as e:
        print(f"ERROR: Could not load data from {DEV_DATA_PATH}. Details: {e}")
        exit()
    
    # --- 2. Run Experiment ---
    print("\nStarting RAG Experiment (n=1 to 10) - Results will be saved to disk...")
    rag_results = run_rag_experiment(
        model, tokenizer, device, 
        dev_questions, dev_answers
    )

    # --- 3. Print Final Summary Table ---
    print("\n\n=== Final Results Summary: RAG Performance vs. n (EM/F1 Only) ===")
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
