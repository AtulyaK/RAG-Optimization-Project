import os
import glob
import pandas as pd
from datasets import Dataset
from typing import Dict, List, Any

# --- LOCAL EMBEDDING REQUIREMENTS ---
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain_core.exceptions import OutputParserException 

# --- RAGAS & GROQ IMPORTS ---
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

try:
    from langchain_groq import ChatGroq
except ImportError:
    print("FATAL: langchain-groq not installed. Please install: pip install langchain-groq")
    exit()

# ============================================
# CONFIGURATION
# ============================================

INPUT_DIR = "ragas_evaluation_data"
INPUT_SUB_DIR = "qnli_and_sys"
# NOTE: Set your environment variable: export GROQ_API_KEY="YOUR_KEY"

# Groq Configuration (LLM Judge)
GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_API_KEY = "groq_api_key"

# Local Embedding Configuration (for Answer Relevancy)
LOCAL_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" 


# ============================================
# CUSTOM LOCAL EMBEDDING WRAPPER
# ============================================

class LocalSentenceTransformerEmbeddings(Embeddings):
    """A wrapper for the SentenceTransformer model to be used as a LangChain Embeddings object."""

    def __init__(self, model_name: str = LOCAL_EMBEDDING_MODEL_NAME):
        # Load model on initialization
        print(f"Loading local embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Encodes a list of documents
        embeddings = self.model.encode(texts, convert_to_numpy=False, convert_to_tensor=False).tolist()
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        # Encodes a single query string
        embedding = self.model.encode([text], convert_to_numpy=False, convert_to_tensor=False).tolist()
        return embedding[0]


# ============================================
# UTILITY AND EVALUATION FUNCTIONS
# ============================================

def load_ragas_data_from_folder(folder_path: str, folder_sub_path: str) -> Dict[int, Dataset]:
    """Loads all saved .jsonl files into a dictionary of HuggingFace Datasets."""
    all_datasets = {}
    
    jsonl_files = glob.glob(os.path.join(folder_path, folder_sub_path, "ragas_data_n_*.jsonl"))
    
    if not jsonl_files:
        print(f"Error: No .jsonl files found in the directory '{folder_path}'. Please run the data collection script first.")
        return {}

    for file_path in jsonl_files:
        try:
            n_value = int(file_path.split('_n_')[-1].replace('.jsonl', ''))
            df = pd.read_json(file_path, lines=True)
            
            # Contexts must be a list of strings for RAGAS
            df['contexts'] = df['contexts'].apply(lambda x: x if isinstance(x, list) else [x])
            
            all_datasets[n_value] = Dataset.from_pandas(df)
            print(f"Loaded data for n={n_value} ({len(df)} samples).")
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            
    return all_datasets

def run_evaluation(datasets: Dict[int, Dataset], groq_key: str, hf_token: str):
    """Initializes Groq and Local Embeddings, then runs RAGAS evaluation."""
    
    # --- 1. Initialize Groq LLM (Judge) ---
    try:

        print(f"\nInitializing Groq LLM Judge ({GROQ_MODEL})...")
        groq_llm = ChatGroq(
            temperature=0, 
            model_name=GROQ_MODEL,
            groq_api_key=GROQ_API_KEY
        )
        judge_llm_wrapper = LangchainLLMWrapper(groq_llm)
        
    except Exception as e:
        print(f"FATAL: Groq LLM initialization failed. RAGAS cannot run. Error: {e}")
        return

    # --- 2. Initialize Local Embeddings (Answer Relevancy) ---
    try:
         local_st_embeddings = LocalSentenceTransformerEmbeddings(model_name=LOCAL_EMBEDDING_MODEL_NAME) 
         judge_emb_wrapper = LangchainEmbeddingsWrapper(local_st_embeddings)
         print(f"Successfully initialized local SentenceTransformer for embeddings.")
    except Exception as e:
         print(f"WARNING: Failed to initialize local embeddings ({e}). Answer Relevancy will be skipped.")
         judge_emb_wrapper = None
    
    # --- 3. Run RAGAS Evaluation ---
    
    ragas_metrics = [faithfulness, answer_relevancy, context_recall, context_precision]
    
    # Assign the Judge LLM and Embeddings to all metrics
    for metric in ragas_metrics:
        metric.llm = judge_llm_wrapper
        if hasattr(metric, 'embeddings') and judge_emb_wrapper:
            metric.embeddings = judge_emb_wrapper
            
    title = f"Groq ({GROQ_MODEL}) + Local Embeddings"
    print(f"\n--- Starting RAGAS Evaluation: {title} ---")
    
    all_results_df = []
    
    for n, dataset in sorted(datasets.items()):
        print(f"Evaluating n={n} ({len(dataset)} samples)...")
        
        try:
            result = evaluate(dataset, metrics=ragas_metrics)
            result_df = result.to_pandas().mean().to_frame(name=f'n={n}').T
            all_results_df.append(result_df)
            
        except OutputParserException as e:
            print(f"  RAGAS ERROR for n={n}: Output parsing failed (usually due to LLM not returning perfect JSON). Skipping n.")
            # For Groq, this usually means adding more prompt constraints, but we skip for the current run.
        except Exception as e:
            print(f"  RAGAS FAILED for n={n} due to unhandled error: {e}. Skipping n.")


    # --- 4. Print Summary ---
    if all_results_df:
        final_summary = pd.concat(all_results_df)
        print("\n" + "="*80)
        print(f"Final RAGAS Summary ({title}):")
        print("="*80)
        print(final_summary.to_markdown())
        print("="*80 + "\n")
    else:
        print("\nNo successful RAGAS evaluations were completed.")


if __name__ == "__main__":
    
    # Check for the required keys
    groq_api_key = os.getenv("GROQ_API_KEY")
    hf_token = os.getenv("HF_TOKEN") # Pass this if your system requires it for SentenceTransformer downloads

    # 0. Load all saved data files
    ragas_datasets = load_ragas_data_from_folder(INPUT_DIR, INPUT_SUB_DIR)
    
    if ragas_datasets:
        run_evaluation(ragas_datasets, groq_api_key, hf_token)
