import json
import os
import subprocess
import pandas as pd

# --- Configuration Constants ---
WIKI_DATA_PATH = "/home/akadur/AIF2Lab1StarterCode/data/wiki_snippets_dev.json"
WIKI_INDEX_PATH = "bm25_wiki_index" # The folder where the Lucene index will be saved
COLLECTION_DIR = "wiki_collection_jsonl" # Temporary folder to hold the formatted JSONL file

# ============================================
# BM25 Indexing for Wikipedia Snippets (Fixed for Structured Data)
# ============================================

def build_wikipedia_index(data_path: str, index_path: str, collection_dir: str):
    """
    Creates a Pyserini index for the provided Wikipedia snippets data.
    
    This function is optimized for the HotpotQA-style structured data
    where the file is a single JSON dictionary mapping questions/IDs to lists of snippets.
    It flattens this structure into a list of documents for Pyserini indexing.
    """
    if os.path.isdir(index_path):
        print(f"Index already exists at {index_path}. Skipping index creation.")
        return

    print(f"Formatting data for Pyserini indexing from: {data_path}...")
    os.makedirs(collection_dir, exist_ok=True)
    
    # 1. Load the Wikipedia data as a single JSON object (dictionary)
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            # Load the entire file, which is assumed to be a dictionary: {question_id: list_of_snippets}
            wiki_data = json.load(f)
            if not isinstance(wiki_data, dict):
                raise TypeError("Data loaded is not a dictionary. Expected format: {ID: [Snippets...], ...}")
            print(f"Successfully loaded data as a single dictionary with {len(wiki_data)} root entries.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load and validate Wikipedia data. Details: {e}")
        return

    # 2. Format and Flatten data into Pyserini's expected JSONL structure
    pyserini_docs = []
    doc_counter = 0
    
    # Iterate over the structure: {question_id: [snippet1, snippet2, ...]}
    for question_id, list_of_snippets in wiki_data.items():
        if not isinstance(list_of_snippets, list):
            print(f"Warning: Skipping non-list entry for ID {question_id}.")
            continue
            
        for snippet_text in list_of_snippets:
            if not isinstance(snippet_text, str) or not snippet_text.strip():
                continue # Skip empty or non-string snippets

            # Create a unique ID for each snippet using a counter
            unique_doc_id = f"{question_id}-{doc_counter}" 
            
            doc_content = {
                "id": unique_doc_id, 
                "contents": snippet_text, # This is the field Pyserini indexes and searches against
                "source_id": question_id # Keep the original question ID for tracking if needed
            }
            pyserini_docs.append(json.dumps(doc_content))
            doc_counter += 1
            
    if not pyserini_docs:
        print("ERROR: No valid documents found after flattening the structured data.")
        return

    # Write all documents to a single JSONL file in the collection directory
    collection_filepath = os.path.join(collection_dir, "wiki_docs.jsonl")
    with open(collection_filepath, "w", encoding='utf-8') as f:
        f.write("\n".join(pyserini_docs))
    
    print(f"Indexing {len(pyserini_docs)} documents...")
    
    # 3. Run Pyserini's indexer as a subprocess
    try:
        # We use JsonCollection and DefaultLuceneDocumentGenerator, and crucially, -storeRaw
        subprocess.run([
            "python", "-m", "pyserini.index",
              "-collection", "JsonCollection",
              "-generator", "DefaultLuceneDocumentGenerator",
              "-threads", "4", 
              "-input", collection_dir,
              "-index", index_path,
              "-storeRaw" # MUST store raw data to retrieve the full passage text later
        ], check=True, capture_output=False) 
        print(f"→ Indexing complete. Index stored at ./{index_path}")
    except subprocess.CalledProcessError as e:
        print(f"CRITICAL ERROR during Pyserini indexing. Details: {e}")
        # The specific error details are often useful for debugging Pyserini issues
        raise


if __name__ == "__main__":
    build_wikipedia_index(WIKI_DATA_PATH, WIKI_INDEX_PATH, COLLECTION_DIR)
