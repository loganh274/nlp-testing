import pandas as pd
from datasets import load_dataset, Dataset, ClassLabel
from huggingface_hub import hf_hub_download
import os

# Configuration
OUTPUT_DIR = "./data/composite_dataset"
SEED = 42

# ---------------------------------------------------------
# 1. Amazon Reviews (Hugging Face) - Explicit Download Method
# ---------------------------------------------------------
def prepare_amazon_data(sample_size=20000):
    print("Downloading Amazon Reviews (Software) parquet file...")
    
    try:
        # Step 1: Download the file explicitly using huggingface_hub
        # This handles the URL resolution and caching automatically
        local_parquet_path = hf_hub_download(
            repo_id="McAuley-Lab/Amazon-Reviews-2023",
            filename="data/raw_review_Software.parquet",
            repo_type="dataset"
        )
        print(f"  - File cached at: {local_parquet_path}")

        # Step 2: Load the local file using the parquet builder
        dataset = load_dataset(
            "parquet", 
            data_files={"train": local_parquet_path}, 
            split="train", 
            streaming=True
        )
    except Exception as e:
        print(f"Error loading Amazon dataset: {e}")
        print("Tip: Ensure 'huggingface_hub' and 'zstandard' are installed.")
        return pd.DataFrame(columns=["text", "label"])
    
    data = []
    # Iterate and map labels
    # 1.0 -> 0 (Very Negative/Blocker)
    # 2.0 -> 1 (Negative/Critical)
    # 3.0 -> 2 (Neutral/Major)
    # 4.0 -> 3 (Positive/Minor)
    # 5.0 -> 4 (Very Positive/Trivial)
    
    print(f"Processing Amazon stream (aiming for {sample_size} samples)...")
    iterator = iter(dataset)
    try:
        # We assume about 50% retention rate after filtering, so grab double
        for _ in range(sample_size * 2): 
            try:
                item = next(iterator)
                # Check for rating and text existence
                if item.get('rating') is not None and item.get('text'):
                    rating = int(item['rating'])
                    label = rating - 1 # Maps 1-5 to 0-4
                    
                    text = item['text']
                    if 0 <= label <= 4:
                        data.append({"text": text, "label": label})
                
                if len(data) >= sample_size:
                    break
            except StopIteration:
                break
            except Exception:
                continue 
    except StopIteration:
        pass

    print(f"  - Successfully processed {len(data)} Amazon samples")
    return pd.DataFrame(data)

# ---------------------------------------------------------
# 2. TAWOS / Jira Data (Local CSV)
# ---------------------------------------------------------
def prepare_tawos_data(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: TAWOS file not found at {filepath}. Skipping.")
        return pd.DataFrame(columns=["text", "label"])

    print("Loading TAWOS/Jira Data...")
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading TAWOS file: {e}")
        return pd.DataFrame(columns=["text", "label"])
    
    priority_map = {
        'Blocker': 0, 'Critical': 1, 'Major': 2, 'Minor': 3, 'Trivial': 4
    }
    
    # Check for columns (loosely)
    cols = [c.lower() for c in df.columns]
    if 'priority' not in cols:
        print("  - Error: TAWOS CSV missing 'priority' column.")
        return pd.DataFrame(columns=["text", "label"])
    
    # Normalize columns
    df.columns = [c.lower() for c in df.columns]
    # Assume 'description' or 'body' is the text
    text_col = 'description' if 'description' in df.columns else 'body'
    
    if text_col not in df.columns:
         print(f"  - Error: TAWOS CSV missing '{text_col}' column.")
         return pd.DataFrame(columns=["text", "label"])

    df = df.dropna(subset=[text_col, 'priority'])
    df['label'] = df['priority'].map(priority_map)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    
    df = df[[text_col, 'label']].rename(columns={text_col: 'text'})
    
    print(f"  - Loaded {len(df)} TAWOS samples")
    return df

# ---------------------------------------------------------
# 3. Google Play Finance (Local CSV)
# ---------------------------------------------------------
def prepare_google_play_data(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: Google Play file not found at {filepath}. Skipping.")
        return pd.DataFrame(columns=["text", "label"])

    print("Loading Google Play Finance Data...")
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading Google Play file: {e}")
        return pd.DataFrame(columns=["text", "label"])
    
    text_col = 'content' if 'content' in df.columns else 'review'
    rating_col = 'score' if 'score' in df.columns else 'rating'
    
    if text_col not in df.columns or rating_col not in df.columns:
        print(f"  - Error: Could not find '{text_col}' or '{rating_col}' columns in Google Play CSV.")
        return pd.DataFrame(columns=["text", "label"])

    df = df.dropna(subset=[text_col, rating_col])
    df['label'] = df[rating_col].astype(int) - 1
    df = df[df['label'].between(0, 4)]
    
    df = df[[text_col, 'label']].rename(columns={text_col: 'text'})
    
    print(f"  - Loaded {len(df)} Google Play samples")
    return df

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    if not os.path.exists("./data"):
        os.makedirs("./data")

    # 1. Load Data
    df_amazon = prepare_amazon_data(sample_size=50000) 
    
    # NOTE: You must manually download these files and put them in the 'data' folder
    # TAWOS: https://github.com/SOLAR-group/TAWOS or Zenodo
    # Google Play: https://www.kaggle.com/datasets/ ... (Finance category)
    df_tawos = prepare_tawos_data("./data/tawos_dump.csv") 
    df_gplay = prepare_google_play_data("./data/google_play_finance.csv")

    # 2. Combine
    print("\nCombining Datasets...")
    dfs_to_concat = [df for df in [df_amazon, df_tawos, df_gplay] if not df.empty]
    
    if not dfs_to_concat:
        print("CRITICAL ERROR: No data loaded.")
        print("1. Amazon failed? Check internet connection.")
        print("2. Local files missing? You must manually download TAWOS/GooglePlay CSVs to ./data/")
        exit()
        
    combined_df = pd.concat(dfs_to_concat, ignore_index=True)
    
    # 3. Stratified Sampling
    print("Balancing classes...")
    min_class_count = combined_df['label'].value_counts().min()
    target_samples = min(max(min_class_count, 1000), 15000)
    
    balanced_dfs = []
    for label in range(5):
        class_subset = combined_df[combined_df['label'] == label]
        if len(class_subset) > target_samples:
            balanced_dfs.append(class_subset.sample(target_samples, random_state=SEED))
        else:
            balanced_dfs.append(class_subset)
            
    final_df = pd.concat(balanced_dfs).sample(frac=1, random_state=SEED).reset_index(drop=True)
    
    print(f"\nFinal Dataset Statistics:")
    print(final_df['label'].value_counts().sort_index())
    
    # 4. Save
    print(f"\nSaving to {OUTPUT_DIR}...")
    hf_dataset = Dataset.from_pandas(final_df)
    hf_dataset = hf_dataset.cast_column('label', ClassLabel(num_classes=5, names=["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]))
    hf_dataset.save_to_disk(OUTPUT_DIR)
    
    print(f"Done! Dataset saved to {OUTPUT_DIR}")