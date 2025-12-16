import pandas as pd
from datasets import Dataset, ClassLabel
import kagglehub
import os
import sys
import glob

# Configuration
OUTPUT_DIR = "./data/composite_dataset"
SEED = 42

# ---------------------------------------------------------
# 1. Amazon Reviews (Kaggle)
# ---------------------------------------------------------
def prepare_amazon_data(sample_size=20000):
    print("Downloading Amazon Reviews from Kaggle...")
    
    try:
        # Download latest version from Kaggle
        path = kagglehub.dataset_download("dongrelaxman/amazon-reviews-dataset")
        print(f"  - Dataset downloaded to: {path}")
        
        # Find CSV files in the downloaded path
        csv_files = glob.glob(os.path.join(path, "*.csv"))
        if not csv_files:
            # Try looking in subdirectories
            csv_files = glob.glob(os.path.join(path, "**/*.csv"), recursive=True)
        
        if not csv_files:
            print(f"  - No CSV files found in {path}")
            return pd.DataFrame(columns=["text", "label"])
        
        print(f"  - Found files: {csv_files}")
        
        # Load the first CSV file found with robust parsing options
        df = pd.read_csv(
            csv_files[0], 
            nrows=sample_size * 2,  # Load extra for filtering
            on_bad_lines='skip',    # Skip malformed lines
            encoding='utf-8',
            engine='python'         # More flexible parser
        )
        print(f"  - Loaded {len(df)} rows from {csv_files[0]}")
        print(f"  - Columns: {df.columns.tolist()}")
        
        # Try to identify the text and rating columns
        text_col = None
        rating_col = None
        
        # Common column name patterns for text
        for col in ['Review Text', 'reviewText', 'review_text', 'text', 'review', 'Review', 'Text', 'content']:
            if col in df.columns:
                text_col = col
                break
        
        # Common column name patterns for rating
        for col in ['Rating', 'overall', 'rating', 'score', 'Score', 'stars']:
            if col in df.columns:
                rating_col = col
                break
        
        if text_col is None or rating_col is None:
            print(f"  - Could not identify text or rating columns.")
            print(f"  - Available columns: {df.columns.tolist()}")
            return pd.DataFrame(columns=["text", "label"])
        
        print(f"  - Using text column: '{text_col}', rating column: '{rating_col}'")
        
        # Process the data
        df = df.dropna(subset=[text_col, rating_col])
        
        # Handle "Rated X out of 5 stars" format
        if df[rating_col].dtype == 'object' and df[rating_col].str.contains('Rated', na=False).any():
            df['label'] = df[rating_col].str.extract(r'Rated (\d+) out of')[0].astype(float) - 1
        else:
            df['label'] = df[rating_col].astype(int) - 1  # Maps 1-5 to 0-4
        
        df = df.dropna(subset=['label'])
        df['label'] = df['label'].astype(int)
        df = df[df['label'].between(0, 4)]
        df = df[[text_col, 'label']].rename(columns={text_col: 'text'})
        
        # Sample if we have more than needed
        if len(df) > sample_size:
            df = df.sample(sample_size, random_state=SEED)
        
        print(f"  - Successfully processed {len(df)} Amazon samples")
        return df
        
    except Exception as e:
        print(f"Error loading Amazon dataset from Kaggle: {e}")
        return pd.DataFrame(columns=["text", "label"])

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
    
    priority_map = {'Blocker': 0, 'Critical': 1, 'Major': 2, 'Minor': 3, 'Trivial': 4}
    
    # Normalize columns
    df.columns = [c.lower() for c in df.columns]
    
    if 'priority' not in df.columns:
        return pd.DataFrame(columns=["text", "label"])
    
    text_col = 'description' if 'description' in df.columns else 'body'
    if text_col not in df.columns:
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
        return pd.DataFrame(columns=["text", "label"])

    df = df.dropna(subset=[text_col, rating_col])
    df['label'] = df[rating_col].astype(int) - 1
    df = df[df['label'].between(0, 4)]
    
    df = df[[text_col, 'label']].rename(columns={text_col: 'text'})
    print(f"  - Loaded {len(df)} Google Play samples")
    return df

# ---------------------------------------------------------
# 4. Mock Data Generator (Fallback)
# ---------------------------------------------------------
def generate_mock_data(count=100):
    """
    Generates fake data if everything else fails, allowing the pipeline to be tested.
    """
    print("!!! GENERATING MOCK DATA (FOR TESTING PIPELINE) !!!")
    data = []
    labels = [
        (0, "This is absolutely broken and unusable. I am leaving."),
        (1, "Essential feature missing. I cannot do my job."),
        (2, "It would be good to have this feature for my workflow."),
        (3, "Nice to have, but not urgent. Good app generally."),
        (4, "Amazing app, works perfectly! Love it.")
    ]
    import random
    for _ in range(count):
        label, text = random.choice(labels)
        data.append({"text": text, "label": label})
    return pd.DataFrame(data)

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    if not os.path.exists("./data"):
        os.makedirs("./data")

    # 1. Load Data
    df_amazon = prepare_amazon_data(sample_size=50000) 
    df_tawos = prepare_tawos_data("./data/tawos_dump.csv") 
    df_gplay = prepare_google_play_data("./data/google_play_finance.csv")

    # 2. Combine
    print("\nCombining Datasets...")
    dfs_to_concat = [df for df in [df_amazon, df_tawos, df_gplay] if not df.empty]
    
    # FALLBACK LOGIC
    if not dfs_to_concat:
        print("CRITICAL ERROR: No real data loaded from sources.")
        print("1. Amazon failed? Check internet/file path logic.")
        print("2. Local files missing? You need TAWOS/GooglePlay CSVs in ./data/")
        print("\n--> Switching to MOCK DATA for pipeline verification.")
        
        df_mock = generate_mock_data(200)
        dfs_to_concat = [df_mock]
        
    combined_df = pd.concat(dfs_to_concat, ignore_index=True)
    
    # 3. Stratified Sampling
    print("Balancing classes...")
    min_class_count = combined_df['label'].value_counts().min()
    
    # Ensure we don't try to sample more than exists
    target_samples = min(max(min_class_count, 50), 15000) 
    
    balanced_dfs = []
    for label in range(5):
        class_subset = combined_df[combined_df['label'] == label]
        if len(class_subset) > 0:
            if len(class_subset) > target_samples:
                balanced_dfs.append(class_subset.sample(target_samples, random_state=SEED))
            else:
                balanced_dfs.append(class_subset)
            
    if balanced_dfs:
        final_df = pd.concat(balanced_dfs).sample(frac=1, random_state=SEED).reset_index(drop=True)
        
        print(f"\nFinal Dataset Statistics:")
        print(final_df['label'].value_counts().sort_index())
        
        # 4. Save
        print(f"\nSaving to {OUTPUT_DIR}...")
        hf_dataset = Dataset.from_pandas(final_df)
        hf_dataset = hf_dataset.cast_column('label', ClassLabel(num_classes=5, names=["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]))
        hf_dataset.save_to_disk(OUTPUT_DIR)
        
        print(f"Done! Dataset saved to {OUTPUT_DIR}")
    else:
        print("Error: Dataset empty after balancing. Check mock data generation.")