"""
Generate embeddings for any dataset based on config.yaml.
This script should be run once to create the embeddings files.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pickle
import yaml
from typing import Dict, List, Any
import sys


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_dataset(config: Dict[str, Any]) -> pd.DataFrame:
    """Load dataset from CSV file(s)."""
    dataset_config = config['dataset']

    # Check if single file or multiple files
    if 'file_path' in dataset_config:
        file_path = dataset_config['file_path']
        print(f"Loading dataset from: {file_path}")
        df = pd.read_csv(file_path)
    elif 'file_paths' in dataset_config:
        file_paths = dataset_config['file_paths']
        print(f"Loading dataset from {len(file_paths)} files...")
        dfs = [pd.read_csv(fp) for fp in file_paths]
        df = pd.concat(dfs, ignore_index=True)
    else:
        raise ValueError("Config must specify either 'file_path' or 'file_paths'")

    print(f"Total records: {len(df)}")
    return df


def validate_config(config: Dict[str, Any], df: pd.DataFrame) -> None:
    """Validate that config columns exist in dataframe."""
    errors = []

    # Check text columns
    for text_col in config['text_columns']:
        col_name = text_col['column']
        if col_name not in df.columns:
            errors.append(f"Text column '{col_name}' not found in dataset")

    # Check metadata columns
    metadata = config['metadata']
    if metadata['date_column'] not in df.columns:
        errors.append(f"Date column '{metadata['date_column']}' not found in dataset")

    if metadata['score_column'] not in df.columns:
        errors.append(f"Score column '{metadata['score_column']}' not found in dataset")

    if 'id_column' in metadata and metadata['id_column'] not in df.columns:
        errors.append(f"ID column '{metadata['id_column']}' not found in dataset")

    # Check display columns
    if 'display_columns' in metadata:
        for col in metadata['display_columns']:
            if col not in df.columns:
                errors.append(f"Display column '{col}' not found in dataset")

    if errors:
        print("\n[ERROR] Configuration validation errors:")
        for error in errors:
            print(f"  - {error}")
        print(f"\nAvailable columns in dataset: {list(df.columns)}")
        sys.exit(1)

    print("[OK] Configuration validated successfully")


def generate_embeddings_for_column(
    df: pd.DataFrame,
    column_name: str,
    model: SentenceTransformer,
    batch_size: int = 1000,
    encoding_batch_size: int = 32
) -> np.ndarray:
    """Generate embeddings for a specific text column."""
    print(f"\nGenerating embeddings for column '{column_name}'...")

    # Fill NaN values with empty string
    texts = df[column_name].fillna('').tolist()

    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = model.encode(
            batch_texts,
            show_progress_bar=True,
            batch_size=encoding_batch_size,
            normalize_embeddings=True  # Normalized for cosine similarity
        )
        all_embeddings.append(batch_embeddings)
        print(f"Processed {min(i+batch_size, len(texts))}/{len(texts)} items")

    embeddings = np.vstack(all_embeddings)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embeddings size: {embeddings.nbytes / (1024*1024):.2f} MB")

    return embeddings


def main():
    # Load configuration
    print("Loading configuration from config.yaml...")
    config = load_config()

    # Load dataset
    df = load_dataset(config)

    # Validate configuration
    validate_config(config, df)

    # Load embedding model
    model_name = config['embeddings']['model']
    print(f"\nLoading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    # Create output directory
    output_dir = Path(config['embeddings']['output_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)

    # Generate embeddings for each text column
    embedding_files = {}
    batch_size = config['embeddings'].get('batch_size', 32)

    for text_col in config['text_columns']:
        col_name = text_col['column']
        internal_name = text_col['name']

        embeddings = generate_embeddings_for_column(
            df=df,
            column_name=col_name,
            model=model,
            batch_size=1000,
            encoding_batch_size=batch_size
        )

        # Save embeddings
        output_file = output_dir / f'{internal_name}_embeddings.npz'
        np.savez_compressed(
            output_file,
            embeddings=embeddings.astype(np.float32)
        )

        file_size = output_file.stat().st_size / (1024*1024)
        embedding_files[internal_name] = {
            'file': output_file,
            'size': file_size
        }
        print(f"Saved: {output_file} ({file_size:.2f} MB)")

    # Prepare metadata
    print("\nPreparing metadata...")
    metadata = {}

    # Add text columns
    for text_col in config['text_columns']:
        col_name = text_col['column']
        internal_name = text_col['name']
        metadata[internal_name] = df[col_name].fillna('').values

    # Add date column (use the actual column name as the key)
    date_col = config['metadata']['date_column']
    metadata[date_col] = df[date_col].values

    # Add score column (use the actual column name as the key)
    score_col = config['metadata']['score_column']
    metadata[score_col] = df[score_col].values

    # Add ID column if specified (use the actual column name as the key)
    if 'id_column' in config['metadata']:
        id_col = config['metadata']['id_column']
        metadata[id_col] = df[id_col].values

    # Add display columns if specified (use the actual column names as keys)
    if 'display_columns' in config['metadata']:
        for col in config['metadata']['display_columns']:
            metadata[col] = df[col].values

    # Save metadata
    metadata_file = output_dir / 'metadata.pkl'
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f, protocol=4)

    metadata_size = metadata_file.stat().st_size / (1024*1024)
    print(f"Metadata saved: {metadata_file} ({metadata_size:.2f} MB)")

    # Summary
    total_size = sum(info['size'] for info in embedding_files.values()) + metadata_size

    print(f"\n{'='*60}")
    print("[OK] Embedding generation complete!")
    print(f"{'='*60}")
    print(f"\nGenerated embeddings:")
    for name, info in embedding_files.items():
        print(f"  - {name}: {info['file']} ({info['size']:.2f} MB)")
    print(f"\nMetadata: {metadata_file} ({metadata_size:.2f} MB)")
    print(f"\nTotal size: {total_size:.2f} MB")
    print(f"\nNext step: Run 'streamlit run app.py' to start the app!")


if __name__ == "__main__":
    main()
