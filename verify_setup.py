"""
Setup verification script - run this before generate_embeddings.py
Checks that your configuration and dataset are valid
"""

import yaml
import pandas as pd
from pathlib import Path
import sys


def load_config():
    """Load and validate config.yaml exists"""
    config_path = Path("config.yaml")

    if not config_path.exists():
        print("[ERROR] config.yaml not found!")
        print("\nQuick fix:")
        print("   cp config.example.yaml config.yaml")
        print("   # Then edit config.yaml to match your dataset")
        return None

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("[OK] config.yaml loaded successfully")
        return config
    except yaml.YAMLError as e:
        print(f"[ERROR] Error parsing config.yaml: {e}")
        return None


def check_dataset(config):
    """Check dataset file exists and is readable"""
    dataset_config = config.get('dataset', {})

    # Check for single file
    if 'file_path' in dataset_config:
        file_path = Path(dataset_config['file_path'])
        if not file_path.exists():
            print(f"[ERROR] Dataset file not found: {file_path}")
            print(f"\nExpected location: {file_path.absolute()}")
            return None

        try:
            df = pd.read_csv(file_path, nrows=5)
            print(f"[OK] Dataset file found: {file_path}")
            return df
        except Exception as e:
            print(f"[ERROR] Error reading CSV: {e}")
            return None

    # Check for multiple files
    elif 'file_paths' in dataset_config:
        file_paths = [Path(fp) for fp in dataset_config['file_paths']]

        for fp in file_paths:
            if not fp.exists():
                print(f"[ERROR] Dataset file not found: {fp}")
                return None

        try:
            dfs = [pd.read_csv(fp, nrows=5) for fp in file_paths]
            df = pd.concat(dfs, ignore_index=True)
            print(f"[OK] All {len(file_paths)} dataset files found")
            return df
        except Exception as e:
            print(f"[ERROR] Error reading CSV files: {e}")
            return None

    else:
        print("[ERROR] config.yaml must specify 'file_path' or 'file_paths'")
        return None


def validate_columns(config, df):
    """Validate that all configured columns exist in dataset"""
    errors = []

    # Check text columns
    print("\nValidating text columns...")
    for text_col in config.get('text_columns', []):
        col_name = text_col['column']
        if col_name in df.columns:
            print(f"  [OK] '{col_name}' ({text_col['display_name']})")
        else:
            print(f"  [ERROR] '{col_name}' NOT FOUND")
            errors.append(f"Text column '{col_name}' not found in dataset")

    # Check metadata columns
    print("\nValidating metadata columns...")
    metadata = config.get('metadata', {})

    date_col = metadata.get('date_column')
    if date_col:
        if date_col in df.columns:
            print(f"  [OK] Date column: '{date_col}'")
            # Try parsing a sample date
            try:
                pd.to_datetime(df[date_col].iloc[0])
                print(f"       [OK] Date format is parseable")
            except:
                print(f"       [WARN] Date format may not be parseable by pandas")
        else:
            print(f"  [ERROR] Date column '{date_col}' NOT FOUND")
            errors.append(f"Date column '{date_col}' not found")

    score_col = metadata.get('score_column')
    if score_col:
        if score_col in df.columns:
            print(f"  [OK] Score column: '{score_col}'")
            # Check if numeric
            try:
                float(df[score_col].iloc[0])
                print(f"       [OK] Score column is numeric")
            except:
                print(f"       [WARN] Score column may not be numeric")
        else:
            print(f"  [ERROR] Score column '{score_col}' NOT FOUND")
            errors.append(f"Score column '{score_col}' not found")

    id_col = metadata.get('id_column')
    if id_col:
        if id_col in df.columns:
            print(f"  [OK] ID column: '{id_col}'")
        else:
            print(f"  [WARN] ID column '{id_col}' not found (optional)")

    # Check display columns
    display_cols = metadata.get('display_columns', [])
    if display_cols:
        print("\nValidating display columns...")
        for col in display_cols:
            if col in df.columns:
                print(f"  [OK] '{col}'")
            else:
                print(f"  [WARN] '{col}' not found (optional, but won't display)")

    return errors


def show_dataset_info(df):
    """Display dataset summary"""
    print("\n" + "="*60)
    print("Dataset Summary")
    print("="*60)
    print(f"Columns: {len(df.columns)}")
    print(f"Sample size (first 5 rows loaded)")
    print(f"\nAvailable columns:")
    for i, col in enumerate(df.columns, 1):
        sample_val = str(df[col].iloc[0])[:50]
        print(f"  {i}. {col}: {sample_val}...")


def main():
    print("Semantic Search Engine - Setup Verification")
    print("="*60)

    # Step 1: Load config
    config = load_config()
    if config is None:
        sys.exit(1)

    # Step 2: Check dataset
    print("\nChecking dataset...")
    df = check_dataset(config)
    if df is None:
        sys.exit(1)

    # Step 3: Validate columns
    errors = validate_columns(config, df)

    # Step 4: Show summary
    show_dataset_info(df)

    # Final verdict
    print("\n" + "="*60)
    if errors:
        print("[ERROR] Configuration has errors:")
        for error in errors:
            print(f"  - {error}")
        print("\nFix these issues in config.yaml, then run this script again")
        sys.exit(1)
    else:
        print("[OK] All checks passed! You're ready to run:")
        print("\n   python generate_embeddings.py")
        print("\nAfter embeddings are generated, run:")
        print("\n   streamlit run app.py")
        sys.exit(0)


if __name__ == "__main__":
    main()
