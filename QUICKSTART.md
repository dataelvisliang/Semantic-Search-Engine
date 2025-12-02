# Quick Start Guide

Get your semantic search engine running in 5 minutes.

## Prerequisites

- Python 3.9 or higher
- A CSV file with text data

## Step-by-Step Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- Streamlit (web framework)
- sentence-transformers (embedding models)
- pandas, numpy, plotly (data processing & viz)
- PyYAML (config file parsing)

### 2. Prepare Your Data

Create a `data/` folder and place your CSV file there:

```bash
mkdir data
# Copy your CSV file to data/
cp /path/to/your/dataset.csv data/
```

**Your CSV must have at minimum:**
- ✅ At least one text column (e.g., "Comment", "Review", "Description")
- ✅ At least one date column (e.g., "Date", "CreatedAt", "Timestamp")
- ✅ At least one numeric column (e.g., "Rating", "Score", "Priority")

### 3. Create Configuration

Copy the example config and customize it:

```bash
cp config.example.yaml config.yaml
```

Edit `config.yaml` to match your dataset. **Minimal example:**

```yaml
dataset:
  file_path: "data/your_file.csv"

text_columns:
  - name: "review"
    display_name: "Review"
    column: "ReviewText"  # <-- Change to your actual column name

metadata:
  date_column: "CreatedDate"  # <-- Change to your date column
  score_column: "Rating"       # <-- Change to your score column

app:
  title: "My Semantic Search"
  page_icon: "🔍"
  default_search_target: "review"

embeddings:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  batch_size: 32
  output_dir: "embeddings"
```

### 4. Validate Configuration

Run the embedding script with `--help` to validate config (no actual embedding yet):

```bash
python generate_embeddings.py
```

If you see errors like `"Column 'XYZ' not found"`, check your `config.yaml` column names match your CSV exactly (case-sensitive!).

### 5. Generate Embeddings

Once validation passes, let the script complete:

```bash
python generate_embeddings.py
```

**Expected output:**
```
Loading configuration from config.yaml...
Loading dataset from: data/your_file.csv
Total records: 50000
✅ Configuration validated successfully

Loading embedding model: sentence-transformers/all-MiniLM-L6-v2

Generating embeddings for column 'ReviewText'...
Processed 50000/50000 items
Embeddings shape: (50000, 384)
Embeddings size: 73.24 MB

Saving embeddings...
✅ Embedding generation complete!

Generated embeddings:
  - review: embeddings/review_embeddings.npz (69.50 MB)

Metadata: embeddings/metadata.pkl (15.23 MB)

Total size: 84.73 MB

Next step: Run 'streamlit run app.py' to start the app!
```

**Time estimates:**
- 10,000 records: ~1-2 minutes
- 50,000 records: ~5-8 minutes
- 100,000 records: ~10-15 minutes
- 500,000 records: ~45-60 minutes

### 6. Launch the App

```bash
streamlit run app.py
```

Your browser will automatically open to `http://localhost:8501`

## First Search

1. **Enter a query** in plain English (e.g., "product arrived damaged")
2. **Select search target** (which text column to search)
3. **Adjust parameters** (optional - defaults work well)
4. **Click "Search"**

Results appear instantly with:
- Preview table of top matches
- Time trend visualization
- Relevance score distribution
- Expandable detail view by time period

## Troubleshooting

### "Column 'X' not found in dataset"

Check that column names in `config.yaml` **exactly match** your CSV headers (case-sensitive).

```bash
# View your CSV columns:
head -1 data/your_file.csv
```

### "Embeddings not found"

You need to run `generate_embeddings.py` first. The app won't work without embeddings.

### Models downloading very slowly

First-time run downloads ~100MB of models. This is one-time only. Models cache to `~/.cache/huggingface/`.

To use a different cache location:
```bash
export HF_HOME=/path/to/cache
python generate_embeddings.py
```

### Out of memory during embedding generation

Reduce `batch_size` in `config.yaml`:

```yaml
embeddings:
  batch_size: 16  # Lower from 32
```

Or process in chunks by splitting your CSV and using `file_paths` instead of `file_path`.

### App is slow

**During search:**
- Lower `top_k_retrieval` (default: 1000 → try 500)
- Lower `top_k_rerank` (default: 500 → try 200)

**On startup:**
- Embeddings load time scales with file size (~2-3 seconds for 100MB)
- Models load once and cache (5-10 seconds first time only)

## Next Steps

- **Customize visualizations**: Edit time granularity (day/month/year)
- **Export results**: Copy data from preview table
- **Try different search targets**: Switch between text columns
- **Adjust score thresholds**: Filter for higher precision
- **Explore trends**: Expand time periods to see top results

## Example Configurations

See `config.example.yaml` for:
- ✅ E-commerce product reviews
- ✅ Customer support tickets
- ✅ Employee surveys
- ✅ Social media comments

## Need Help?

Check the main [README.md](README.md) for:
- Full configuration reference
- Advanced options
- Deployment guides
- Use case examples
