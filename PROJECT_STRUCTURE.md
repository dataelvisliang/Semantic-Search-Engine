# Project Structure

## Overview

This is a **generic semantic search engine** that works with any text dataset. Simply configure `config.yaml` to match your CSV structure, and the engine handles the rest.

## File Structure

```
SemanticSearchEngine/
├── app.py                      # Main Streamlit application (generic, config-driven)
├── generate_embeddings.py      # Embedding generation script (reads config)
├── verify_setup.py             # Setup validation tool (checks config + dataset)
├── config.yaml                 # Your actual configuration (customize this!)
├── config.example.yaml         # Example configurations for different use cases
├── requirements.txt            # Python dependencies
├── README.md                   # Comprehensive documentation
├── QUICKSTART.md               # 5-minute setup guide
├── LICENSE                     # MIT License
├── .gitignore                  # Git ignore patterns
├── .gitattributes              # Git LFS configuration for embeddings
├── data/                       # Your CSV files go here
│   └── .gitkeep
└── embeddings/                 # Generated embeddings (created by generate_embeddings.py)
    ├── [name]_embeddings.npz   # One file per text column
    └── metadata.pkl            # Metadata (dates, scores, text, etc.)
```

## Key Files Explained

### Core Application

- **`app.py`**: Generic Streamlit app that reads `config.yaml` and dynamically:
  - Loads embeddings for all configured text columns
  - Creates radio buttons for search target selection
  - Uses configured date/score columns for visualization
  - Displays configured metadata columns in results

- **`generate_embeddings.py`**: Config-driven embedding generator:
  - Reads `config.yaml` to know which columns to embed
  - Validates dataset structure
  - Generates separate `.npz` files for each text column
  - Saves all metadata to `metadata.pkl`

- **`verify_setup.py`**: Pre-flight check tool:
  - Validates `config.yaml` syntax
  - Checks dataset file exists
  - Verifies all configured columns exist in CSV
  - Shows helpful error messages

### Configuration

- **`config.yaml`**: **Your main configuration file** - customize this for your dataset
  - Specify CSV file path(s)
  - Define which text columns to embed
  - Set metadata columns (date, score, etc.)
  - Configure app title, defaults, etc.

- **`config.example.yaml`**: Example configurations for:
  - Amazon product reviews
  - Customer support tickets
  - Employee surveys
  - Social media comments

### Documentation

- **`README.md`**: Comprehensive guide covering:
  - What makes this universal
  - Quick start instructions
  - Full configuration reference
  - Use case examples
  - Deployment options
  - Troubleshooting

- **`QUICKSTART.md`**: 5-minute setup guide for getting started fast

- **`PROJECT_STRUCTURE.md`**: This file - explains the project layout

## Workflow

### 1. Setup Phase

```
User prepares CSV → Creates config.yaml → Runs verify_setup.py
```

**Tools**: `verify_setup.py` checks everything before embedding

### 2. Embedding Phase

```
Run generate_embeddings.py → Creates embeddings/ folder with:
  - [name]_embeddings.npz for each text column
  - metadata.pkl with all metadata
```

**One-time process** (10-30 minutes depending on dataset size)

### 3. Runtime Phase

```
Run app.py → Streamlit loads:
  - config.yaml (app settings)
  - embeddings/*.npz (pre-computed vectors)
  - embeddings/metadata.pkl (dates, scores, text)
```

**Instant startup** (~2-3 seconds to load embeddings)

## Configuration Flow

```
config.yaml
    ↓
    ├──> text_columns[] ──────> generate_embeddings.py
    │                              ↓
    │                         [name]_embeddings.npz (one per column)
    │
    ├──> metadata.date_column ──> Used for trend visualization
    ├──> metadata.score_column ──> Used for score distribution
    ├──> app.default_search_target ──> Which column to search by default
    └──> app.title ──────────────> App title in UI
```

## Extending the Engine

### Add New Text Columns

Just add to `config.yaml`:

```yaml
text_columns:
  - name: "new_field"
    display_name: "New Field"
    column: "ActualColumnName"
```

Then re-run `generate_embeddings.py`.

### Change Date/Score Columns

Update `config.yaml`:

```yaml
metadata:
  date_column: "DifferentDateColumn"
  score_column: "DifferentScoreColumn"
```

Then re-run `generate_embeddings.py` (need to regenerate metadata).

### Add Display Columns

Update `config.yaml`:

```yaml
metadata:
  display_columns:
    - "Category"
    - "Priority"
    - "Status"
```

Then re-run `generate_embeddings.py`.

## Deployment

### Local Development

```bash
streamlit run app.py
```

### Streamlit Cloud

1. Push to GitHub (with embeddings via Git LFS)
2. Deploy on share.streamlit.io

### Docker

```bash
docker build -t semantic-search .
docker run -p 8501:8501 semantic-search
```

## Technical Stack

- **Streamlit**: Web UI framework
- **sentence-transformers**: Embedding models (all-MiniLM-L6-v2)
- **BGE Reranker**: Cross-encoder for precise relevance scoring
- **Plotly**: Interactive visualizations
- **pandas**: Data processing
- **PyYAML**: Config file parsing

## Model Weights

Models auto-download on first run to `~/.cache/huggingface/`:

- `all-MiniLM-L6-v2`: ~80 MB (embedding model)
- `bge-reranker-base`: ~280 MB (reranker model)

**Total**: ~360 MB (one-time download)

## Data Flow

```
User Query
    ↓
Encode with sentence-transformers
    ↓
Cosine similarity vs all embeddings[selected_target]
    ↓
Top-K retrieval (fast, 100-20,000 candidates)
    ↓
BGE reranker scores (precise, 50-5,000 candidates)
    ↓
Filter by threshold (0.20-1.00)
    ↓
Filter by date range
    ↓
Results: Preview table + Visualizations + Details
```

## Performance Characteristics

- **Embedding generation**: ~1-2 minutes per 10,000 records per text column
- **App startup**: ~2-3 seconds (loading embeddings)
- **Search query**: ~3-15 seconds depending on top-k values
  - Cosine similarity: <1 second (100K+ docs)
  - BGE reranking: ~5-10 seconds (500-1,000 candidates on CPU)

## Next Steps

1. **Customize config.yaml** for your dataset
2. **Run verify_setup.py** to check everything
3. **Run generate_embeddings.py** to create embeddings
4. **Run streamlit run app.py** to launch the app
5. **Start searching!**

## Questions?

- See [README.md](README.md) for comprehensive documentation
- See [QUICKSTART.md](QUICKSTART.md) for 5-minute setup guide
- See [config.example.yaml](config.example.yaml) for example configurations
