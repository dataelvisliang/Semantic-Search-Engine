"""
Semantic Search Engine - Generic text analytics with embeddings
Configurable via config.yaml
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import yaml
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any


# Load configuration
@st.cache_data
def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


# Initialize config first
try:
    CONFIG = load_config()
except FileNotFoundError:
    st.error("❌ config.yaml not found! Please create a config.yaml file in the app directory.")
    st.stop()

# Page config
st.set_page_config(
    page_title=CONFIG['app']['title'],
    page_icon=CONFIG['app']['page_icon'],
    layout="wide"
)

# Initialize session state
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = {}
if 'metadata' not in st.session_state:
    st.session_state.metadata = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'reranker' not in st.session_state:
    st.session_state.reranker = None
if 'preview_seed' not in st.session_state:
    st.session_state.preview_seed = 0


@st.cache_resource
def load_embedding_model(model_name: str):
    """Load the sentence transformer model"""
    return SentenceTransformer(model_name)


@st.cache_resource
def load_reranker_model():
    """Load the BGE reranker cross-encoder model"""
    return CrossEncoder('BAAI/bge-reranker-base')


@st.cache_data
def load_embeddings_and_metadata(config: Dict[str, Any]):
    """Load pre-computed embeddings and metadata based on config"""
    output_dir = Path(config['embeddings']['output_dir'])
    metadata_path = output_dir / 'metadata.pkl'

    if not metadata_path.exists():
        return None, None

    # Load all embedding files
    embeddings = {}
    for text_col in config['text_columns']:
        internal_name = text_col['name']
        emb_path = output_dir / f'{internal_name}_embeddings.npz'

        if not emb_path.exists():
            st.error(f"❌ Embedding file not found: {emb_path}")
            return None, None

        emb_data = np.load(emb_path)
        embeddings[internal_name] = emb_data['embeddings']

    # Load metadata
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)

    return embeddings, metadata


def rerank_with_bge(query: str, texts: list, reranker: CrossEncoder) -> list:
    """
    Use BGE reranker cross-encoder to score relevance
    """
    # Create query-text pairs
    pairs = [[query, text] for text in texts]

    # Get scores from the cross-encoder
    scores = reranker.predict(pairs)

    # Normalize scores to 0-1 range using sigmoid-like transformation
    normalized_scores = 1 / (1 + np.exp(-np.array(scores)))

    return normalized_scores.tolist()


def semantic_search(query_embedding, embeddings, top_k=500):
    """
    Perform cosine similarity search
    """
    similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    top_scores = similarities[top_indices]

    return top_indices, top_scores


def aggregate_by_period(df, date_column, period='month'):
    """
    Aggregate results by time period
    """
    df[date_column] = pd.to_datetime(df[date_column])

    if period == 'day':
        df['period'] = df[date_column].dt.strftime('%Y-%m-%d')
    elif period == 'month':
        df['period'] = df[date_column].dt.strftime('%Y-%m')
    else:  # year
        df['period'] = df[date_column].dt.strftime('%Y')

    # Count by period
    period_counts = df.groupby('period').size().reset_index(name='count')
    period_counts = period_counts.sort_values('period')

    return period_counts, df


def create_trend_chart(period_counts, period_type):
    """
    Create trend line/bar chart with improved styling
    """
    fig = go.Figure()

    # Add bar trace with gradient color
    fig.add_trace(go.Bar(
        x=period_counts['period'],
        y=period_counts['count'],
        name='Count',
        marker=dict(
            color=period_counts['count'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Count"),
            line=dict(color='rgba(255,255,255,0.3)', width=1.5)
        ),
        text=period_counts['count'],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
    ))

    fig.update_layout(
        title=dict(
            text=f"📈 Mentions Over Time ({period_type.capitalize()})",
            font=dict(size=20, color='#2c3e50', family="Arial Black")
        ),
        xaxis=dict(
            title=dict(text=period_type.capitalize(), font=dict(size=14, color='#34495e')),
            gridcolor='rgba(189, 195, 199, 0.3)',
            showgrid=True
        ),
        yaxis=dict(
            title=dict(text="Count", font=dict(size=14, color='#34495e')),
            gridcolor='rgba(189, 195, 199, 0.3)',
            showgrid=True
        ),
        plot_bgcolor='rgba(236, 240, 241, 0.4)',
        paper_bgcolor='white',
        hovermode='x unified',
        margin=dict(t=80, b=60, l=60, r=40)
    )

    return fig


def create_score_distribution_chart(scores):
    """
    Create pie chart for score distribution
    """
    # Define score ranges
    bins = [0.0, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = ['0.0-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
    score_ranges = pd.cut(scores, bins=bins, labels=labels, include_lowest=True)

    # Count distribution
    dist = score_ranges.value_counts().sort_index()

    fig = go.Figure(data=[go.Pie(
        labels=dist.index,
        values=dist.values,
        hole=0.4,
        marker=dict(
            colors=['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71', '#27ae60'],
            line=dict(color='white', width=2)
        ),
        textinfo='label+percent',
        textposition='outside',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])

    fig.update_layout(
        title=dict(
            text="🥧 Relevance Score Distribution",
            font=dict(size=20, color='#2c3e50', family="Arial Black")
        ),
        annotations=[dict(
            text=f'Total<br>{len(scores)}',
            x=0.5, y=0.5,
            font=dict(size=16, color='#34495e', family="Arial"),
            showarrow=False
        )],
        paper_bgcolor='white',
        margin=dict(t=80, b=40, l=40, r=40)
    )

    return fig


# Main app
def main():
    st.title(f"🔍 {CONFIG['app']['title']}")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Check for embeddings
        embeddings, metadata = load_embeddings_and_metadata(CONFIG)

        if embeddings is None or metadata is None:
            st.error("⚠️ Embeddings not found! Please run generate_embeddings.py first.")
            st.info("Run: `python generate_embeddings.py`")
            st.stop()

        # Store in session state
        st.session_state.embeddings = embeddings
        st.session_state.metadata = metadata

        # Load models
        if st.session_state.model is None:
            with st.spinner("Loading embedding model..."):
                st.session_state.model = load_embedding_model(CONFIG['embeddings']['model'])

        if st.session_state.reranker is None:
            with st.spinner("Loading reranker model..."):
                st.session_state.reranker = load_reranker_model()

        st.success(f"✅ Loaded {len(metadata[CONFIG['metadata']['date_column']])} records")

    # Main content area
    st.markdown("##### 🔍 Search")

    # Search input
    search_placeholder = CONFIG['app'].get('search_placeholder', 'Enter your search query...')
    search_query = st.text_input(
        "Enter search phrase:",
        placeholder=search_placeholder,
        help="Enter keywords or phrases to search for"
    )

    # Search target selector (dynamic based on config)
    search_target_options = [col['name'] for col in CONFIG['text_columns']]
    search_target_display = {col['name']: col['display_name'] for col in CONFIG['text_columns']}

    default_target = CONFIG['app'].get('default_search_target', search_target_options[0])

    selected_target = st.radio(
        "Search in:",
        options=search_target_options,
        format_func=lambda x: search_target_display[x],
        index=search_target_options.index(default_target) if default_target in search_target_options else 0,
        horizontal=True,
        help="Choose which text field to search in"
    )

    st.markdown("##### Search Parameters")
    col1, col2 = st.columns([1, 1])
    with col1:
        top_k_retrieval = st.slider(
            "🔍 Top-K for Cosine Similarity",
            min_value=100,
            max_value=20000,
            value=CONFIG['app'].get('default_top_k_retrieval', 1000),
            step=100,
            help="Number of candidates to retrieve using cosine similarity. Higher = more comprehensive but slower."
        )
    with col2:
        top_k_rerank = st.slider(
            "🎯 Top-K for Reranking",
            min_value=50,
            max_value=5000,
            value=CONFIG['app'].get('default_top_k_rerank', 500),
            step=50,
            help="Number of top candidates to rerank with BGE. Must be ≤ cosine similarity top-k. Higher = better quality but slower."
        )

    # Ensure rerank top_k doesn't exceed retrieval top_k
    if top_k_rerank > top_k_retrieval:
        st.warning(f"⚠️ Rerank Top-K ({top_k_rerank}) cannot exceed Retrieval Top-K ({top_k_retrieval}). Adjusting to {top_k_retrieval}.")
        top_k_rerank = top_k_retrieval

    st.markdown("##### Filters")
    col3, col4 = st.columns([1, 1])

    with col3:
        # Date range filter
        date_col = CONFIG['metadata']['date_column']
        all_dates = pd.to_datetime(st.session_state.metadata[date_col])
        min_date = all_dates.min().date()
        max_date = all_dates.max().date()

        date_range = st.date_input(
            "📅 Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            help="Filter results by date range"
        )

    with col4:
        # Score threshold
        score_threshold = st.slider(
            "Minimum Relevance Score",
            min_value=0.20,
            max_value=1.00,
            value=CONFIG['app'].get('default_score_threshold', 0.60),
            step=0.05,
            help="Only show results with relevance score greater than this threshold"
        )

    # Column selector for preview
    available_columns = []
    for text_col in CONFIG['text_columns']:
        available_columns.append(search_target_display[text_col['name']])

    # Add date and score columns from config
    date_col = CONFIG['metadata']['date_column']
    score_col = CONFIG['metadata']['score_column']
    available_columns.extend([date_col, score_col])

    selected_columns = st.multiselect(
        "Select columns to display:",
        options=available_columns,
        default=[search_target_display[selected_target], date_col, score_col][:2],
        help="Choose which columns to show in the preview table"
    )

    st.markdown("---")

    # Search button
    if st.button("🔍 Search", type="primary", use_container_width=True):
        if not search_query:
            st.warning("Please enter a search query")
            st.stop()

        # Step 1: Generate query embedding
        with st.spinner("Generating query embedding..."):
            query_embedding = st.session_state.model.encode(
                [search_query],
                normalize_embeddings=True
            )[0]

        # Step 2: Semantic search using selected target embeddings
        st.info(f"🔍 Searching in {search_target_display[selected_target]}...")

        selected_embeddings = st.session_state.embeddings[selected_target]

        with st.spinner(f"Finding top {top_k_retrieval} similar items..."):
            top_indices, cos_scores = semantic_search(
                query_embedding,
                selected_embeddings,
                top_k=top_k_retrieval
            )

        # Step 3: Select top candidates for reranking
        rerank_indices = top_indices[:top_k_rerank]

        st.info(f"🎯 Reranking top {len(rerank_indices)} candidates with BGE cross-encoder...")

        # Step 4: Rerank with BGE
        with st.spinner(f"Reranking {len(rerank_indices)} results with BGE cross-encoder..."):
            candidate_texts = [
                st.session_state.metadata[selected_target][idx]
                for idx in rerank_indices
            ]

            rerank_scores = rerank_with_bge(
                search_query,
                candidate_texts,
                st.session_state.reranker
            )

            rerank_scores = np.array(rerank_scores)

        # Step 5: Filter by threshold
        mask = rerank_scores >= score_threshold

        filtered_indices = rerank_indices[mask]
        filtered_scores = rerank_scores[mask]

        if len(filtered_indices) == 0:
            st.warning(f"No results found with relevance score > {score_threshold:.2f}. Try lowering the score threshold or a different query.")
            st.stop()

        st.success(f"✅ Found {len(filtered_indices)} highly relevant results (score > {score_threshold:.2f})")

        # Step 6: Prepare results dataframe
        date_col = CONFIG['metadata']['date_column']
        score_col = CONFIG['metadata']['score_column']

        results_data = {
            'index': filtered_indices,
            'relevance_score': filtered_scores,
            date_col: st.session_state.metadata[date_col][filtered_indices],
            score_col: st.session_state.metadata[score_col][filtered_indices]
        }

        # Add all text columns
        for text_col in CONFIG['text_columns']:
            internal_name = text_col['name']
            display_name = text_col['display_name']
            results_data[display_name] = st.session_state.metadata[internal_name][filtered_indices]

        results_df = pd.DataFrame(results_data)

        # Step 7: Date range filtering
        if len(date_range) == 2:
            start_date, end_date = date_range
            results_df[date_col] = pd.to_datetime(results_df[date_col])
            results_df = results_df[
                (results_df[date_col].dt.date >= start_date) &
                (results_df[date_col].dt.date <= end_date)
            ]

        if len(results_df) == 0:
            st.warning("No results found in the selected date range")
            st.stop()

        # Step 8: Preview results
        st.markdown("---")
        st.subheader(f"📋 Preview Results (Top {min(20, len(results_df))} of {len(results_df)})")

        if selected_columns:
            preview_df = results_df[selected_columns].head(20).copy()

            # Configure columns using actual column names from config
            date_col = CONFIG['metadata']['date_column']
            score_col = CONFIG['metadata']['score_column']

            column_config = {}
            if date_col in selected_columns:
                column_config[date_col] = st.column_config.DateColumn(date_col, format="YYYY-MM-DD", width="small")
            if score_col in selected_columns:
                column_config[score_col] = st.column_config.NumberColumn(score_col, format="%.2f", width="small")

            st.dataframe(
                preview_df,
                use_container_width=True,
                hide_index=True,
                column_config=column_config
            )
        else:
            st.info("Please select at least one column to display")

        # Step 9: Visualization options
        st.markdown("---")
        st.subheader("📊 Visualizations")

        viz_col1, viz_col2 = st.columns([1, 1])

        with viz_col1:
            period_type = st.selectbox(
                "Time Granularity",
                options=['month', 'year', 'day'],
                help="Choose the time period for trend aggregation"
            )

        with viz_col2:
            chart_type = st.selectbox(
                "Chart Type",
                options=['trend', 'distribution', 'both'],
                help="Choose visualization type"
            )

        # Generate visualizations
        if chart_type in ['trend', 'both']:
            date_col = CONFIG['metadata']['date_column']
            period_counts, results_with_period = aggregate_by_period(results_df.copy(), date_col, period=period_type)

            st.plotly_chart(
                create_trend_chart(period_counts, period_type),
                use_container_width=True
            )

        if chart_type in ['distribution', 'both']:
            st.plotly_chart(
                create_score_distribution_chart(results_df['relevance_score'].values),
                use_container_width=True
            )

        # Step 10: Detailed results by period
        if chart_type in ['trend', 'both']:
            st.markdown("---")
            st.subheader(f"📝 Top Results by {period_type.capitalize()}")

            date_col = CONFIG['metadata']['date_column']
            score_col = CONFIG['metadata']['score_column']

            for period in period_counts['period'].values[::-1]:
                period_data = results_with_period[results_with_period['period'] == period]
                period_sorted = period_data.nlargest(10, 'relevance_score')

                with st.expander(f"📅 {period} ({len(period_data)} results)"):
                    for idx, row in period_sorted.iterrows():
                        col_a, col_b = st.columns([4, 1])

                        with col_a:
                            # Display the selected search target text
                            text_content = row[search_target_display[selected_target]]
                            st.markdown(f"**{text_content}**")
                            st.caption(f"{date_col}: {row[date_col]}")

                        with col_b:
                            score_color = 'green' if row['relevance_score'] >= 0.8 else 'orange' if row['relevance_score'] >= 0.6 else 'red'
                            st.markdown(f"Score: :{score_color}[**{row['relevance_score']:.3f}**]")
                            st.caption(f"{score_col}: {row[score_col]}")

                        st.markdown("---")


if __name__ == "__main__":
    main()
