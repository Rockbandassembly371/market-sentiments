#!/usr/bin/env python3
"""
Create corrected training data using taxonomy-derived labels.

This script:
1. Loads taxonomy YAML and extracts event sentiment from base_impact
2. Queries ClickHouse for headlines with matched events
3. Assigns corrected labels based on taxonomy sentiment
4. Creates train/val CSV files for model retraining
"""

import yaml
import pandas as pd
from clickhouse_driver import Client
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Label mapping for training
LABEL_TO_INT = {
    'negative': 0,
    'neutral': 1,
    'positive': 2
}


def load_taxonomy_sentiment(taxonomy_path: str) -> dict[str, str]:
    """
    Load taxonomy YAML and extract event_id -> sentiment mapping.
    
    Sentiment is derived from the sign of base_impact['normal']:
    - negative value → 'negative'
    - zero → 'neutral'
    - positive value → 'positive'
    """
    with open(taxonomy_path, 'r') as f:
        taxonomy = yaml.safe_load(f)
    
    event_sentiment = {}
    
    categories = taxonomy.get('categories', [])
    for cat in categories:
        subcategories = cat.get('subcategories', [])
        for subcat in subcategories:
            events = subcat.get('events', [])
            for event in events:
                event_id = event.get('id')
                base_impact = event.get('base_impact', {})
                
                # Get normal impact value
                normal_impact = base_impact.get('normal', 0)
                
                # Derive sentiment from impact sign
                if normal_impact < 0:
                    sentiment = 'negative'
                elif normal_impact > 0:
                    sentiment = 'positive'
                else:
                    sentiment = 'neutral'
                
                event_sentiment[event_id] = sentiment
    
    logger.info(f"Loaded {len(event_sentiment)} events from taxonomy")
    return event_sentiment


def query_clickhouse(limit: int = 1000000) -> pd.DataFrame:
    """
    Query ClickHouse for headlines with matched taxonomy events.
    
    Returns DataFrame with columns: headline, event_id, vader_label
    """
    client = Client(host='localhost', database='aion_master')
    
    query = f"""
    SELECT 
        n.title AS headline,
        t.event_id,
        n.sentiment_score
    FROM aion_master.news_master_v1 n
    LEFT JOIN aion_master.news_taxonomy_v1 t 
        ON n.row_hash = t.row_hash
    WHERE n.title != ''
      AND n.published_at >= '2024-01-01'
    ORDER BY n.published_at DESC
    LIMIT {limit}
    """
    
    logger.info(f"Executing query (limit={limit})...")
    result = client.execute(query)
    
    df = pd.DataFrame(result, columns=['headline', 'event_id', 'sentiment_score'])
    
    # Convert VADER sentiment_score to label for comparison
    def score_to_label(score):
        if score >= 0.05:
            return 'positive'
        elif score <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    df['vader_label'] = df['sentiment_score'].apply(score_to_label)
    
    logger.info(f"Retrieved {len(df)} headlines")
    logger.info(f"Headlines with event match: {df['event_id'].notna().sum()}")
    
    return df


def create_corrected_labels(df: pd.DataFrame, event_sentiment: dict) -> pd.DataFrame:
    """
    Assign corrected labels based on taxonomy sentiment.
    
    For headlines with matched events: use taxonomy-derived sentiment
    For headlines without matches: keep VADER label
    """
    def get_corrected_label(row):
        event_id = row['event_id']
        
        # If event matched, use taxonomy sentiment
        if pd.notna(event_id) and event_id in event_sentiment:
            return event_sentiment[event_id]
        
        # Otherwise, use VADER label
        return row['vader_label']
    
    df['corrected_label'] = df.apply(get_corrected_label, axis=1)
    
    # Convert to integer labels
    df['label'] = df['corrected_label'].map(LABEL_TO_INT)
    
    return df


def create_train_val_split(df: pd.DataFrame, val_ratio: float = 0.2, seed: int = 42):
    """
    Create train/val split and save to CSV files.
    """
    # Shuffle data
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Split
    val_size = int(len(df_shuffled) * val_ratio)
    val_df = df_shuffled.iloc[:val_size]
    train_df = df_shuffled.iloc[val_size:]
    
    # Select only required columns
    output_cols = ['headline', 'label']
    
    # Save to CSV
    output_dir = Path('aion-sentiment-in/data')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / 'train_corrected.csv'
    val_path = output_dir / 'val_corrected.csv'
    
    train_df[output_cols].to_csv(train_path, index=False)
    val_df[output_cols].to_csv(val_path, index=False)
    
    logger.info(f"Saved {len(train_df)} training samples to {train_path}")
    logger.info(f"Saved {len(val_df)} validation samples to {val_path}")
    
    # Print label distribution
    print("\nTraining label distribution:")
    print(train_df['label'].value_counts().sort_index())
    print("\nValidation label distribution:")
    print(val_df['label'].value_counts().sort_index())
    
    return train_df, val_df


def main():
    # Configuration
    taxonomy_path = 'aion_taxonomy/taxonomy_india_v2_calibrated.yaml'
    limit = 500000  # Start with 500K for testing
    
    logger.info("=" * 80)
    logger.info("Creating Corrected Training Data")
    logger.info("=" * 80)
    
    # Step 1: Load taxonomy sentiment
    logger.info("\nStep 1: Loading taxonomy sentiment mapping...")
    event_sentiment = load_taxonomy_sentiment(taxonomy_path)
    
    # Step 2: Query ClickHouse
    logger.info("\nStep 2: Querying ClickHouse for headlines...")
    df = query_clickhouse(limit=limit)
    
    # Step 3: Create corrected labels
    logger.info("\nStep 3: Creating corrected labels...")
    df = create_corrected_labels(df, event_sentiment)
    
    # Print comparison
    print("\nLabel comparison:")
    comparison = df.groupby(['vader_label', 'corrected_label']).size().unstack(fill_value=0)
    print(comparison)
    
    # Step 4: Create train/val split
    logger.info("\nStep 4: Creating train/val split...")
    train_df, val_df = create_train_val_split(df, val_ratio=0.2)
    
    logger.info("\n" + "=" * 80)
    logger.info("DONE - Ready for training")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
