#!/usr/bin/env python3
"""
Data Preparation Script for AION Sentiment Model
Using VADER to generate correct sentiment labels from ClickHouse data.

Usage:
    python prepare_data_vader.py --host localhost --port 9000 --database aion_master \
        --table news_master_v1 --output_dir data --date_start 2024-01-01 --date_end 2026-03-31
"""

import argparse
import os
import logging
import pandas as pd
from clickhouse_driver import Client
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Extract news and label with VADER")
    parser.add_argument('--host', default='localhost', help='ClickHouse host')
    parser.add_argument('--port', type=int, default=9000, help='ClickHouse native port')
    parser.add_argument('--user', default='default', help='ClickHouse user')
    parser.add_argument('--password', default='', help='ClickHouse password')
    parser.add_argument('--database', default='aion_master', help='Database name')
    parser.add_argument('--table', default='news_master_v1', help='Table name')
    parser.add_argument('--output_dir', default='data', help='Directory to save CSVs')
    parser.add_argument('--date_start', default='2024-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--date_end', default='2026-03-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--limit', type=int, default=100000, help='Max rows to fetch')
    parser.add_argument('--test_size', type=float, default=0.2, help='Validation split size')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for split')
    return parser.parse_args()

def fetch_data(client, table, date_start, date_end, limit):
    """Fetch headlines from ClickHouse."""
    limit_clause = f"LIMIT {limit}" if limit and limit > 0 else ""
    query = f"""
        SELECT
            title AS headline,
            toDate(published_at) AS date
        FROM {table}
        WHERE published_at >= '{date_start}' AND published_at <= '{date_end}'
          AND title != ''
        {limit_clause}
    """
    logger.info(f"Executing query: {query}")
    result = client.execute(query, with_column_types=True)
    # result is list of tuples, column names from with_column_types
    columns = [col[0] for col in result[1]]
    data = result[0]
    df = pd.DataFrame(data, columns=columns)
    logger.info(f"Fetched {len(df)} rows")
    return df

def label_vader(df, text_col='headline'):
    """Apply VADER sentiment analysis to the dataframe."""
    analyzer = SentimentIntensityAnalyzer()
    logger.info("Applying VADER sentiment analysis...")
    sentiments = []
    labels = []
    scores = []
    for text in tqdm(df[text_col], desc="Processing headlines"):
        polarity = analyzer.polarity_scores(text)
        compound = polarity['compound']
        scores.append(compound)
        if compound >= 0.05:
            label = 'POS'
        elif compound <= -0.05:
            label = 'NEG'
        else:
            label = 'NEU'
        labels.append(label)
        sentiments.append(polarity)
    df['vader_score'] = scores
    df['vader_label'] = labels
    # Optional: store full sentiment dict as JSON
    df['vader_sentiment'] = sentiments
    return df

def save_splits(df, output_dir, test_size, random_state):
    """Split into train/val and save CSVs."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Map string labels to integers (AION standard encoding)
    label_map = {'POS': 2, 'NEU': 1, 'NEG': 0}
    df['label'] = df['vader_label'].map(label_map)
    
    # Use only headline and label for training
    train_df, val_df = train_test_split(df[['headline', 'label']],
                                        test_size=test_size,
                                        random_state=random_state,
                                        stratify=df['label'])
    
    train_path = os.path.join(output_dir, 'train.csv')
    val_path = os.path.join(output_dir, 'val.csv')
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    logger.info(f"Saved {len(train_df)} training samples to {train_path}")
    logger.info(f"Saved {len(val_df)} validation samples to {val_path}")

def main():
    args = parse_args()
    client = Client(host=args.host,
                    port=args.port,
                    user=args.user,
                    password=args.password,
                    database=args.database)
    df = fetch_data(client, args.table, args.date_start, args.date_end, args.limit)
    if df.empty:
        logger.error("No data fetched. Check date range or table.")
        sys.exit(1)
    df = label_vader(df)
    # Optional: inspect label distribution
    label_counts = df['vader_label'].value_counts()
    logger.info(f"Label distribution:\n{label_counts}")
    save_splits(df, args.output_dir, args.test_size, args.random_state)
    logger.info("Data preparation complete.")

if __name__ == "__main__":
    main()
