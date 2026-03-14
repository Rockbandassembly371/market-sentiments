-- =============================================================================
-- AION Open-Source Project: Sentiment Analysis Pipeline
-- File: extract_data.sql
-- Description: SQL query to extract news sentiment data for model training
-- License: Apache License, Version 2.0
--
-- Copyright 2026 AION Contributors
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.
--
-- AION Attribution: This file is part of the AION open-source ecosystem.
-- Please visit https://github.com/aion for more information.
-- =============================================================================

-- =============================================================================
-- DATA EXTRACTION QUERY
-- =============================================================================
-- Source Database: ClickHouse
-- Source Table: aion_master.news_master_v1 (957K+ rows with sentiment)
-- Date Range: 2024-01-01 onwards (or all available data)
-- Row Limit: 10000 rows for initial training (balanced sentiment distribution)
--
-- SENTIMENT SCORING METHODOLOGY:
--   The source data uses AION's UNIFIED_ROUTER_V4 classification system.
--   For OSS, we extract the labels and train our own transformer model.
--
-- Label Distribution (approx):
--   NEG: 129K rows (avg_score: -0.70, confidence: 99.99%)
--   NEU: 476K rows (avg_score: 0.00, confidence: 99.99%)
--   POS: 350K rows (avg_score: +0.74, confidence: 99.99%)
--
-- Columns extracted:
--   - title: News headline for sentiment analysis
--   - published_at: Publication timestamp
--   - sentiment_label: NEG/NEU/POS classification (target variable)
--   - classification_confidence: Confidence score (0-100)
--   - tickers: Mentioned stock tickers (array)
--   - sentiment_score: Raw sentiment score (-1 to 1)
--   - source: News source
--   - relevance: Relevance score (0-100)
-- =============================================================================

-- SAMPLE EXTRACTION (10K rows for initial training)
SELECT
    title AS headline,
    published_at AS publish_date,
    sentiment_label,
    classification_confidence AS confidence_score,
    arrayStringConcat(tickers, ',') AS ticker,
    sentiment_score,
    source,
    relevance
FROM
    aion_master.news_master_v1
WHERE
    sentiment_label IS NOT NULL
    AND published_at >= '2024-01-01'
ORDER BY
    published_at DESC
LIMIT 10000;

-- =============================================================================
-- FULL DATASET EXTRACTION (957K+ rows - for production training)
-- =============================================================================
-- Uncomment below for full dataset extraction:
--
-- SELECT
--     title AS headline,
--     published_at AS publish_date,
--     sentiment_label,
--     classification_confidence AS confidence_score,
--     arrayStringConcat(tickers, ',') AS ticker,
--     sentiment_score,
--     source,
--     relevance
-- FROM
--     aion_master.news_master_v1
-- WHERE
--     sentiment_label IS NOT NULL
--     AND published_at >= '2024-01-01'
-- ORDER BY
--     published_at DESC;
-- =============================================================================

-- =============================================================================
-- BALANCED DATASET EXTRACTION (equal samples per sentiment class)
-- =============================================================================
-- For balanced training (equal NEG/NEU/POS samples):
--
-- WITH sentiment_samples AS (
--     SELECT
--         title AS headline,
--         published_at AS publish_date,
--         sentiment_label,
--         classification_confidence AS confidence_score,
--         arrayStringConcat(tickers, ',') AS ticker,
--         sentiment_score,
--         source,
--         relevance,
--         row_number() OVER (PARTITION BY sentiment_label ORDER BY published_at DESC) as rn
--     FROM aion_master.news_master_v1
--     WHERE sentiment_label IS NOT NULL AND published_at >= '2024-01-01'
-- )
-- SELECT
--     headline, publish_date, sentiment_label, confidence_score,
--     ticker, sentiment_score, source, relevance
-- FROM sentiment_samples
-- WHERE rn <= 3000  -- 3000 per class = 9000 total
-- ORDER BY sentiment_label, published_at DESC;
-- =============================================================================

-- =============================================================================
-- EXECUTION INSTRUCTIONS
-- =============================================================================
-- Execute and export to CSV using clickhouse-client:
--
-- # Sample dataset (10K rows):
-- clickhouse-client --query "$(cat extract_data.sql)" --format=CSVWithNames > data/raw_extracted.csv
--
-- # Full dataset:
-- clickhouse-client --query "
--   SELECT title, published_at, sentiment_label, classification_confidence,
--          arrayStringConcat(tickers, ',') AS ticker, sentiment_score, source, relevance
--   FROM aion_master.news_master_v1
--   WHERE sentiment_label IS NOT NULL AND published_at >= '2024-01-01'
--   ORDER BY published_at DESC
-- " --format=CSVWithNames > data/raw_extracted.csv
--
-- # Balanced dataset (3K per class):
-- clickhouse-client --query_file balanced_extract.sql --format=CSVWithNames > data/raw_extracted.csv
-- =============================================================================
