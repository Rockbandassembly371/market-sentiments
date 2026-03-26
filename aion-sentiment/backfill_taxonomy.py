#!/usr/bin/env python3
# =============================================================================
# AION Taxonomy - Backfill Script
# =============================================================================
# Copyright (c) 2026 AION Contributors
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""
Backfill Taxonomy Classification for Historical Headlines.

This script:
1. Fetches headlines from ClickHouse (aion_master.news_master_v1)
2. Classifies each headline using the taxonomy pipeline
3. Stores classification results in aion_master.news_taxonomy_v1

Usage:
    python backfill_taxonomy.py --limit 200000 --batch-size 1000
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import clickhouse_connect
import pandas as pd

# Add taxonomy to path
sys.path.insert(0, 'aion_taxonomy/src')

from aion_taxonomy import TaxonomyPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("backfill.log")]
)
logger = logging.getLogger(__name__)


def fetch_headlines(
    client,
    limit: int = 200000,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    use_body: bool = False,
) -> List[Dict[str, Any]]:
    """
    Fetch headlines from ClickHouse.

    Only fetches columns without existing sentiment data:
    - row_hash, title, published_at

    Args:
        client: ClickHouse client.
        limit: Maximum number of rows to fetch.
        date_from: Start date (YYYY-MM-DD).
        date_to: End date (YYYY-MM-DD).
        use_body: If True, also fetch body text and concatenate with title.

    Returns:
        List of dicts with row_hash, headline, published_at.
    """
    date_filter = ""
    if date_from:
        date_filter += f" AND toDate(published_at) >= '{date_from}'"
    if date_to:
        date_filter += f" AND toDate(published_at) <= '{date_to}'"

    if use_body:
        query = f"""
        SELECT
            row_hash,
            title,
            body,
            published_at
        FROM aion_master.news_master_v1
        WHERE title != ''
        {date_filter}
        ORDER BY published_at DESC
        LIMIT {limit}
        """
    else:
        query = f"""
        SELECT
            row_hash,
            title,
            published_at
        FROM aion_master.news_master_v1
        WHERE title != ''
        {date_filter}
        ORDER BY published_at DESC
        LIMIT {limit}
        """

    logger.info(f"Fetching headlines: limit={limit}, from={date_from}, to={date_to}, use_body={use_body}")
    result = client.query(query)

    headlines = []
    for row in result.result_rows:
        if use_body:
            # Concatenate title and body for matching
            title = row[1] or ''
            body = row[2] or ''
            headline = f"{title} {body}".strip()
            if not headline:
                continue
        else:
            headline = row[1]
        
        headlines.append({
            'row_hash': row[0],
            'headline': headline,
            'published_at': row[3] if use_body else row[2]
        })

    logger.info(f"Fetched {len(headlines)} headlines")
    return headlines


def classify_headlines(
    pipeline: TaxonomyPipeline,
    headlines: List[Dict[str, Any]],
    batch_size: int = 100,
) -> List[Dict[str, Any]]:
    """
    Classify headlines using taxonomy pipeline.

    Args:
        pipeline: TaxonomyPipeline instance.
        headlines: List of headline dicts.
        batch_size: Batch size for processing.

    Returns:
        List of classification results.
    """
    results = []
    total = len(headlines)

    for i in range(0, total, batch_size):
        batch = headlines[i:i+batch_size]

        for item in batch:
            result = pipeline.process(
                headline=item['headline'],
                date=item['published_at'].isoformat() if item['published_at'] else None,
            )
            result['row_hash'] = item['row_hash']
            result['published_at'] = item['published_at']
            results.append(result)

        # Progress logging
        if (i // batch_size) % 10 == 0:
            progress = min(i + batch_size, total) / total * 100
            logger.info(f"Progress: {progress:.1f}% ({min(i + batch_size, total)}/{total})")

    return results


def store_results(
    client,
    results: List[Dict[str, Any]],
    table: str = "aion_master.news_taxonomy_v1",
    batch_size: int = 5000,
) -> int:
    """
    Store classification results to ClickHouse.

    Args:
        client: ClickHouse client.
        results: List of classification results.
        table: Target table name.
        batch_size: Insert batch size.

    Returns:
        Number of rows inserted.
    """
    total_inserted = 0

    for i in range(0, len(results), batch_size):
        batch = results[i:i+batch_size]
        
        # Prepare values for insertion
        values = []
        for r in batch:
            event = r.get('event', {})
            sector_impacts = json.dumps(r.get('sector_signals', {}))
            
            row = (
                r.get('row_hash', ''),
                r.get('published_at'),
                event.get('event_id') or '',
                event.get('event_name') or None,
                event.get('category_id') or None,
                event.get('subcategory_id') or None,
                float(event.get('match_score', 0.0)),
                ','.join(event.get('matched_keywords', [])),
                r.get('impact_level', 'normal'),
                float(r.get('macro_signal', 0.0)),
                float(r.get('confidence', 0.0)),
                sector_impacts,
                datetime.now(),
                '2.0.0',
            )
            values.append(row)

        # Insert batch
        client.insert(table, values)
        total_inserted += len(batch)
        logger.debug(f"Inserted batch {i//batch_size + 1}: {len(batch)} rows")

    logger.info(f"Stored {total_inserted} classification results to {table}")
    return total_inserted


def main():
    parser = argparse.ArgumentParser(description="Backfill taxonomy classification")
    parser.add_argument("--limit", type=int, default=200000, help="Number of headlines to process")
    parser.add_argument("--batch-size", type=int, default=1000, help="Classification batch size")
    parser.add_argument("--insert-batch-size", type=int, default=5000, help="Insert batch size")
    parser.add_argument("--date-from", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--date-to", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--taxonomy-path", type=str, default="taxonomy_india_v2.yaml",
                        help="Path to taxonomy YAML file")
    parser.add_argument("--clickhouse-host", type=str, default="localhost", help="ClickHouse host")
    parser.add_argument("--clickhouse-port", type=int, default=8123, help="ClickHouse HTTP port")
    parser.add_argument("--dry-run", action="store_true", help="Don't store results, just classify")
    parser.add_argument("--use-body", action="store_true", help="Use body text concatenated with title for matching")
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("AION TAXONOMY - BACKFILL CLASSIFICATION")
    logger.info("=" * 80)
    logger.info(f"Limit: {args.limit}")
    logger.info(f"Classification batch size: {args.batch_size}")
    logger.info(f"Insert batch size: {args.insert_batch_size}")
    logger.info(f"Date range: {args.date_from} to {args.date_to}")
    logger.info(f"Taxonomy: {args.taxonomy_path}")
    logger.info(f"Use body text: {args.use_body}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info("=" * 80)

    # Initialize ClickHouse client
    logger.info("Connecting to ClickHouse...")
    client = clickhouse_connect.get_client(
        host=args.clickhouse_host,
        port=args.clickhouse_port,
        database='aion_master'
    )
    logger.info("Connected to ClickHouse")

    # Initialize taxonomy pipeline
    logger.info(f"Loading taxonomy from {args.taxonomy_path}...")
    pipeline = TaxonomyPipeline(taxonomy_path=args.taxonomy_path)
    logger.info(f"Loaded {len(pipeline.list_events())} events")

    # Fetch headlines
    headlines = fetch_headlines(
        client,
        limit=args.limit,
        date_from=args.date_from,
        date_to=args.date_to,
        use_body=args.use_body,
    )

    if not headlines:
        logger.warning("No headlines fetched. Exiting.")
        return

    # Classify headlines
    logger.info(f"Classifying {len(headlines)} headlines...")
    start_time = time.time()

    results = classify_headlines(pipeline, headlines, batch_size=args.batch_size)

    elapsed = time.time() - start_time
    logger.info(f"Classification complete in {elapsed:.1f}s ({len(headlines)/elapsed*60:.1f} headlines/min)")

    # Store results
    if not args.dry_run:
        inserted = store_results(client, results, batch_size=args.insert_batch_size)
        logger.info(f"Stored {inserted} classification results to ClickHouse")
    else:
        logger.info("Dry run - results not stored")

    # Summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 80)

    event_counts = {}
    for r in results:
        event_id = r.get('event', {}).get('event_id') or 'NO_MATCH'
        event_counts[event_id] = event_counts.get(event_id, 0) + 1

    # Top 20 events
    sorted_events = sorted(event_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    logger.info("\nTop 20 events:")
    for event_id, count in sorted_events:
        logger.info(f"  {event_id}: {count} ({count/len(results)*100:.1f}%)")

    # Match rate
    matched = sum(1 for r in results if r.get('event', {}).get('event_id'))
    logger.info(f"\nMatch rate: {matched}/{len(results)} ({matched/len(results)*100:.1f}%)")

    # Average confidence
    avg_confidence = sum(r.get('confidence', 0) for r in results) / len(results)
    logger.info(f"Average confidence: {avg_confidence:.3f}")

    # Average macro signal
    avg_signal = sum(r.get('macro_signal', 0) for r in results) / len(results)
    logger.info(f"Average macro signal: {avg_signal:+.3f}")

    # Signal distribution
    positive = sum(1 for r in results if r.get('macro_signal', 0) > 0.1)
    negative = sum(1 for r in results if r.get('macro_signal', 0) < -0.1)
    neutral = len(results) - positive - negative
    logger.info(f"\nSignal distribution:")
    logger.info(f"  Positive: {positive} ({positive/len(results)*100:.1f}%)")
    logger.info(f"  Neutral: {neutral} ({neutral/len(results)*100:.1f}%)")
    logger.info(f"  Negative: {negative} ({negative/len(results)*100:.1f}%)")

    logger.info("\n" + "=" * 80)
    logger.info("BACKFILL COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
