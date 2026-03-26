#!/usr/bin/env python3
# =============================================================================
# AION Taxonomy - Calibration Script (with base_impact adjustment)
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
Calibrate Taxonomy Event Volatility and Base Impacts from Backfill Data.

This script:
1. Queries aion_master.news_taxonomy_v1 for event statistics
2. Updates event_volatility for events with count >= 30
3. Adjusts base_impact values if discrepancy > 0.1
4. Generates a calibrated taxonomy YAML
5. Creates no_match_events.txt list

Usage:
    python calibrate_taxonomy.py --taxonomy-path taxonomy_india_v2.yaml --min-count 30
"""

import argparse
import logging
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import clickhouse_connect
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("calibrate.log")]
)
logger = logging.getLogger(__name__)


def query_event_statistics(
    client,
    table: str = "aion_master.news_taxonomy_v1",
) -> pd.DataFrame:
    """
    Query event statistics from ClickHouse.

    Args:
        client: ClickHouse client.
        table: Source table name.

    Returns:
        DataFrame with event statistics.
    """
    query = f"""
    SELECT
        event_id,
        count() AS event_count,
        avg(macro_signal) AS avg_signal,
        stddevPop(macro_signal) AS event_volatility,
        avg(confidence) AS avg_confidence,
        quantile(0.1)(macro_signal) AS p10_signal,
        quantile(0.9)(macro_signal) AS p90_signal,
        min(macro_signal) AS min_signal,
        max(macro_signal) AS max_signal
    FROM {table}
    WHERE event_id != ''
    GROUP BY event_id
    ORDER BY event_count DESC
    """

    logger.info(f"Querying event statistics from {table}...")
    result = client.query(query)

    columns = [
        'event_id', 'event_count', 'avg_signal', 'event_volatility',
        'avg_confidence', 'p10_signal', 'p90_signal', 'min_signal', 'max_signal'
    ]

    df = pd.DataFrame(result.result_rows, columns=columns)
    logger.info(f"Queried statistics for {len(df)} events")

    return df


def load_taxonomy(yaml_path: str) -> Dict[str, Any]:
    """
    Load taxonomy YAML file.

    Args:
        yaml_path: Path to taxonomy YAML file.

    Returns:
        Taxonomy dictionary.
    """
    logger.info(f"Loading taxonomy from {yaml_path}...")
    with open(yaml_path, 'r', encoding='utf-8') as f:
        taxonomy = yaml.safe_load(f)
    logger.info(f"Loaded taxonomy version {taxonomy['metadata']['version']}")
    return taxonomy


def build_event_index(taxonomy: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Build an index of all events in the taxonomy for easy lookup.

    Args:
        taxonomy: Taxonomy dictionary.

    Returns:
        Dictionary mapping event_id to event dict and location info.
    """
    event_index = {}

    for cat_idx, category in enumerate(taxonomy.get('categories', [])):
        for subcat_idx, subcategory in enumerate(category.get('subcategories', [])):
            for event_idx, event in enumerate(subcategory.get('events', [])):
                event_id = event.get('id')
                if event_id:
                    event_index[event_id] = {
                        'event': event,
                        'category_idx': cat_idx,
                        'subcategory_idx': subcat_idx,
                        'event_idx': event_idx,
                        'category_id': category.get('id'),
                        'subcategory_id': subcategory.get('id'),
                    }

    logger.info(f"Built event index with {len(event_index)} events")
    return event_index


def scale_base_impact(
    base_impact: Dict[str, float],
    old_normal: float,
    new_normal: float,
) -> Dict[str, float]:
    """
    Scale base_impact values proportionally.

    Args:
        base_impact: Original base_impact dict.
        old_normal: Original normal impact value.
        new_normal: New normal impact value from observed data.

    Returns:
        Scaled base_impact dict.
    """
    if old_normal == 0:
        # Can't scale from zero, use observed value directly
        return {
            'mild': round(new_normal * 0.5, 3),
            'normal': round(new_normal, 3),
            'severe': round(new_normal * 1.5, 3),
        }

    # Calculate scaling ratio
    ratio = new_normal / old_normal

    # Scale all values
    scaled = {}
    for key, value in base_impact.items():
        scaled[key] = round(value * ratio, 3)

    # Ensure sign consistency
    for key in scaled:
        if scaled[key] == 0.0:
            scaled[key] = 0.0  # Keep zero as zero
        elif new_normal > 0 and scaled[key] < 0:
            scaled[key] = abs(scaled[key])
        elif new_normal < 0 and scaled[key] > 0:
            scaled[key] = -abs(scaled[key])

    return scaled


def calibrate_taxonomy(
    taxonomy: Dict[str, Any],
    stats_df: pd.DataFrame,
    min_count: int = 30,
    adjust_base_impact: bool = True,
    discrepancy_threshold: float = 0.1,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Update taxonomy with calibrated event_volatility and base_impact values.

    Args:
        taxonomy: Taxonomy dictionary.
        stats_df: DataFrame with event statistics.
        min_count: Minimum event count for reliable calibration.
        adjust_base_impact: Whether to adjust base_impact values.
        discrepancy_threshold: Threshold for base_impact adjustment.

    Returns:
        Tuple of (calibrated_taxonomy, calibration_metadata).
    """
    # Convert stats to dict for easy lookup
    stats_dict = stats_df.set_index('event_id').to_dict('index')

    # Build event index
    event_index = build_event_index(taxonomy)

    # Track calibration results
    calibration_meta = {
        'calibrated_at': datetime.now().isoformat(),
        'min_count_threshold': min_count,
        'discrepancy_threshold': discrepancy_threshold,
        'total_events': len(event_index),
        'calibrated_events': 0,
        'base_impact_adjusted': 0,
        'insufficient_data_events': 0,
        'no_match_events': 0,
        'no_match_event_ids': [],
        'adjustments': [],
    }

    # Update each event
    for event_id, info in event_index.items():
        event = info['event']

        if event_id in stats_dict:
            stats = stats_dict[event_id]

            if stats['event_count'] >= min_count:
                # Update event_volatility
                volatility = float(stats['event_volatility']) if stats['event_volatility'] else 0.0
                event['event_volatility'] = round(volatility, 3)
                calibration_meta['calibrated_events'] += 1

                # Check for base_impact adjustment
                base_impact = event.get('base_impact', {})
                normal_impact = base_impact.get('normal', 0.0)
                avg_signal = float(stats['avg_signal']) if stats['avg_signal'] else 0.0

                discrepancy = abs(avg_signal - normal_impact)

                if adjust_base_impact and discrepancy > discrepancy_threshold:
                    # Scale base_impact proportionally
                    new_base_impact = scale_base_impact(base_impact, normal_impact, avg_signal)
                    event['base_impact'] = new_base_impact
                    calibration_meta['base_impact_adjusted'] += 1

                    calibration_meta['adjustments'].append({
                        'event_id': event_id,
                        'old_normal': normal_impact,
                        'new_normal': avg_signal,
                        'old_base_impact': base_impact.copy(),
                        'new_base_impact': new_base_impact.copy(),
                        'event_count': stats['event_count'],
                    })

                    logger.debug(
                        f"Adjusted {event_id}: {normal_impact:.3f} -> {avg_signal:.3f} "
                        f"(ratio: {avg_signal/normal_impact if normal_impact != 0 else 'N/A':.3f})"
                    )
            else:
                # Insufficient data
                event['event_volatility'] = None
                calibration_meta['insufficient_data_events'] += 1
        else:
            # No match in backfill data
            event['event_volatility'] = None
            calibration_meta['no_match_events'] += 1
            calibration_meta['no_match_event_ids'].append(event_id)

    return taxonomy, calibration_meta


def save_taxonomy(taxonomy: Dict[str, Any], output_path: str) -> None:
    """
    Save calibrated taxonomy to YAML file.

    Args:
        taxonomy: Calibrated taxonomy dictionary.
        output_path: Output file path.
    """
    logger.info(f"Saving calibrated taxonomy to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(taxonomy, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    logger.info(f"Saved calibrated taxonomy")


def save_no_match_events(event_ids: List[str], output_path: str) -> None:
    """
    Save list of no-match events to text file.

    Args:
        event_ids: List of event IDs with no matches.
        output_path: Output file path.
    """
    logger.info(f"Saving no-match events to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("NO MATCH EVENTS - Not found in backfill data\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total: {len(event_ids)} events\n\n")
        for event_id in sorted(event_ids):
            f.write(f"  - {event_id}\n")
    logger.info(f"Saved {len(event_ids)} no-match events")


def generate_summary(
    calibration_meta: Dict[str, Any],
    output_path: str,
) -> None:
    """
    Generate final calibration summary.

    Args:
        calibration_meta: Calibration metadata.
        output_path: Output file path.
    """
    logger.info(f"Generating summary to {output_path}...")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("AION TAXONOMY - CALIBRATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Calibrated at: {calibration_meta['calibrated_at']}\n")
        f.write(f"Min count threshold: {calibration_meta['min_count_threshold']}\n")
        f.write(f"Discrepancy threshold: {calibration_meta['discrepancy_threshold']}\n\n")

        f.write("RESULTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total events in taxonomy: {calibration_meta['total_events']}\n")
        f.write(f"Events calibrated (count >= {calibration_meta['min_count_threshold']}): {calibration_meta['calibrated_events']}\n")
        f.write(f"Events with adjusted base impacts: {calibration_meta['base_impact_adjusted']}\n")
        f.write(f"Events still without matches: {calibration_meta['no_match_events']}\n\n")

        if calibration_meta['adjustments']:
            f.write("\n")
            f.write("BASE IMPACT ADJUSTMENTS\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Event ID':<40} {'Old Normal':>12} {'New Normal':>12} {'Ratio':>10} {'Count':>8}\n")
            f.write("-" * 80 + "\n")

            for adj in sorted(calibration_meta['adjustments'], key=lambda x: abs(x['new_normal'] - x['old_normal']), reverse=True):
                old_normal = adj['old_normal']
                new_normal = adj['new_normal']
                ratio = new_normal / old_normal if old_normal != 0 else 0
                f.write(f"{adj['event_id']:<40} {old_normal:>12.4f} {new_normal:>12.4f} {ratio:>10.3f} {adj['event_count']:>8}\n")

        f.write("\n")
        f.write("NEXT STEPS\n")
        f.write("-" * 80 + "\n")
        f.write("1. Review the calibrated taxonomy YAML\n")
        f.write("2. Consider adding keywords for no-match events\n")
        f.write("3. Optionally rerun backfill with body text included\n")

    logger.info(f"Generated summary")


def main():
    parser = argparse.ArgumentParser(description="Calibrate taxonomy event volatility and base impacts")
    parser.add_argument("--min-count", type=int, default=30, help="Minimum event count for reliable calibration")
    parser.add_argument("--discrepancy-threshold", type=float, default=0.1, help="Threshold for base_impact adjustment")
    parser.add_argument("--taxonomy-path", type=str, default="taxonomy_india_v2.yaml", help="Path to taxonomy YAML")
    parser.add_argument("--output-path", type=str, default="taxonomy_india_v2_calibrated.yaml", help="Output calibrated YAML path")
    parser.add_argument("--no-match-path", type=str, default="no_match_events.txt", help="Output no-match events path")
    parser.add_argument("--summary-path", type=str, default="calibration_summary.txt", help="Output summary path")
    parser.add_argument("--clickhouse-host", type=str, default="localhost", help="ClickHouse host")
    parser.add_argument("--clickhouse-port", type=int, default=8123, help="ClickHouse HTTP port")
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("AION TAXONOMY - CALIBRATION")
    logger.info("=" * 80)
    logger.info(f"Min count threshold: {args.min_count}")
    logger.info(f"Discrepancy threshold: {args.discrepancy_threshold}")
    logger.info(f"Taxonomy: {args.taxonomy_path}")
    logger.info(f"Output: {args.output_path}")
    logger.info("=" * 80)

    # Initialize ClickHouse client
    logger.info("Connecting to ClickHouse...")
    client = clickhouse_connect.get_client(
        host=args.clickhouse_host,
        port=args.clickhouse_port,
        database='aion_master'
    )
    logger.info("Connected to ClickHouse")

    # Query event statistics
    stats_df = query_event_statistics(client)

    if stats_df.empty:
        logger.warning("No event statistics found. Exiting.")
        return

    # Load taxonomy
    taxonomy = load_taxonomy(args.taxonomy_path)

    # Calibrate taxonomy
    logger.info(f"Calibrating taxonomy (min_count={args.min_count}, discrepancy_threshold={args.discrepancy_threshold})...")
    calibrated_taxonomy, calibration_meta = calibrate_taxonomy(
        taxonomy, stats_df,
        min_count=args.min_count,
        adjust_base_impact=True,
        discrepancy_threshold=args.discrepancy_threshold,
    )

    # Save calibrated taxonomy
    save_taxonomy(calibrated_taxonomy, args.output_path)

    # Save no-match events
    save_no_match_events(calibration_meta['no_match_event_ids'], args.no_match_path)

    # Generate summary
    generate_summary(calibration_meta, args.summary_path)

    # Log summary
    logger.info("\n" + "=" * 80)
    logger.info("CALIBRATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total events: {calibration_meta['total_events']}")
    logger.info(f"Calibrated: {calibration_meta['calibrated_events']}")
    logger.info(f"Base impact adjusted: {calibration_meta['base_impact_adjusted']}")
    logger.info(f"No match: {calibration_meta['no_match_events']}")
    logger.info("=" * 80)
    logger.info("CALIBRATION COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
