#!/usr/bin/env python3
# =============================================================================
# AION Open-Source Project: Sample Data Generator
# File: create_sample_data.py
# Description: Creates sample training data for AION-Sentiment-IN model
# License: Apache License, Version 2.0
#
# Copyright 2026 AION Contributors
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
#
# AION Attribution: This file is part of the AION open-source ecosystem.
# Please visit https://github.com/aion for more information.
# =============================================================================
"""
Sample Data Generator for AION Sentiment Analysis.

This script generates synthetic training data for testing the training pipeline.
In production, you would use extract_data.sql and prepare_data.py to create
real data from your ClickHouse database.

Generates:
    - data/processed/train.csv (8000 samples)
    - data/processed/val.csv (2000 samples)

Each sample contains:
    - headline: Financial news headline text
    - label: Sentiment label (0=negative, 1=neutral, 2=positive)
"""

import os
import random
from pathlib import Path

# Sample financial headlines by sentiment category
POSITIVE_HEADLINES = [
    "Stock market reaches all-time high amid economic optimism",
    "Tech stocks surge as earnings beat expectations",
    "Fed announces rate cut, markets rally",
    "Unemployment drops to record low, investors celebrate",
    "Oil prices stabilize after OPEC agreement",
    "Banking sector shows strong Q4 results",
    "Retail sales exceed forecasts, consumer confidence up",
    "Manufacturing index hits 5-year high",
    "GDP growth surpasses analyst expectations",
    "Inflation concerns ease as prices stabilize",
    "Housing market shows signs of recovery",
    "Corporate profits rise despite global challenges",
    "Investor sentiment improves on trade deal news",
    "Small business optimism index climbs",
    "Export growth drives economic expansion",
    "Stock futures point to higher open after positive data",
    "Bull market continues historic run",
    "Dividend announcements boost shareholder value",
    "IPO market heats up with successful debuts",
    "Mergers and acquisitions activity picks up",
    "Renewable energy stocks soar on policy support",
    "Healthcare sector innovation drives growth",
    "AI companies attract record investment",
    "Electric vehicle sales accelerate globally",
    "Cloud computing revenue exceeds projections",
    "Semiconductor demand remains strong",
    "E-commerce growth continues post-pandemic",
    "Travel and leisure stocks rebound sharply",
    "Restaurant chains report strong same-store sales",
    "Luxury goods market expands in emerging economies",
    "Infrastructure spending bill passes, construction stocks rise",
    "Defense contractors secure major government deals",
    "Pharmaceutical breakthrough sends biotech stocks higher",
    "Fintech disruption creates new investment opportunities",
    "Cryptocurrency adoption grows among institutional investors",
    "Real estate investment trusts outperform benchmarks",
    "Agricultural commodities gain on supply concerns",
    "Gold prices rise as safe-haven demand increases",
    "Dollar strengthens against major currencies",
    "Emerging markets attract foreign capital inflows",
    "Bond yields climb on growth optimism",
    "Credit spreads tighten as default risks decline",
    "Insurance sector benefits from rate increases",
    "Utility stocks provide stable dividends",
    "Telecom sector expands 5G infrastructure",
    "Media streaming services add millions of subscribers",
    "Gaming industry revenue hits new record",
    "Sports betting legalization boosts casino stocks",
    "Cannabis stocks rally on regulatory progress",
    "Space exploration companies secure funding",
]

NEUTRAL_HEADLINES = [
    "Markets close mixed as investors await economic data",
    "Stock index fluctuates in narrow trading range",
    "Analysts maintain neutral outlook on sector",
    "Trading volume remains below average",
    "Currency pairs trade sideways ahead of central bank meeting",
    "Commodity prices show little movement",
    "Bond market signals uncertainty on rate path",
    "Investors weigh competing economic indicators",
    "Market volatility index holds steady",
    "Sector rotation continues without clear direction",
    "Earnings season results in line with expectations",
    "Economic data releases meet forecasts",
    "Central bank maintains dovish stance",
    "Treasury yields unchanged after auction",
    "Credit markets remain stable",
    "Equity flows balance between value and growth",
    "International markets show mixed performance",
    "Emerging market currencies hold ground",
    "Oil inventory data shows modest build",
    "Gold holds near key technical level",
    "Silver trades in familiar range",
    "Copper prices reflect balanced supply-demand",
    "Agricultural markets await weather reports",
    "Retail investors remain on sidelines",
    "Institutional positioning shows no clear bias",
    "Hedge fund sentiment neutral on equities",
    "Pension funds rebalance portfolios",
    "Endowments maintain strategic allocation",
    "Sovereign wealth funds diversify holdings",
    "Family offices increase alternative investments",
    "Private equity dry powder reaches record levels",
    "Venture capital funding stabilizes",
    "Real estate cap rates hold steady",
    "Infrastructure assets attract long-term capital",
    "Commodity index funds see modest inflows",
    "ETF flows balanced across asset classes",
    "Mutual fund redemptions offset new purchases",
    "Brokerage accounts show mixed positioning",
    "Robo-advisors maintain model portfolios",
    "Wealth managers recommend balanced approach",
    "Market breadth indicators show neutral signal",
    "Technical analysis suggests consolidation phase",
    "Fundamental analysts see fair valuation",
    "Quantitative models indicate hold rating",
    "Sentiment surveys show divided opinion",
    "Economist forecasts cluster around consensus",
    "Survey data reveals mixed business conditions",
    "Purchasing managers index unchanged",
    "Consumer confidence holds at current level",
    "Business sentiment remains stable",
]

NEGATIVE_HEADLINES = [
    "Stock market tumbles on recession fears",
    "Tech stocks lead broad market selloff",
    "Fed signals aggressive rate hikes, markets plunge",
    "Unemployment claims surge to multi-year high",
    "Oil prices crash on demand concerns",
    "Banking crisis spreads across financial sector",
    "Retail sales disappoint, consumer spending weakens",
    "Manufacturing contracts for third consecutive month",
    "GDP shrinks more than expected",
    "Inflation spikes to 40-year high, stocks fall",
    "Housing market freezes as mortgage rates soar",
    "Corporate earnings miss estimates widely",
    "Trade tensions escalate, global markets decline",
    "Small business confidence plummets",
    "Export orders collapse amid global slowdown",
    "Stock futures point to lower open on negative data",
    "Bear market confirmed as index drops 20%",
    "Dividend cuts spread across multiple sectors",
    "IPO market freezes as valuations collapse",
    "Merger deals abandoned on financing concerns",
    "Energy sector battered by price war",
    "Healthcare stocks pressured by policy uncertainty",
    "AI hype fades as profits fail to materialize",
    "Electric vehicle demand slows sharply",
    "Cloud computing growth decelerates",
    "Chip shortage worsens, auto production halts",
    "E-commerce growth stalls, layoffs announced",
    "Travel restrictions hurt airline stocks",
    "Restaurant closures accelerate amid labor shortage",
    "Luxury goods demand weakens in key markets",
    "Infrastructure projects delayed on budget concerns",
    "Defense spending cuts hit contractor stocks",
    "Drug pricing reform pressures pharma sector",
    "Fintech valuations collapse on profitability concerns",
    "Cryptocurrency crash wipes out billions",
    "REITs decline on property market weakness",
    "Agricultural prices fall on bumper harvest",
    "Gold loses safe-haven appeal",
    "Dollar weakness hurts export competitiveness",
    "Capital flight from emerging markets accelerates",
    "Bond yields invert, recession signal flashes",
    "Credit spreads widen on default fears",
    "Insurance losses mount from natural disasters",
    "Utility sector pressured by rising costs",
    "Telecom competition intensifies, margins shrink",
    "Media stocks decline on cord-cutting acceleration",
    "Gaming revenue misses on mobile weakness",
    "Sports betting adoption slower than expected",
    "Cannabis sector faces regulatory headwinds",
    "Space companies struggle with profitability",
    "Supply chain disruptions continue to plague retailers",
    "Inventory glut forces deep discounting",
    "Consumer debt levels reach concerning highs",
    "Credit card delinquencies tick higher",
    "Auto loan defaults increase sharply",
    "Student debt burden weighs on economy",
    "Housing affordability hits record low",
    "Rent burdens squeeze household budgets",
    "Wage growth fails to keep pace with inflation",
    "Productivity growth remains sluggish",
    "Income inequality widens further",
    "Poverty rates increase in urban areas",
    "Regional banks face deposit outflows",
    "Commercial real estate vacancy rates climb",
    "Office buildings struggle post-pandemic",
    "Shopping mall traffic continues decline",
    "Restaurant bankruptcies accelerate",
    "Airline capacity cuts announced on weak demand",
    "Cruise lines suspend operations on health concerns",
    "Hotel occupancy rates remain depressed",
    "Entertainment venues face attendance challenges",
    "Sports leagues negotiate reduced TV deals",
    "Advertising spending cuts hurt media companies",
    "Publishing industry faces headwinds",
    "Print media circulation continues decline",
    "Radio listenership drops among key demographics",
    "Cable TV subscriptions fall faster than expected",
    "Streaming competition intensifies, profits elusive",
    "Social media engagement plateaus",
    "Tech layoffs spread across major companies",
    "Startup funding dries up in tight market",
    "Venture capital valuations reset lower",
    "Private equity deals face financing challenges",
    "Hedge funds report significant outflows",
    "Pension funding gaps widen on market weakness",
    "Endowment returns disappoint",
    "Sovereign wealth funds face budget pressures",
    "Family offices reduce risk exposure",
    "Insurance companies raise premiums sharply",
    "Reinsurance capacity tightens after losses",
    "Bank lending standards tighten significantly",
    "Corporate bond issuance slows dramatically",
    "High-yield market shuts for most borrowers",
    "Leveraged loan demand evaporates",
    "Distressed debt opportunities multiply",
    "Bankruptcy filings increase across sectors",
    "Restructuring activity picks up pace",
    "Liquidation sales spread across retail",
    "Asset fire sales pressure valuations",
    "Margin calls force position unwinding",
    "Volatility spike triggers risk reduction",
    "Flight to quality benefits government bonds",
    "Safe-haven currencies strengthen sharply",
    "Defensive sectors outperform on market stress",
]


def generate_sample_data(
    num_samples: int = 10000,
    seed: int = 42,
) -> list[tuple[str, int]]:
    """
    Generate sample training data with balanced classes.

    Args:
        num_samples: Total number of samples to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of (headline, label) tuples.
            label: 0=negative, 1=neutral, 2=positive

    """
    random.seed(seed)

    # Calculate samples per class (roughly balanced)
    samples_per_class = num_samples // 3
    remainder = num_samples % 3

    data = []

    # Generate negative samples (label=0)
    negative_count = samples_per_class + (1 if remainder > 0 else 0)
    for _ in range(negative_count):
        headline = random.choice(NEGATIVE_HEADLINES)
        # Add slight variation
        variation = random.choice([
            "",
            " - Market Update",
            " - Breaking",
            " - Analysis",
            " Today",
            " This Week",
            f" (Stocks: {random.choice(['DOWN', 'FALLING', 'LOWER'])})",
        ])
        data.append((headline + variation, 0))

    # Generate neutral samples (label=1)
    neutral_count = samples_per_class + (1 if remainder > 1 else 0)
    for _ in range(neutral_count):
        headline = random.choice(NEUTRAL_HEADLINES)
        variation = random.choice([
            "",
            " - Market Update",
            " - Daily Brief",
            " - Roundup",
            " Today",
            " This Week",
            " (Stocks: MIXED)",
        ])
        data.append((headline + variation, 1))

    # Generate positive samples (label=2)
    positive_count = samples_per_class
    for _ in range(positive_count):
        headline = random.choice(POSITIVE_HEADLINES)
        variation = random.choice([
            "",
            " - Market Update",
            " - Breaking",
            " - Analysis",
            " Today",
            " This Week",
            f" (Stocks: {random.choice(['UP', 'RISING', 'HIGHER'])})",
        ])
        data.append((headline + variation, 2))

    # Shuffle the data
    random.shuffle(data)

    return data


def save_to_csv(data: list[tuple[str, int]], filepath: str) -> None:
    """
    Save data to CSV file.

    Args:
        data: List of (headline, label) tuples.
        filepath: Output file path.

    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("headline,label\n")
        for headline, label in data:
            # Escape quotes in headline
            escaped = headline.replace('"', '""')
            f.write(f'"{escaped}",{label}\n')

    print(f"Saved {len(data)} samples to: {filepath}")


def main():
    """Generate and save sample training and validation data."""
    print("=" * 60)
    print("AION Sentiment Analysis - Sample Data Generator")
    print("=" * 60)

    # Generate full dataset
    print("\nGenerating sample data...")
    all_data = generate_sample_data(num_samples=10000, seed=42)

    # Split into train/val (80/20)
    split_idx = int(len(all_data) * 0.8)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]

    print(f"Total samples: {len(all_data)}")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")

    # Save to CSV
    output_dir = Path(__file__).parent / "data" / "processed"
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"

    save_to_csv(train_data, str(train_path))
    save_to_csv(val_data, str(val_path))

    # Print label distribution
    print("\nLabel Distribution:")
    for name, data in [("Train", train_data), ("Val", val_data)]:
        labels = [label for _, label in data]
        print(f"  {name}: Negative={labels.count(0)}, Neutral={labels.count(1)}, Positive={labels.count(2)}")

    print("\n" + "=" * 60)
    print("Sample data generation complete!")
    print(f"Train: {train_path}")
    print(f"Val: {val_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
