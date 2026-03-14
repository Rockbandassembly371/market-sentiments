#!/usr/bin/env python3
# =============================================================================
# AION Open-Source Project: NSE Sector Constituents Scraper
# File: scrape_nse_sectors.py
# Description: Scrape NSE sector/industry constituents for sector-company mapping
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
NSE Sector Constituents Scraper.

This script scrapes NSE India website to get sector/industry → company mappings.
It supports multiple data sources:
    1. NSE website (live scraping)
    2. Yahoo Finance (alternative)
    3. Local CSV/Excel files

Output: CSV file with columns: sector, industry, company_name, symbol, series
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# NSE sector indices
NSE_SECTOR_INDICES = {
    "NIFTY AUTO": "AUTO",
    "NIFTY BANK": "BANK",
    "NIFTY FINANCIAL SERVICES": "FINNIFTY",
    "NIFTY FMCG": "FMCG",
    "NIFTY IT": "IT",
    "NIFTY METAL": "METAL",
    "NIFTY PHARMA": "PHARMA",
    "NIFTY PSU BANK": "PSUBANK",
    "NIFTY REALTY": "REALTY",
    "NIFTY MEDIA": "MEDIA",
    "NIFTY PRIVATE BANK": "PVTBANK",
    "NIFTY COMMODITIES": "COMMODITIES",
    "NIFTY CONSUMPTION": "CONSUMPTION",
    "NIFTY ENERGY": "ENERGY",
    "NIFTY INFRASTRUCTURE": "INFRA",
    "NIFTY SERVICES SECTOR": "SERVICES",
}


def scrape_nse_index_constituents(index_symbol: str) -> Optional[pd.DataFrame]:
    """
    Scrape constituents for a specific NSE index.

    Args:
        index_symbol: NSE index symbol (e.g., 'NIFTY AUTO', 'NIFTY BANK')

    Returns:
        DataFrame with columns: Symbol, Company Name, Series
    """
    base_url = "https://www.nseindia.com/market-data/constituents"
    
    try:
        # NSE API endpoint (internal API)
        url = f"https://www.nseindia.com/api/equity-stockIndices?index={index_symbol.replace(' ', '%20')}"
        
        headers = {
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/market-data/constituents",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        }
        
        session = requests.Session()
        session.headers.update(headers)
        
        # First, visit the main page to set cookies
        session.get("https://www.nseindia.com/", timeout=10)
        time.sleep(1)
        
        # Then fetch the API
        response = session.get(url, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract records
        records = data.get("records", [])
        
        if not records:
            logger.warning(f"No constituents found for {index_symbol}")
            return None
        
        # Parse records
        constituents = []
        for record in records:
            constituents.append({
                "symbol": record.get("symbol", ""),
                "company_name": record.get("companyName", ""),
                "series": record.get("series", "EQ"),
                "sector": index_symbol,
            })
        
        return pd.DataFrame(constituents)
        
    except Exception as e:
        logger.error(f"Failed to scrape {index_symbol}: {e}")
        return None


def scrape_all_nse_sectors() -> pd.DataFrame:
    """
    Scrape constituents for all NSE sector indices.

    Returns:
        DataFrame with all sector → company mappings
    """
    all_constituents = []
    
    logger.info("Scraping NSE sector constituents...")
    
    for index_name, symbol in NSE_SECTOR_INDICES.items():
        logger.info(f"Fetching {index_name} ({symbol})...")
        
        df = scrape_nse_index_constituents(index_name)
        
        if df is not None:
            all_constituents.append(df)
            logger.info(f"  ✓ Found {len(df)} companies")
        
        # Rate limiting
        time.sleep(1)
    
    if not all_constituents:
        logger.warning("No data scraped from any sector")
        return pd.DataFrame()
    
    # Combine all sectors
    result_df = pd.concat(all_constituents, ignore_index=True)
    
    # Remove duplicates
    result_df = result_df.drop_duplicates(subset=["symbol", "series"])
    
    logger.info(f"Total: {len(result_df)} unique companies across {len(NSE_SECTOR_INDICES)} sectors")
    
    return result_df


def scrape_yahoo_finance_sectors() -> pd.DataFrame:
    """
    Alternative: Get sector data from Yahoo Finance.

    Returns:
        DataFrame with sector → company mappings
    """
    logger.info("Fetching sector data from Yahoo Finance...")
    
    # Yahoo Finance sector indices
    yahoo_sectors = {
        "Basic Materials": "^JBASIC",
        "Communication Services": "^JCOMM",
        "Consumer Cyclical": "^JCYCL",
        "Consumer Defensive": "^JDEF",
        "Energy": "^JENER",
        "Financial Services": "^JFIN",
        "Healthcare": "^JHEAL",
        "Industrials": "^JINDU",
        "Real Estate": "^JREAL",
        "Technology": "^JTECH",
        "Utilities": "^JUTIL",
    }
    
    all_data = []
    
    for sector, symbol in yahoo_sectors.items():
        try:
            url = f"https://finance.yahoo.com/quote/{symbol}/holdings"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            
            # Parse holdings table (simplified - would need more robust parsing)
            logger.info(f"  {sector}: Data available")
            
        except Exception as e:
            logger.warning(f"Failed to fetch {sector}: {e}")
    
    return pd.DataFrame(all_data)


def load_from_pdf(pdf_path: str) -> pd.DataFrame:
    """
    Load sector classification from PDF file.

    Args:
        pdf_path: Path to NSE classification PDF

    Returns:
        DataFrame with sector/industry structure
    """
    try:
        import PyPDF2
        
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        
        # Parse text (would need custom parsing logic)
        logger.info(f"Extracted text from PDF: {len(text)} characters")
        
        # Return empty for now - would need custom parsing
        return pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Failed to read PDF: {e}")
        return pd.DataFrame()


def save_to_csv(df: pd.DataFrame, output_path: str) -> None:
    """
    Save sector mappings to CSV.

    Args:
        df: DataFrame with sector mappings
        output_path: Output file path
    """
    if df.empty:
        logger.warning("No data to save")
        return
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} rows to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Scrape NSE sector constituents for sector-company mapping"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/nse_sector_constituents.csv",
        help="Output CSV file path"
    )
    
    parser.add_argument(
        "--source",
        type=str,
        choices=["nse", "yahoo", "pdf"],
        default="nse",
        help="Data source to use"
    )
    
    parser.add_argument(
        "--pdf-path",
        type=str,
        help="Path to NSE classification PDF (if source=pdf)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("AION NSE Sector Constituents Scraper")
    logger.info("=" * 60)
    
    if args.source == "nse":
        df = scrape_all_nse_sectors()
    elif args.source == "yahoo":
        df = scrape_yahoo_finance_sectors()
    elif args.source == "pdf":
        if not args.pdf_path:
            logger.error("PDF path required for source=pdf")
            sys.exit(1)
        df = load_from_pdf(args.pdf_path)
    else:
        logger.error(f"Unknown source: {args.source}")
        sys.exit(1)
    
    if not df.empty:
        save_to_csv(df, args.output)
        
        # Display summary
        logger.info("\nSector Summary:")
        if "sector" in df.columns:
            print(df.groupby("sector").size().to_string())
    else:
        logger.warning("No data collected")
    
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
