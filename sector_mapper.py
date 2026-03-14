#!/usr/bin/env python3
# =============================================================================
# AION Open-Source Project: Sector Mapper
# File: sector_mapper.py
# Description: Utility for sector → company mapping using NSE group companies data
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
Sector Mapper for AION.

This module provides utilities for mapping companies to sectors and groups
using the NSE Group Companies dataset (591 companies, 44 sectors, 340 groups).

Classes:
    SectorMapper: Main class for sector/company/group mappings.

Example:
    >>> from sector_mapper import SectorMapper
    >>> mapper = SectorMapper()
    >>> mapper.get_sector('HDFC Bank Limited')
    'Financial Services'
    >>> mapper.get_companies_in_sector('Financial Services')
    ['HDFC Bank Limited', 'ICICI Bank Limited', ...]
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd


class SectorMapper:
    """
    Mapper for company → sector → group relationships.

    This class provides bidirectional mapping between companies, sectors,
    and business groups using the NSE Group Companies dataset.

    Attributes:
        data_path (str): Path to the CSV file with company data.
        df (pd.DataFrame): DataFrame with all company data.

    Example:
        >>> mapper = SectorMapper('data/nse_group_companies.csv')
        >>> sector = mapper.get_sector('HDFC Bank Limited')
        >>> companies = mapper.get_companies_in_sector('Financial Services')
        >>> groups = mapper.get_all_groups()
    """

    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the SectorMapper.

        Args:
            data_path: Path to CSV file with company data.
                If None, uses default location in data/ directory.

        Raises:
            FileNotFoundError: If data file not found.
        """
        if data_path is None:
            # Try multiple locations
            possible_paths = [
                'data/nse_group_companies.csv',
                os.path.join(os.path.dirname(__file__), '..', 'data', 'nse_group_companies.csv'),
                os.path.join(os.path.dirname(__file__), 'data', 'nse_group_companies.csv'),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    data_path = path
                    break
            else:
                data_path = 'data/nse_group_companies.csv'

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Company data not found at {data_path}")

        self.data_path = data_path
        self.df = pd.read_csv(data_path)

        # Create lookup tables
        self._company_to_sector: Dict[str, str] = {}
        self._company_to_group: Dict[str, str] = {}
        self._sector_to_companies: Dict[str, List[str]] = {}
        self._group_to_companies: Dict[str, List[str]] = {}

        self._build_lookups()

    def _build_lookups(self) -> None:
        """Build lookup dictionaries for fast access."""
        for _, row in self.df.iterrows():
            company = row['Name of the Company']
            sector = row['Sector']
            group = row['Group Name']

            # Company → Sector
            self._company_to_sector[company] = sector

            # Company → Group
            self._company_to_group[company] = group

            # Sector → Companies
            if sector not in self._sector_to_companies:
                self._sector_to_companies[sector] = []
            self._sector_to_companies[sector].append(company)

            # Group → Companies
            if group not in self._group_to_companies:
                self._group_to_companies[group] = []
            self._group_to_companies[group].append(company)

    def get_sector(self, company_name: str) -> Optional[str]:
        """
        Get sector for a company.

        Args:
            company_name: Full company name.

        Returns:
            Sector name or None if not found.

        Example:
            >>> mapper.get_sector('HDFC Bank Limited')
            'Financial Services'
        """
        return self._company_to_sector.get(company_name)

    def get_group(self, company_name: str) -> Optional[str]:
        """
        Get business group for a company.

        Args:
            company_name: Full company name.

        Returns:
            Group name or None if not found.

        Example:
            >>> mapper.get_group('HDFC Bank Limited')
            'HDFC Bank Limited'
        """
        return self._company_to_group.get(company_name)

    def get_companies_in_sector(self, sector: str) -> List[str]:
        """
        Get all companies in a sector.

        Args:
            sector: Sector name (case-insensitive).

        Returns:
            List of company names in the sector.

        Example:
            >>> mapper.get_companies_in_sector('Financial Services')
            ['HDFC Bank Limited', 'ICICI Bank Limited', ...]
        """
        # Case-insensitive lookup
        sector_lower = sector.lower()
        for s, companies in self._sector_to_companies.items():
            if s.lower() == sector_lower:
                return companies
        return []

    def get_companies_in_group(self, group: str) -> List[str]:
        """
        Get all companies in a business group.

        Args:
            group: Group name (case-insensitive).

        Returns:
            List of company names in the group.

        Example:
            >>> mapper.get_companies_in_group('Aditya Birla Group')
            ['Aditya Birla Capital Limited', 'Grasim Industries Limited', ...]
        """
        # Case-insensitive lookup
        group_lower = group.lower()
        for g, companies in self._group_to_companies.items():
            if g.lower() == group_lower:
                return companies
        return []

    def get_all_sectors(self) -> List[str]:
        """
        Get list of all sectors.

        Returns:
            List of sector names.
        """
        return list(self._sector_to_companies.keys())

    def get_all_groups(self) -> List[str]:
        """
        Get list of all business groups.

        Returns:
            List of group names.
        """
        return list(self._group_to_companies.keys())

    def get_all_companies(self) -> List[str]:
        """
        Get list of all companies.

        Returns:
            List of company names.
        """
        return list(self._company_to_sector.keys())

    def get_sector_summary(self) -> pd.DataFrame:
        """
        Get summary of companies per sector.

        Returns:
            DataFrame with sector names and company counts.
        """
        summary = self.df.groupby('Sector').size().reset_index(name='company_count')
        return summary.sort_values('company_count', ascending=False)

    def get_group_summary(self) -> pd.DataFrame:
        """
        Get summary of companies per group.

        Returns:
            DataFrame with group names and company counts.
        """
        summary = self.df.groupby('Group Name').size().reset_index(name='company_count')
        return summary.sort_values('company_count', ascending=False)

    def search_company(self, query: str) -> pd.DataFrame:
        """
        Search for companies by name.

        Args:
            query: Search query (case-insensitive substring match).

        Returns:
            DataFrame with matching companies.

        Example:
            >>> mapper.search_company('HDFC')
               Name of the Company                    Sector           Group Name
            0  HDFC Bank Limited         Financial Services  HDFC Bank Limited
            1  HDB Financial Services  Financial Services    HDFC Bank Limited
        """
        mask = self.df['Name of the Company'].str.contains(query, case=False, na=False)
        return self.df[mask]

    def to_dataframe(self) -> pd.DataFrame:
        """
        Get full company data as DataFrame.

        Returns:
            DataFrame with all company data.
        """
        return self.df.copy()


def main():
    """Demo the SectorMapper functionality."""
    print("=" * 60)
    print("AION Sector Mapper - Demo")
    print("=" * 60)

    try:
        mapper = SectorMapper()

        print(f"\nTotal Companies: {len(mapper.get_all_companies())}")
        print(f"Total Sectors: {len(mapper.get_all_sectors())}")
        print(f"Total Groups: {len(mapper.get_all_groups())}")

        print("\n" + "=" * 60)
        print("Sector Summary (Top 10)")
        print("=" * 60)
        print(mapper.get_sector_summary().head(10).to_string())

        print("\n" + "=" * 60)
        print("Group Summary (Top 10)")
        print("=" * 60)
        print(mapper.get_group_summary().head(10).to_string())

        # Demo lookups
        print("\n" + "=" * 60)
        print("Sample Lookups")
        print("=" * 60)

        test_companies = ['HDFC Bank Limited', 'Reliance Industries Limited', 'Tata Motors Ltd']
        for company in test_companies:
            sector = mapper.get_sector(company)
            group = mapper.get_group(company)
            print(f"\n{company}:")
            print(f"  Sector: {sector}")
            print(f"  Group: {group}")

        # Demo sector lookup
        print("\n" + "=" * 60)
        print("Companies in 'Financial Services' (first 10)")
        print("=" * 60)
        companies = mapper.get_companies_in_sector('Financial Services')[:10]
        for company in companies:
            print(f"  - {company}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure nse_group_companies.csv exists in data/ directory")


if __name__ == "__main__":
    main()
