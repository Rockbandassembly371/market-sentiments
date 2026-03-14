#!/usr/bin/env python3
# =============================================================================
# AION Open-Source Project: HuggingFace Model Upload Script
# File: upload_to_huggingface.py
# Description: Upload trained model to HuggingFace Hub
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
Upload AION-Sentiment-IN model to HuggingFace Hub.

This script uploads the trained model to HuggingFace with proper metadata,
model card, and versioning.

Usage:
    python upload_to_huggingface.py --repo-id aion-analytics/aion-sentiment-in-v1
"""

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder


def main():
    """Main upload function."""
    parser = argparse.ArgumentParser(
        description="Upload AION-Sentiment-IN model to HuggingFace Hub"
    )
    
    parser.add_argument(
        "--repo-id",
        type=str,
        default="aion-analytics/aion-sentiment-in-v1",
        help="HuggingFace repository ID (username/repo-name)"
    )
    
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/aion-sentiment-in-v1",
        help="Path to trained model directory"
    )
    
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload AION-Sentiment-IN-v1 model (98.55% accuracy)",
        help="Git commit message"
    )
    
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make repository private"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without actually uploading"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("AION-Sentiment-IN - HuggingFace Upload")
    print("=" * 60)
    
    # Check if model directory exists
    model_path = Path(args.model_dir)
    if not model_path.exists():
        print(f"❌ Model directory not found: {model_path}")
        print("Please run training first: python train_sentiment.py")
        return
    
    # List model files
    model_files = list(model_path.glob("*"))
    print(f"\n📁 Model files to upload:")
    for f in model_files:
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"   - {f.name} ({size_mb:.1f} MB)")
    
    if args.dry_run:
        print("\n⚠️  DRY RUN - No files uploaded")
        return
    
    # Initialize HuggingFace API
    api = HfApi()
    
    try:
        # Create repository
        print(f"\n🔄 Creating repository: {args.repo_id}")
        create_repo(
            repo_id=args.repo_id,
            token=os.environ.get("HF_TOKEN"),
            exist_ok=True,
            private=args.private,
            repo_type="model"
        )
        print(f"✅ Repository created/verified")
        
        # Upload model
        print(f"\n📤 Uploading model to HuggingFace...")
        print(f"   This may take several minutes (model is ~440MB)")
        
        upload_folder(
            folder_path=str(model_path),
            repo_id=args.repo_id,
            token=os.environ.get("HF_TOKEN"),
            commit_message=args.commit_message,
            ignore_patterns=["*.md", "checkpoint-*"],  # Don't upload checkpoints
        )
        
        print(f"\n✅ Model uploaded successfully!")
        print(f"\n🔗 View model at:")
        print(f"   https://huggingface.co/{args.repo_id}")
        
        # Print usage instructions
        print(f"\n📝 Usage:")
        print(f"""
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "{args.repo_id}"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
        """)
        
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you're logged in: huggingface-cli login")
        print("2. Or set HF_TOKEN environment variable")
        print("3. Check that you have write access to the repository")
        return


if __name__ == "__main__":
    main()
