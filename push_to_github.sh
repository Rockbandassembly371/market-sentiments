#!/bin/bash
# =============================================================================
# AION Open-Source - GitHub Push Script
# =============================================================================

echo "=============================================="
echo "AION Open-Source - Push to GitHub"
echo "=============================================="
echo ""

cd /Users/lokeshgupta/aion_open_source

# Check if repository exists
if [ ! -d ".git" ]; then
    echo "❌ Error: Not a git repository"
    exit 1
fi

echo "📦 Files to push:"
git status --short | head -10
echo ""

echo "🔗 Remote URL:"
git remote -v
echo ""

echo "🚀 Pushing to GitHub..."
echo "   You will be prompted for GitHub credentials."
echo "   Use your Personal Access Token (PAT) as password."
echo ""

# Attempt push
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ SUCCESS! Code pushed to GitHub"
    echo ""
    echo "📬 Next steps:"
    echo "   1. Go to https://github.com/AION-Analytics/market-sentiments"
    echo "   2. Verify all files are uploaded"
    echo "   3. Upload model to HuggingFace:"
    echo "      cd aion-sentiment-in"
    echo "      python upload_to_huggingface.py"
    echo ""
else
    echo ""
    echo "❌ Push failed. Troubleshooting:"
    echo ""
    echo "Option 1: Use Personal Access Token"
    echo "   1. Go to https://github.com/settings/tokens"
    echo "   2. Generate new token with 'repo' scope"
    echo "   3. Run: git push -u origin main"
    echo "   4. Username: aionlabs@tutamail.com"
    echo "   5. Password: <paste your PAT>"
    echo ""
    echo "Option 2: Use SSH key"
    echo "   1. Copy your SSH key:"
    echo "      cat ~/.ssh/id_ed25519.pub"
    echo "   2. Add to GitHub: https://github.com/settings/keys"
    echo "   3. Add key to agent: ssh-add ~/.ssh/id_ed25519"
    echo "   4. Run: git push -u origin main"
    echo ""
fi
