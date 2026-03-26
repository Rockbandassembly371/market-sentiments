#!/bin/bash
# =============================================================================
# AION Market Sentiment - Release Script
# =============================================================================
# This script handles:
# 1. Building and uploading Python packages to PyPI
# 2. Uploading model to HuggingFace
# 3. Creating GitHub release
#
# Usage: ./release.sh --version 2.0.0 --pypi-upload --hf-upload --gh-release
# =============================================================================

set -e

# Configuration
VERSION=""
PYPI_UPLOAD=false
HF_UPLOAD=false
GH_RELEASE=false
DRY_RUN=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --version)
            VERSION="$2"
            shift 2
            ;;
        --pypi-upload)
            PYPI_UPLOAD=true
            shift
            ;;
        --hf-upload)
            HF_UPLOAD=true
            shift
            ;;
        --gh-release)
            GH_RELEASE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./release.sh --version 2.0.0 --pypi-upload --hf-upload --gh-release"
            exit 1
            ;;
    esac
done

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not installed."
        exit 1
    fi
    
    # Check twine for PyPI upload
    if $PYPI_UPLOAD && ! command -v twine &> /dev/null; then
        log_error "twine is required for PyPI upload. Install with: pip install twine"
        exit 1
    fi
    
    # Check huggingface-cli for HF upload
    if $HF_UPLOAD && ! command -v huggingface-cli &> /dev/null; then
        log_error "huggingface-cli is required. Install with: pip install huggingface_hub"
        exit 1
    fi
    
    # Check git
    if ! command -v git &> /dev/null; then
        log_error "git is required but not installed."
        exit 1
    fi
    
    log_info "All prerequisites met."
}

build_packages() {
    log_info "Building Python packages..."
    
    # Build aion-sentiment
    if [ -d "aion-sentiment" ]; then
        log_info "Building aion-sentiment..."
        cd aion-sentiment
        python3 setup.py sdist bdist_wheel
        cd ..
    fi
    
    # Build aion-sectormap
    if [ -d "aion-sectormap" ]; then
        log_info "Building aion-sectormap..."
        cd aion-sectormap
        python3 setup.py sdist bdist_wheel
        cd ..
    fi
    
    # Build aion-volweight
    if [ -d "aion-volweight" ]; then
        log_info "Building aion-volweight..."
        cd aion-volweight
        python3 setup.py sdist bdist_wheel
        cd ..
    fi
    
    # Build aion-taxonomy
    if [ -d "aion_taxonomy" ]; then
        log_info "Building aion-taxonomy..."
        cd aion_taxonomy
        python3 setup.py sdist bdist_wheel
        cd ..
    fi
    
    log_info "All packages built successfully."
}

upload_pypi() {
    if ! $PYPI_UPLOAD; then
        return
    fi
    
    log_info "Uploading packages to PyPI..."
    
    if $DRY_RUN; then
        log_warn "Dry run - skipping PyPI upload"
        return
    fi
    
    # Upload aion-sentiment
    if [ -d "aion-sentiment/dist" ]; then
        log_info "Uploading aion-sentiment to PyPI..."
        twine upload aion-sentiment/dist/*
    fi
    
    # Upload aion-sectormap
    if [ -d "aion-sectormap/dist" ]; then
        log_info "Uploading aion-sectormap to PyPI..."
        twine upload aion-sectormap/dist/*
    fi
    
    # Upload aion-volweight
    if [ -d "aion-volweight/dist" ]; then
        log_info "Uploading aion-volweight to PyPI..."
        twine upload aion-volweight/dist/*
    fi
    
    # Upload aion-taxonomy
    if [ -d "aion_taxonomy/dist" ]; then
        log_info "Uploading aion-taxonomy to PyPI..."
        twine upload aion_taxonomy/dist/*
    fi
    
    log_info "All packages uploaded to PyPI."
}

upload_huggingface() {
    if ! $HF_UPLOAD; then
        return
    fi
    
    log_info "Uploading model to HuggingFace..."
    
    if $DRY_RUN; then
        log_warn "Dry run - skipping HuggingFace upload"
        return
    fi
    
    # Check if model directory exists
    if [ ! -d "aion-sentiment-in/models/aion-sentiment-in-v2" ]; then
        log_error "Model directory not found. Train model first."
        exit 1
    fi
    
    # Upload using huggingface-cli
    cd aion-sentiment-in/models
    huggingface-cli upload aion-analytics/aion-sentiment-in-v2 aion-sentiment-in-v2 .
    cd ../..
    
    log_info "Model uploaded to HuggingFace: aion-analytics/aion-sentiment-in-v2"
}

create_github_release() {
    if ! $GH_RELEASE; then
        return
    fi
    
    log_info "Creating GitHub release..."
    
    if $DRY_RUN; then
        log_warn "Dry run - skipping GitHub release"
        return
    fi
    
    # Check if CHANGELOG.md exists
    if [ ! -f "CHANGELOG.md" ]; then
        log_error "CHANGELOG.md not found. Create it before release."
        exit 1
    fi
    
    # Extract release notes from CHANGELOG.md
    RELEASE_NOTES=$(sed -n "/## \[${VERSION}\]/,/^## \[/p" CHANGELOG.md | head -n -1)
    
    # Create GitHub release using gh CLI (if available)
    if command -v gh &> /dev/null; then
        gh release create "v${VERSION}" \
            --title "Release v${VERSION}" \
            --notes "${RELEASE_NOTES}"
        log_info "GitHub release created: v${VERSION}"
    else
        log_warn "gh CLI not found. Create release manually at:"
        log_warn "https://github.com/AION-Analytics/market-sentiments/releases/new"
        log_warn "Tag: v${VERSION}"
    fi
}

print_summary() {
    echo ""
    echo "=============================================="
    echo "           RELEASE SUMMARY"
    echo "=============================================="
    echo ""
    echo "Version: ${VERSION}"
    echo ""
    
    if $PYPI_UPLOAD; then
        echo "✓ PyPI Upload: Completed"
        echo "  - aion-sentiment"
        echo "  - aion-sectormap"
        echo "  - aion-volweight"
        echo "  - aion-taxonomy"
    else
        echo "✗ PyPI Upload: Skipped"
    fi
    
    if $HF_UPLOAD; then
        echo "✓ HuggingFace Upload: Completed"
        echo "  - aion-analytics/aion-sentiment-in-v2"
    else
        echo "✗ HuggingFace Upload: Skipped"
    fi
    
    if $GH_RELEASE; then
        echo "✓ GitHub Release: Created"
    else
        echo "✗ GitHub Release: Skipped"
    fi
    
    echo ""
    echo "Next Steps:"
    echo "1. Verify packages on PyPI: https://pypi.org/user/aion-analytics/"
    echo "2. Verify model on HuggingFace: https://huggingface.co/aion-analytics"
    echo "3. Update GitHub releases: https://github.com/AION-Analytics/market-sentiments/releases"
    echo ""
    echo "=============================================="
}

# Main execution
main() {
    echo "=============================================="
    echo "   AION Market Sentiment - Release Script"
    echo "=============================================="
    echo ""
    
    if [ -z "$VERSION" ]; then
        log_error "Version is required. Use --version 2.0.0"
        exit 1
    fi
    
    if ! $PYPI_UPLOAD && ! $HF_UPLOAD && ! $GH_RELEASE; then
        log_error "At least one action required: --pypi-upload, --hf-upload, or --gh-release"
        exit 1
    fi
    
    check_prerequisites
    
    if $PYPI_UPLOAD; then
        build_packages
        upload_pypi
    fi
    
    if $HF_UPLOAD; then
        upload_huggingface
    fi
    
    if $GH_RELEASE; then
        create_github_release
    fi
    
    print_summary
}

main
