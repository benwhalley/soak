#!/bin/bash
set -e

# Clean and build
rm -rf dist build *.egg-info
python -m build
twine check dist/*

# Upload
echo "Ready to upload to PyPI"
read -p "Continue? (y/N) " -r
if [[ $REPLY =~ ^[Yy]$ ]]; then
    twine upload dist/*
    echo "✓ Published: pip install soaking"
fi
