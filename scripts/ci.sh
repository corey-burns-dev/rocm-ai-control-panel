#!/bin/bash
# CI script to lint and format the project

set -e

echo "Running ESLint..."
npm run lint

echo "Running Prettier..."
npm run format

echo "✓ Linting and formatting complete!"
