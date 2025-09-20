#!/bin/bash

# Navigate to the project directory
cd /Users/ryanhofmann/Desktop/ffanalyticspro

# Initialize git repository
git init

# Add the GitHub remote (replace with your actual GitHub repository URL)
git remote add origin https://github.com/ryanehofmann-11/ffanalyticspro.git

# Add all files to staging
git add .

# Commit the changes
git commit -m "Add complete Fantasy Football Analytics Pro project

- Machine learning models for all positions (QB, RB, WR, TE, K, DST)
- Automated roster updater with Sleeper API integration
- Comprehensive test suite
- Data processing and model training notebooks
- FantasyPros scraper for additional data
- Robust ML system with feature engineering
- Complete documentation and requirements"

# Push to GitHub (you may need to authenticate)
git branch -M main
git push -u origin main

echo "Project uploaded to GitHub successfully!"
