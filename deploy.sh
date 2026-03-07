#!/bin/bash
# Quick deployment script

echo "🌾 Smart Harvest AI - Deployment Setup"
echo "========================================"

# Check if git is initialized
if [ ! -d .git ]; then
    echo "📦 Initializing git repository..."
    git init
    git add .
    git commit -m "Initial commit - Smart Harvest AI"
fi

echo ""
echo "✅ Ready to deploy!"
echo ""
echo "Choose your deployment platform:"
echo ""
echo "1. Render (Recommended)"
echo "   → Visit: https://render.com"
echo "   → New Web Service → Connect this repo"
echo "   → Build: pip install -r requirements.txt"
echo "   → Start: gunicorn --chdir backend app_v2:app --bind 0.0.0.0:\$PORT"
echo ""
echo "2. Railway"
echo "   → Visit: https://railway.app"
echo "   → Deploy from GitHub → Select this repo"
echo ""
echo "3. Heroku"
echo "   → Run: heroku create smart-harvest-ai"
echo "   → Run: git push heroku main"
echo ""
echo "📖 Full guide: See DEPLOYMENT.md"
