#!/bin/bash
# Render deployment reference for Smart Harvest AI.
# Render uses Procfile automatically; this script only prints the same commands.

 echo "🌾 Smart Harvest AI - Deployment Setup"
echo "========================================"

 echo ""
echo "✅ Ready to deploy!"
echo ""
echo "Choose your deployment platform:"
echo ""
echo "1. Render"
echo "   → Build: pip install -r requirements.txt"
echo "   → Start: Procfile (one memory-safe gthread worker)"
echo "   → Health check: /api/health"
echo ""
echo "2. Required Render environment variables"
echo "   → GROQ_API_KEY"
echo "   → PINECONE_API_KEY"
echo "   → PINECONE_INDEX_NAME"
echo "   → SECRET_KEY (recommended)"
echo ""
echo "See README.md for deployment and model details."
