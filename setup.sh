#!/bin/bash

# Agentic Grocery Price Scanner Setup Script

set -e

echo "🛒 Setting up Agentic Grocery Price Scanner"
echo "=========================================="

# Check Python version
echo "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.9+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
echo "✅ Found Python $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Installing development dependencies..."
pip install -r requirements-dev.txt

# Install package in development mode
echo "Installing package in development mode..."
pip install -e .

# Create necessary directories
echo "Creating directories..."
mkdir -p db logs

# Copy environment file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your specific settings"
else
    echo "✅ .env file already exists"
fi

# Test configuration
echo "Testing configuration..."
if grocery-scanner test-config; then
    echo "✅ Configuration test passed"
else
    echo "⚠️  Configuration test had issues - check the output above"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Edit .env file if needed"
echo ""
echo "3. Test the CLI:"
echo "   grocery-scanner test-config"
echo ""
echo "4. Start scraping:"
echo "   grocery-scanner scrape --query 'milk' --limit 10"
echo ""
echo "5. Launch web interface:"
echo "   grocery-scanner web"
echo ""
echo "For development:"
echo "- Run tests: pytest"
echo "- Format code: black ."
echo "- Check types: mypy agentic_grocery_price_scanner"