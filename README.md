# Agentic Grocery Price Scanner

A multi-agent system for scraping grocery prices and optimizing shopping lists across Canadian grocery stores.

## Features

- 🤖 **Multi-Agent System**: Uses LangGraph for coordinating specialized agents
- 🛒 **Store Integration**: Supports Metro, Walmart Canada, and FreshCo
- 📊 **Vector Search**: Intelligent ingredient-to-product matching using Qdrant
- 🎯 **Recipe Parsing**: Extract ingredients from recipes automatically
- 💰 **Price Optimization**: Find the best deals across multiple stores
- 🌐 **Web Interface**: Streamlit-based dashboard for easy interaction
- 📱 **CLI Tools**: Command-line interface for automation

## Installation

### Prerequisites

- Python 3.9+
- Ollama (for local LLM)
- Qdrant (for vector database)

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd agentic-grocery-price-scanner
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   # For development:
   pip install -r requirements-dev.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

5. **Initialize the database:**
   ```bash
   grocery-scanner test-config
   ```

## Quick Start

### CLI Usage

1. **Test configuration:**
   ```bash
   grocery-scanner test-config
   ```

2. **Scrape products:**
   ```bash
   grocery-scanner scrape --query "milk" --limit 20
   ```

3. **Match recipe ingredients:**
   ```bash
   grocery-scanner match --ingredients "flour" "sugar" "eggs"
   ```

4. **Optimize shopping list:**
   ```bash
   grocery-scanner optimize --budget 50.00 --strategy cheapest
   ```

### Web Interface

Launch the Streamlit dashboard:
```bash
grocery-scanner web --port 8501
```

## Project Structure

```
agentic_grocery_price_scanner/
├── agents/                 # LangGraph agents
├── mcps/                  # Model Context Protocols
├── data_models/           # Pydantic data models
├── config/               # Configuration management
├── utils/                # Database and logging utilities
├── cli.py               # Command-line interface
└── __init__.py          # Package initialization

config/
└── stores.yaml          # Store configurations

tests/                   # Test suite
db/                     # SQLite database
logs/                   # Application logs
```

## Configuration

### Store Configuration

Edit `config/stores.yaml` to add or modify store configurations:

```yaml
stores:
  metro_ca:
    name: "Metro"
    base_url: "https://www.metro.ca"
    search_url_template: "https://www.metro.ca/en/online-grocery/search?filter={query}"
    selectors:
      product_name: ".product-name"
      price: ".price-update"
    # ... more configuration
```

### Environment Variables

Key environment variables in `.env`:

- `DEBUG=false` - Enable debug mode
- `LOG_LEVEL=INFO` - Set logging level
- `DB_DB_PATH=db/grocery_scanner.db` - Database path
- `QDRANT_HOST=localhost` - Qdrant host
- `LLM_DEFAULT_MODEL=llama2` - Default Ollama model

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=agentic_grocery_price_scanner

# Run specific test file
pytest tests/test_data_models.py
```

### Code Formatting

```bash
# Format code
black agentic_grocery_price_scanner tests

# Check formatting
black --check agentic_grocery_price_scanner tests

# Sort imports
isort agentic_grocery_price_scanner tests
```

### Type Checking

```bash
mypy agentic_grocery_price_scanner
```

## Architecture

### Multi-Agent System

The system uses specialized agents coordinated by LangGraph:

1. **Scraper Agent**: Collects product data from stores
2. **Matcher Agent**: Matches ingredients to products using vector similarity
3. **Optimizer Agent**: Finds optimal shopping strategies
4. **Recipe Agent**: Parses and processes recipe data

### Data Flow

1. Recipe input → Extract ingredients
2. Ingredients → Vector embedding → Product matching
3. Matched products → Price comparison → Optimization
4. Output → Shopping list with store recommendations

## API Reference

### Data Models

- `Recipe`: Recipe with ingredients and metadata
- `Ingredient`: Individual recipe ingredient
- `Product`: Store product with pricing info
- `Store`: Store configuration and status

### CLI Commands

- `grocery-scanner scrape`: Scrape store data
- `grocery-scanner match`: Match ingredients to products  
- `grocery-scanner optimize`: Optimize shopping lists
- `grocery-scanner test-config`: Test configuration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Roadmap

- [ ] Implement LangGraph agent workflows
- [ ] Add more Canadian grocery stores
- [ ] Recipe import from popular websites
- [ ] Mobile app interface
- [ ] Price history tracking
- [ ] Deal alerts and notifications