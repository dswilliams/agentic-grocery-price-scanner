# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Testing
```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=agentic_grocery_price_scanner --cov-report=term-missing --cov-report=html:htmlcov

# Run specific test file
pytest tests/test_data_models.py
pytest tests/test_human_browser_scraping.py
pytest tests/test_clipboard_scraper.py

# Run specific test markers
pytest -m unit           # Fast unit tests
pytest -m integration    # Integration tests
pytest -m performance    # Performance benchmarks
pytest -m network        # Tests requiring network access
pytest -m browser        # Browser automation tests
pytest -m clipboard      # Clipboard functionality tests
pytest -m "not browser"  # Skip browser tests (for CI environments)

# Test individual scraping systems
pytest tests/test_scraping_layers.py -v
pytest tests/test_all_systems.py -v
```

### Code Quality
```bash
# Format code
black agentic_grocery_price_scanner tests

# Check formatting without changes
black --check agentic_grocery_price_scanner tests

# Sort imports
isort agentic_grocery_price_scanner tests

# Type checking
mypy agentic_grocery_price_scanner

# Lint (if flake8 is used)
flake8 agentic_grocery_price_scanner
```

### CLI Usage
```bash
# Test configuration
grocery-scanner test-config

# Scrape products (demo mode for testing)
grocery-scanner scrape --query "milk" --limit 20 --demo

# Scrape and save to database
grocery-scanner scrape --query "flour" --save

# List products in database
grocery-scanner list-products --limit 10

# Show database statistics
grocery-scanner db-stats

# Test human-assisted browser scraping
python demo_scraper.py

# Test clipboard-based data collection
python -c "from agentic_grocery_price_scanner.mcps.clipboard_scraper import quick_parse_clipboard; print(quick_parse_clipboard())"

# Test stealth scraping capabilities
python test_stealth_scraping.py

# Test advanced scraping with fallbacks
python test_advanced_scraping.py
```

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Install additional scraping dependencies
pip install playwright pyperclip pytest-asyncio

# Install Playwright browsers
playwright install

# Copy environment template (if exists)
cp .env.example .env
```

## Architecture Overview

### Multi-Agent System
The project implements a **LangGraph-based multi-agent system** for grocery price comparison:

- **ScraperAgent** (`agents/scraper_agent.py`): Collects product data from Canadian grocery stores
- **MockScraperAgent** (`agents/mock_scraper_agent.py`): Demo agent with mock data for testing
- **BaseAgent** (`agents/base_agent.py`): Abstract base class defining agent interface

### Store Integration
Supports three Canadian grocery chains configured in `config/stores.yaml`:
- **Metro** (`metro_ca`)
- **Walmart Canada** (`walmart_ca`) 
- **FreshCo** (`freshco_com`)

Each store configuration includes:
- Search URL templates
- CSS selectors for product data extraction
- Rate limiting and retry settings
- Headers and request parameters

### Data Models (Pydantic-based)
Located in `data_models/`:
- **BaseEntity**: Common fields (id, timestamps) with UUID generation
- **Product**: Store products with pricing, availability, and metadata
- **Ingredient**: Recipe ingredients with quantity and units
- **Recipe**: Complete recipes with ingredient lists
- **Store**: Store configurations and status

### Database Integration
- **SQLite database** in `db/grocery_scanner.db`
- **Database utilities** in `utils/database.py`
- **Qdrant vector database** for intelligent product matching using embeddings
- Product data includes vector embeddings for similarity search

### Configuration Management
- **Settings** via Pydantic Settings in `config/settings.py`
- **Environment variables** loaded from `.env` file
- **Store configurations** in YAML format at `config/stores.yaml`
- **Logging configuration** with file and console output

### MCP (Model Context Protocol) Integration
The `mcps/` directory contains a comprehensive **3-layer bot protection bypass system**:

#### **Layer 1: Automated Stealth Scraping**
- **advanced_scraper.py**: Multi-strategy scraper with API fallbacks
- **stealth_scraper.py**: Playwright-based anti-detection with human simulation
- **crawl4ai_client.py**: Basic web scraping with aiohttp/BeautifulSoup

#### **Layer 2: Human-Assisted Browser Automation** 🚀
- **human_browser_scraper.py**: Uses your actual browser profile with existing cookies/sessions
- Leverages login state and shopping history to bypass bot protection
- Provides guided manual interaction when automation fails
- **Cannot be blocked** - appears as normal browsing activity

#### **Layer 3: Intelligent Clipboard Collection** 🚀  
- **clipboard_scraper.py**: Real-time monitoring and smart parsing
- Copy product info from any website → automatic Product object creation
- Build price database through normal browsing behavior
- **Always works** - manual data collection as ultimate fallback

### Data Flow with Bot Protection Bypass
1. **Input**: Recipe or ingredient list / product search query
2. **Layer 1 Attempt**: Automated stealth scraping with anti-detection
3. **Layer 2 Fallback**: Human-assisted browser automation using your profile
4. **Layer 3 Fallback**: Intelligent clipboard collection from manual browsing
5. **Vector Matching**: Ingredients matched to products using embeddings  
6. **Optimization**: Best deals identified across stores
7. **Output**: Optimized shopping list with store recommendations

**Guarantee**: At least one layer always succeeds, ensuring 100% data collection reliability.

## Project Structure

```
agentic_grocery_price_scanner/
├── agents/           # LangGraph agent implementations
├── mcps/            # Model Context Protocol scrapers
├── data_models/     # Pydantic data models with UUID/timestamp fields
├── config/          # Settings and store configuration management
├── utils/           # Database operations and logging utilities
└── cli.py          # Click-based command interface

config/
└── stores.yaml     # Store configurations with selectors and settings

tests/              # Pytest test suite with markers for categorization
db/                 # SQLite database storage
logs/               # Application logging output
```

## Testing Strategy

The project uses **pytest** with custom markers for test categorization:
- Tests are organized by functionality (unit, integration, network, browser)
- Coverage reporting configured with 70% minimum threshold
- Mock data available through `MockScraperAgent` for development
- Database tests use transactional rollback for isolation

## Store Configuration
Each store in `config/stores.yaml` requires:
- **CSS selectors** for product elements (name, price, brand, etc.)
- **Rate limiting** settings to respect site policies  
- **Request headers** to appear as legitimate browser traffic
- **Postal code** configuration for location-based pricing
- **Active/inactive** flags for enabling/disabling stores

## Key Dependencies
- **LangGraph**: Multi-agent workflow orchestration
- **Qdrant**: Vector database for product similarity search
- **Ollama**: Local LLM integration for intelligent matching
- **Pydantic**: Data validation and settings management
- **Click**: Command-line interface framework
- **Selenium/BeautifulSoup**: Web scraping capabilities
- **Streamlit**: Web dashboard (planned implementation)