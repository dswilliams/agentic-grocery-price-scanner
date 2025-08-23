"""
Command Line Interface for the Agentic Grocery Price Scanner.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

import click

from .config import get_settings, load_store_configs
from .utils.logging import setup_logging


@click.group()
@click.option(
    "--config", 
    type=click.Path(exists=True), 
    help="Path to configuration file"
)
@click.option(
    "--debug/--no-debug", 
    default=False, 
    help="Enable debug mode"
)
@click.option(
    "--log-level", 
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    default="INFO",
    help="Set logging level"
)
@click.pass_context
def cli(ctx: click.Context, config: Optional[str], debug: bool, log_level: str) -> None:
    """Agentic Grocery Price Scanner - Multi-agent grocery price comparison system."""
    ctx.ensure_object(dict)
    
    # Load settings
    settings = get_settings()
    if debug:
        settings.debug = True
        settings.log_level = "DEBUG"
    else:
        settings.log_level = log_level
    
    # Setup logging
    setup_logging(settings.log_level, settings.log_file)
    
    # Store settings in context
    ctx.obj["settings"] = settings
    ctx.obj["config_file"] = config or settings.config_file
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting {settings.app_name}")


@cli.command()
@click.option(
    "--stores", 
    multiple=True, 
    help="Specific stores to scrape (default: all active stores)"
)
@click.option(
    "--query", 
    required=True, 
    help="Search query for products"
)
@click.option(
    "--limit", 
    type=int, 
    default=50, 
    help="Maximum number of products to scrape per store"
)
@click.option(
    "--save", 
    is_flag=True,
    help="Save results to database"
)
@click.option(
    "--demo",
    is_flag=True,
    help="Use mock data instead of real web scraping"
)
@click.pass_context
def scrape(ctx: click.Context, stores: tuple, query: str, limit: int, save: bool, demo: bool) -> None:
    """Scrape product data from grocery stores."""
    settings = ctx.obj["settings"]
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting scrape for query: {query}")
    
    async def run_scraper():
        from .agents import ScraperAgent, MockScraperAgent
        from .utils import get_db_manager
        
        # Initialize scraper agent (real or mock)
        if demo:
            scraper = MockScraperAgent()
            click.echo("🎭 Using demo mode with mock data")
        else:
            scraper = ScraperAgent()
        
        # Prepare inputs
        inputs = {
            "query": query,
            "limit": limit
        }
        if stores:
            inputs["stores"] = list(stores)
        
        # Execute scraping
        click.echo(f"🔍 Scraping stores for '{query}'...")
        result = await scraper.execute(inputs)
        
        if result["success"]:
            click.echo(f"✅ Scraping completed!")
            click.echo(f"   Total products: {result['total_products']}")
            click.echo(f"   Stores scraped: {result['stores_scraped']}")
            
            if result['stores_failed'] > 0:
                click.echo(f"   Stores failed: {result['stores_failed']}")
                for store_id, error in result['errors'].items():
                    click.echo(f"     - {store_id}: {error}")
            
            # Display sample products
            products = result['products']
            if products:
                click.echo(f"\n📦 Sample products:")
                for product in products[:5]:
                    price_display = f"${product.price}"
                    if product.on_sale and product.sale_price:
                        price_display = f"${product.sale_price} (was ${product.price})"
                    click.echo(f"   - {product.name}")
                    click.echo(f"     Store: {product.store_id} | Price: {price_display}")
                    if product.brand:
                        click.echo(f"     Brand: {product.brand}")
                
                if len(products) > 5:
                    click.echo(f"   ... and {len(products) - 5} more")
            
            # Save to database if requested
            if save:
                try:
                    db_manager = get_db_manager()
                    saved_count = 0
                    
                    for product in products:
                        # Convert to dict for database storage
                        product_dict = product.model_dump()
                        product_dict['keywords'] = ','.join(product.keywords)
                        
                        # Insert product (simplified - would need proper upsert logic)
                        db_manager.execute_query("""
                            INSERT OR REPLACE INTO products 
                            (id, name, brand, price, currency, store_id, 
                             image_url, product_url, in_stock, on_sale, keywords,
                             created_at, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                                   datetime('now'), datetime('now'))
                        """, (
                            str(product.id), product.name, product.brand, 
                            float(product.price), product.currency.value, product.store_id,
                            str(product.image_url) if product.image_url else None,
                            str(product.product_url) if product.product_url else None,
                            product.in_stock, product.on_sale, 
                            ','.join(product.keywords)
                        ))
                        saved_count += 1
                    
                    click.echo(f"💾 Saved {saved_count} products to database")
                    
                except Exception as e:
                    click.echo(f"⚠️  Failed to save to database: {e}")
        else:
            click.echo(f"❌ Scraping failed: {result.get('error', 'Unknown error')}")
    
    # Run the async function
    try:
        asyncio.run(run_scraper())
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        click.echo(f"❌ Error: {e}")


@cli.command()
@click.option(
    "--recipe-file", 
    type=click.Path(exists=True),
    help="Path to recipe file (JSON/YAML)"
)
@click.option(
    "--ingredients", 
    multiple=True,
    help="Individual ingredients to match"
)
@click.pass_context
def match(ctx: click.Context, recipe_file: Optional[str], ingredients: tuple) -> None:
    """Match recipe ingredients to store products."""
    logger = logging.getLogger(__name__)
    
    if not recipe_file and not ingredients:
        click.echo("Error: Must provide either --recipe-file or --ingredients")
        return
    
    if recipe_file:
        click.echo(f"Matching ingredients from recipe file: {recipe_file}")
        # TODO: Load recipe and extract ingredients
    
    if ingredients:
        click.echo(f"Matching individual ingredients: {', '.join(ingredients)}")
    
    # TODO: Implement matching logic
    click.echo("Ingredient matching functionality will be implemented in agents module")


@cli.command()
@click.option(
    "--budget", 
    type=float,
    help="Maximum budget for optimization"
)
@click.option(
    "--stores", 
    multiple=True,
    help="Stores to include in optimization"
)
@click.option(
    "--strategy", 
    type=click.Choice(["cheapest", "balanced", "quality"]),
    default="cheapest",
    help="Optimization strategy"
)
@click.pass_context
def optimize(ctx: click.Context, budget: Optional[float], stores: tuple, strategy: str) -> None:
    """Optimize shopping list across multiple stores."""
    logger = logging.getLogger(__name__)
    
    click.echo(f"Optimizing shopping list with strategy: {strategy}")
    
    if budget:
        click.echo(f"Budget constraint: ${budget:.2f}")
    
    if stores:
        click.echo(f"Store constraints: {', '.join(stores)}")
    
    # TODO: Implement optimization logic
    click.echo("Shopping list optimization will be implemented in agents module")


@cli.command()
@click.pass_context
def test_config(ctx: click.Context) -> None:
    """Test configuration and store connectivity."""
    settings = ctx.obj["settings"]
    config_file = ctx.obj["config_file"]
    
    logger = logging.getLogger(__name__)
    click.echo("Testing configuration...")
    
    try:
        # Test settings
        click.echo(f"✓ Settings loaded successfully")
        click.echo(f"  App: {settings.app_name}")
        click.echo(f"  Debug: {settings.debug}")
        click.echo(f"  Log level: {settings.log_level}")
        
        # Test store configurations
        store_configs = load_store_configs(config_file)
        click.echo(f"✓ Loaded {len(store_configs)} store configurations")
        
        for store_id, store in store_configs.items():
            active_status = "active" if getattr(store, 'active', True) else "inactive"
            click.echo(f"  - {store.name} ({store_id}): {active_status}")
        
        # Test database path
        db_path = Path(settings.database.db_path)
        if db_path.parent.exists():
            click.echo(f"✓ Database directory exists: {db_path.parent}")
        else:
            click.echo(f"⚠ Database directory will be created: {db_path.parent}")
        
        # Test log directory
        log_path = Path(settings.log_file)
        if log_path.parent.exists():
            click.echo(f"✓ Log directory exists: {log_path.parent}")
        else:
            click.echo(f"⚠ Log directory will be created: {log_path.parent}")
        
        click.echo("\n✓ Configuration test completed successfully!")
        
    except Exception as e:
        logger.error(f"Configuration test failed: {e}")
        click.echo(f"✗ Configuration test failed: {e}")


@cli.command()
@click.option(
    "--store", 
    help="Filter by store ID"
)
@click.option(
    "--limit", 
    type=int, 
    default=20, 
    help="Maximum products to show"
)
@click.pass_context
def list_products(ctx: click.Context, store: Optional[str], limit: int) -> None:
    """List products from the database."""
    from .utils import get_db_manager
    
    db_manager = get_db_manager()
    
    # Build query
    query = "SELECT name, brand, price, currency, store_id, in_stock, on_sale FROM products"
    params = []
    
    if store:
        query += " WHERE store_id = ?"
        params.append(store)
    
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    
    try:
        products = db_manager.fetch_all(query, tuple(params))
        
        if not products:
            click.echo("No products found in database")
            return
        
        click.echo(f"📦 Found {len(products)} products:")
        click.echo("")
        
        for product in products:
            status_icons = []
            if not product['in_stock']:
                status_icons.append("❌ Out of Stock")
            if product['on_sale']:
                status_icons.append("🏷️  On Sale")
            
            status_text = " | ".join(status_icons) if status_icons else "✅ Available"
            
            click.echo(f"• {product['name']}")
            click.echo(f"  Brand: {product['brand'] or 'N/A'}")
            click.echo(f"  Price: ${product['price']} {product['currency']}")
            click.echo(f"  Store: {product['store_id']}")
            click.echo(f"  Status: {status_text}")
            click.echo("")
            
    except Exception as e:
        click.echo(f"Error querying database: {e}")


@cli.command()
@click.pass_context
def db_stats(ctx: click.Context) -> None:
    """Show database statistics."""
    from .utils import get_db_manager
    
    db_manager = get_db_manager()
    
    try:
        # Get table counts
        tables = ["recipes", "ingredients", "products", "stores"]
        click.echo("📊 Database Statistics:")
        click.echo("")
        
        for table in tables:
            count = db_manager.get_table_count(table)
            click.echo(f"{table.capitalize()}: {count}")
        
        # Get products by store
        store_stats = db_manager.fetch_all("""
            SELECT store_id, COUNT(*) as count 
            FROM products 
            GROUP BY store_id 
            ORDER BY count DESC
        """)
        
        if store_stats:
            click.echo("")
            click.echo("Products by Store:")
            for stat in store_stats:
                click.echo(f"  {stat['store_id']}: {stat['count']}")
                
    except Exception as e:
        click.echo(f"Error getting database stats: {e}")


@cli.command()
@click.option(
    "--port", 
    type=int, 
    default=8501,
    help="Port for Streamlit app"
)
def web(port: int) -> None:
    """Launch web interface (Streamlit app)."""
    click.echo(f"Launching web interface on port {port}...")
    # TODO: Import and run Streamlit app
    click.echo("Web interface will be implemented with Streamlit")


# Vector Database Commands

@cli.group()
def vector():
    """Vector database operations for product similarity search."""
    pass


@vector.command()
@click.option(
    "--collection", 
    default="grocery_products",
    help="Collection name"
)
def init_vector_db(collection: str) -> None:
    """Initialize vector database and collection."""
    from .vector_db import QdrantVectorDB
    
    try:
        click.echo("🚀 Initializing vector database...")
        
        vector_db = QdrantVectorDB(
            collection_name=collection,
            in_memory=True  # Use in-memory storage for demo
        )
        
        stats = vector_db.get_collection_stats()
        click.echo(f"✅ Vector database initialized successfully!")
        click.echo(f"   Collection: {collection}")
        click.echo(f"   Total vectors: {stats.get('total_vectors', 0)}")
        click.echo(f"   Status: {stats.get('collection_status', 'unknown')}")
        
    except Exception as e:
        click.echo(f"❌ Failed to initialize vector database: {e}")


@vector.command()
@click.option(
    "--query", 
    required=True,
    help="Search query for similar products"
)
@click.option(
    "--limit", 
    type=int, 
    default=10,
    help="Maximum number of results"
)
@click.option(
    "--store", 
    help="Filter by specific store"
)
@click.option(
    "--method", 
    type=click.Choice(["automated_stealth", "human_browser", "clipboard_manual", "api_direct", "mock_data"]),
    help="Filter by collection method"
)
@click.option(
    "--min-confidence", 
    type=float, 
    default=0.5,
    help="Minimum confidence score"
)
def search_similar(query: str, limit: int, store: Optional[str], method: Optional[str], min_confidence: float) -> None:
    """Search for similar products using vector similarity."""
    from .vector_db import QdrantVectorDB
    from .data_models.base import DataCollectionMethod
    
    try:
        click.echo(f"🔍 Searching for products similar to: '{query}'")
        
        vector_db = QdrantVectorDB()
        
        # Set up filters
        store_filter = [store] if store else None
        method_filter = [DataCollectionMethod(method)] if method else None
        
        results = vector_db.search_similar_products(
            query=query,
            limit=limit,
            store_filter=store_filter,
            collection_method_filter=method_filter,
            min_confidence=min_confidence
        )
        
        if not results:
            click.echo("No similar products found")
            return
        
        click.echo(f"📦 Found {len(results)} similar products:")
        click.echo("")
        
        for i, (product, score) in enumerate(results, 1):
            confidence_weight = product.get_collection_confidence_weight()
            
            click.echo(f"{i}. {product.name}")
            click.echo(f"   Store: {product.store_id} | Price: ${product.price}")
            click.echo(f"   Brand: {product.brand or 'N/A'}")
            click.echo(f"   Similarity: {score:.3f} | Confidence: {product.confidence_score:.2f} | Weighted: {confidence_weight:.2f}")
            click.echo(f"   Method: {product.collection_method}")
            if product.keywords:
                click.echo(f"   Keywords: {', '.join(product.keywords[:5])}")
            click.echo("")
            
    except Exception as e:
        click.echo(f"❌ Search failed: {e}")


@vector.command()
def vector_stats() -> None:
    """Show vector database statistics."""
    from .vector_db import QdrantVectorDB
    
    try:
        vector_db = QdrantVectorDB()
        stats = vector_db.get_collection_stats()
        
        click.echo("📊 Vector Database Statistics:")
        click.echo("")
        click.echo(f"Total vectors: {stats.get('total_vectors', 0)}")
        click.echo(f"Collection status: {stats.get('collection_status', 'unknown')}")
        click.echo(f"Optimizer status: {stats.get('optimizer_status', 'unknown')}")
        
        # Method distribution
        method_dist = stats.get('method_distribution', {})
        if method_dist:
            click.echo("")
            click.echo("Collection Methods:")
            for method, count in method_dist.items():
                click.echo(f"  {method}: {count}")
        
        # Indexed fields
        indexed_fields = stats.get('indexed_fields', [])
        if indexed_fields:
            click.echo("")
            click.echo(f"Indexed fields: {', '.join(indexed_fields)}")
            
    except Exception as e:
        click.echo(f"❌ Failed to get vector stats: {e}")


@vector.command()
@click.option(
    "--clipboard-text", 
    help="Clipboard text to parse and add"
)
@click.option(
    "--store-id", 
    default="unknown",
    help="Store identifier"
)
def add_clipboard_product(clipboard_text: Optional[str], store_id: str) -> None:
    """Add product from clipboard text."""
    from .vector_db import QdrantVectorDB, ProductNormalizer
    import pyperclip
    
    try:
        # Get clipboard text if not provided
        if not clipboard_text:
            clipboard_text = pyperclip.paste()
            click.echo(f"📋 Using current clipboard content")
        
        if not clipboard_text.strip():
            click.echo("❌ No clipboard text found")
            return
        
        click.echo(f"🔄 Parsing clipboard text...")
        
        # Parse and normalize
        normalizer = ProductNormalizer()
        product = normalizer.normalize_clipboard_data(clipboard_text, store_id)
        
        if not product:
            click.echo("❌ Could not parse product data from clipboard")
            return
        
        # Add to vector database
        vector_db = QdrantVectorDB()
        vector_db.add_product(product)
        
        click.echo(f"✅ Successfully added product to vector database!")
        click.echo(f"   Name: {product.name}")
        click.echo(f"   Price: ${product.price}")
        click.echo(f"   Store: {product.store_id}")
        click.echo(f"   Confidence: {product.confidence_score:.2f}")
        click.echo(f"   Method: {product.collection_method}")
        
    except Exception as e:
        click.echo(f"❌ Failed to add clipboard product: {e}")


@vector.command()
@click.option(
    "--source", 
    type=click.Choice(["automated_stealth", "human_browser", "clipboard_manual", "api_direct", "mock_data"]),
    help="Show products from specific collection method"
)
@click.option(
    "--limit", 
    type=int, 
    default=20,
    help="Maximum products to show"
)
def list_vector_products(source: Optional[str], limit: int) -> None:
    """List products from vector database."""
    from .vector_db import QdrantVectorDB
    from .data_models.base import DataCollectionMethod
    
    try:
        vector_db = QdrantVectorDB()
        
        if source:
            method = DataCollectionMethod(source)
            results = vector_db.search_similar_products(
                query="*",  # Match all
                limit=limit,
                collection_method_filter=[method],
                min_confidence=0.0
            )
            click.echo(f"📦 Products collected via {source} (limit: {limit}):")
        else:
            results = vector_db.search_similar_products(
                query="*",  # Match all
                limit=limit,
                min_confidence=0.0
            )
            click.echo(f"📦 All products in vector database (limit: {limit}):")
        
        if not results:
            click.echo("No products found")
            return
        
        click.echo("")
        
        for i, (product, _) in enumerate(results, 1):
            confidence_weight = product.get_collection_confidence_weight()
            
            click.echo(f"{i}. {product.name}")
            click.echo(f"   Store: {product.store_id} | Price: ${product.price}")
            click.echo(f"   Confidence: {product.confidence_score:.2f} | Weighted: {confidence_weight:.2f}")
            click.echo(f"   Method: {product.collection_method}")
            click.echo(f"   Created: {product.created_at.strftime('%Y-%m-%d %H:%M')}")
            click.echo("")
            
    except Exception as e:
        click.echo(f"❌ Failed to list vector products: {e}")


@vector.command()
@click.confirmation_option(prompt="Are you sure you want to clear the vector database?")
def clear_vectors() -> None:
    """Clear all vectors from the database."""
    from .vector_db import QdrantVectorDB
    
    try:
        vector_db = QdrantVectorDB()
        success = vector_db.clear_collection()
        
        if success:
            click.echo("✅ Vector database cleared successfully")
        else:
            click.echo("❌ Failed to clear vector database")
            
    except Exception as e:
        click.echo(f"❌ Error clearing vector database: {e}")


# Shopping Optimization Commands

@cli.group()
def optimize():
    """Multi-store shopping optimization using OptimizerAgent."""
    pass


@optimize.command()
@click.option(
    "--ingredients-file", 
    type=click.Path(exists=True),
    help="File containing ingredient list (JSON or text)"
)
@click.option(
    "--ingredients", 
    help="Comma-separated list of ingredients"
)
@click.option(
    "--strategy", 
    type=click.Choice(["cost_only", "convenience", "balanced", "quality_first", "time_efficient", "adaptive"]),
    default="adaptive",
    help="Optimization strategy"
)
@click.option(
    "--max-budget", 
    type=float,
    help="Maximum budget constraint"
)
@click.option(
    "--max-stores", 
    type=int,
    default=3,
    help="Maximum number of stores to visit"
)
@click.option(
    "--preferred-stores", 
    help="Comma-separated list of preferred store IDs"
)
@click.option(
    "--avoid-stores", 
    help="Comma-separated list of stores to avoid"
)
@click.option(
    "--quality-threshold", 
    type=float,
    default=0.7,
    help="Minimum quality threshold for products"
)
@click.option(
    "--output", 
    type=click.Path(),
    help="Output file for optimization results (JSON)"
)
@click.option(
    "--verbose", 
    is_flag=True,
    help="Show detailed optimization process"
)
def shopping_list(ingredients_file: str, ingredients: str, strategy: str, max_budget: float,
                 max_stores: int, preferred_stores: str, avoid_stores: str, 
                 quality_threshold: float, output: str, verbose: bool) -> None:
    """Optimize complete shopping list across multiple stores."""
    from .agents.optimizer_agent import OptimizerAgent, OptimizationCriteria
    from .data_models.ingredient import Ingredient
    from .data_models.base import UnitType
    import asyncio
    import json
    from decimal import Decimal
    
    async def run_optimization():
        try:
            # Parse ingredients
            ingredient_list = []
            
            if ingredients_file:
                with open(ingredients_file, 'r') as f:
                    content = f.read().strip()
                    try:
                        # Try to parse as JSON
                        data = json.loads(content)
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, str):
                                    ingredient_list.append(Ingredient(name=item, quantity=1.0, unit=UnitType.PIECES))
                                elif isinstance(item, dict):
                                    ingredient_list.append(Ingredient(**item))
                    except json.JSONDecodeError:
                        # Parse as line-separated text
                        for line in content.split('\n'):
                            line = line.strip()
                            if line:
                                ingredient_list.append(Ingredient(name=line, quantity=1.0, unit=UnitType.PIECES))
            
            elif ingredients:
                for ingredient_name in ingredients.split(','):
                    ingredient_name = ingredient_name.strip()
                    if ingredient_name:
                        ingredient_list.append(Ingredient(name=ingredient_name, quantity=1.0, unit=UnitType.PIECES))
            
            else:
                click.echo("❌ Please provide either --ingredients-file or --ingredients")
                return
            
            if not ingredient_list:
                click.echo("❌ No ingredients found")
                return
            
            # Create optimization criteria
            criteria = OptimizationCriteria(
                max_budget=Decimal(str(max_budget)) if max_budget else None,
                max_stores=max_stores,
                preferred_stores=preferred_stores.split(',') if preferred_stores else [],
                avoid_stores=avoid_stores.split(',') if avoid_stores else [],
                quality_threshold=quality_threshold
            )
            
            click.echo(f"🛒 Optimizing shopping list for {len(ingredient_list)} ingredients...")
            if verbose:
                click.echo(f"   Strategy: {strategy}")
                click.echo(f"   Max budget: ${max_budget}" if max_budget else "   Budget: No limit")
                click.echo(f"   Max stores: {max_stores}")
                click.echo(f"   Quality threshold: {quality_threshold}")
            
            # Initialize optimizer
            optimizer = OptimizerAgent()
            
            # Progress callback
            def progress_callback(message: str):
                if verbose:
                    click.echo(f"   {message}")
            
            # Run optimization
            result = await optimizer.optimize_shopping_list(
                ingredients=ingredient_list,
                criteria=criteria,
                strategy=strategy
            )
            
            if result['success']:
                # Display results
                recommended_trips = result['recommended_strategy']
                savings_analysis = result['savings_analysis']
                optimization_summary = result['optimization_summary']
                
                click.echo(f"✅ Optimization complete!")
                click.echo(f"   Selected strategy: {optimization_summary['selected_strategy']}")
                click.echo(f"   Total stores: {optimization_summary['total_stores']}")
                click.echo(f"   Total cost: ${savings_analysis['optimized_cost']:.2f}")
                click.echo(f"   Total savings: ${savings_analysis['total_savings']:.2f} ({savings_analysis['savings_percentage']:.1f}%)")
                click.echo(f"   Coverage: {optimization_summary['coverage_percentage']:.1f}%")
                click.echo(f"   Estimated time: {optimization_summary['total_time']} minutes")
                
                click.echo("\n🏪 Shopping Plan:")
                for i, trip in enumerate(recommended_trips, 1):
                    click.echo(f"\n{i}. {trip['store_name']} ({trip['store_id']})")
                    click.echo(f"   Items: {trip['total_items']}")
                    click.echo(f"   Cost: ${trip['total_cost']:.2f}")
                    click.echo(f"   Time: {trip['estimated_time']} minutes")
                    
                    if verbose:
                        click.echo("   Products:")
                        for product_info in trip['products']:
                            product = product_info['product']
                            ingredient = product_info['ingredient']
                            price_display = f"${product['price']}"
                            if product['on_sale'] and product['sale_price']:
                                price_display = f"${product['sale_price']} (was ${product['price']}) 🏷️"
                            click.echo(f"     • {ingredient['name']} → {product['name']} - {price_display}")
                
                # Show alternative strategies
                alternatives = result.get('alternative_strategies', {})
                if alternatives and verbose:
                    click.echo("\n📊 Alternative Strategies:")
                    for alt_name, alt_data in alternatives.items():
                        click.echo(f"   {alt_name}: ${alt_data['total_cost']:.2f} ({alt_data['total_stores']} stores, {alt_data['total_time']} min)")
                
                # Show unmatched ingredients
                unmatched = result.get('unmatched_ingredients', [])
                if unmatched:
                    click.echo(f"\n⚠️  Unmatched ingredients ({len(unmatched)}):")
                    for ing in unmatched:
                        click.echo(f"   • {ing['name']} ({ing['quantity']} {ing['unit']})")
                
                # Show selection reasoning
                if verbose and optimization_summary.get('selection_reasoning'):
                    click.echo(f"\n💡 Selection reasoning: {optimization_summary['selection_reasoning']}")
                
                # Save results if requested
                if output:
                    with open(output, 'w') as f:
                        json.dump(result, f, indent=2, default=str)
                    click.echo(f"\n💾 Results saved to {output}")
            
            else:
                click.echo(f"❌ Optimization failed: {result.get('error', 'Unknown error')}")
        
        except Exception as e:
            click.echo(f"❌ Error during optimization: {e}")
    
    asyncio.run(run_optimization())


@optimize.command()
@click.option(
    "--ingredients", 
    required=True,
    help="Comma-separated list of ingredients"
)
@click.option(
    "--max-budget", 
    type=float,
    help="Maximum budget constraint"
)
def cost_only(ingredients: str, max_budget: float) -> None:
    """Pure cost optimization strategy."""
    from .agents.optimizer_agent import OptimizerAgent, OptimizationCriteria
    from .data_models.ingredient import Ingredient
    from .data_models.base import UnitType
    import asyncio
    from decimal import Decimal
    
    async def run_cost_optimization():
        try:
            # Parse ingredients
            ingredient_list = []
            for ingredient_name in ingredients.split(','):
                ingredient_name = ingredient_name.strip()
                if ingredient_name:
                    ingredient_list.append(Ingredient(name=ingredient_name, quantity=1.0, unit=UnitType.PIECES))
            
            criteria = OptimizationCriteria(
                max_budget=Decimal(str(max_budget)) if max_budget else None,
                max_stores=10  # Allow many stores for cost optimization
            )
            
            click.echo(f"💰 Cost-only optimization for {len(ingredient_list)} ingredients...")
            
            optimizer = OptimizerAgent()
            result = await optimizer.optimize_shopping_list(
                ingredients=ingredient_list,
                criteria=criteria,
                strategy="cost_only"
            )
            
            if result['success']:
                savings_analysis = result['savings_analysis']
                click.echo(f"✅ Cheapest option found!")
                click.echo(f"   Total cost: ${savings_analysis['optimized_cost']:.2f}")
                click.echo(f"   Stores needed: {len(result['recommended_strategy'])}")
                click.echo(f"   Savings vs convenience: ${savings_analysis['total_savings']:.2f} ({savings_analysis['savings_percentage']:.1f}%)")
                
                # Show store breakdown
                for trip in result['recommended_strategy']:
                    click.echo(f"   {trip['store_name']}: ${trip['total_cost']:.2f} ({trip['total_items']} items)")
            else:
                click.echo(f"❌ Cost optimization failed: {result.get('error', 'Unknown error')}")
        
        except Exception as e:
            click.echo(f"❌ Error: {e}")
    
    asyncio.run(run_cost_optimization())


@optimize.command()
@click.option(
    "--ingredients", 
    required=True,
    help="Comma-separated list of ingredients"
)
@click.option(
    "--preferred-store", 
    help="Preferred store for convenience shopping"
)
def convenience(ingredients: str, preferred_store: str) -> None:
    """Convenience optimization (single store preferred)."""
    from .agents.optimizer_agent import OptimizerAgent, OptimizationCriteria
    from .data_models.ingredient import Ingredient
    from .data_models.base import UnitType
    import asyncio
    
    async def run_convenience_optimization():
        try:
            # Parse ingredients
            ingredient_list = []
            for ingredient_name in ingredients.split(','):
                ingredient_name = ingredient_name.strip()
                if ingredient_name:
                    ingredient_list.append(Ingredient(name=ingredient_name, quantity=1.0, unit=UnitType.PIECES))
            
            criteria = OptimizationCriteria(
                max_stores=1,
                preferred_stores=[preferred_store] if preferred_store else []
            )
            
            click.echo(f"🚗 Convenience optimization for {len(ingredient_list)} ingredients...")
            if preferred_store:
                click.echo(f"   Preferred store: {preferred_store}")
            
            optimizer = OptimizerAgent()
            result = await optimizer.optimize_shopping_list(
                ingredients=ingredient_list,
                criteria=criteria,
                strategy="convenience"
            )
            
            if result['success'] and result['recommended_strategy']:
                trip = result['recommended_strategy'][0]
                savings_analysis = result['savings_analysis']
                
                click.echo(f"✅ Convenience option found!")
                click.echo(f"   Store: {trip['store_name']}")
                click.echo(f"   Total cost: ${trip['total_cost']:.2f}")
                click.echo(f"   Items available: {trip['total_items']}")
                click.echo(f"   Shopping time: {trip['estimated_time']} minutes")
                
                # Show coverage
                coverage = result['optimization_summary']['coverage_percentage']
                click.echo(f"   Coverage: {coverage:.1f}%")
                
                if coverage < 100:
                    unmatched = result.get('unmatched_ingredients', [])
                    click.echo(f"   Missing items: {', '.join(ing['name'] for ing in unmatched)}")
            else:
                click.echo(f"❌ Convenience optimization failed: {result.get('error', 'Unknown error')}")
        
        except Exception as e:
            click.echo(f"❌ Error: {e}")
    
    asyncio.run(run_convenience_optimization())


@optimize.command()
@click.option(
    "--strategies", 
    default="cost_only,convenience,balanced",
    help="Comma-separated strategies to compare"
)
@click.option(
    "--ingredients", 
    required=True,
    help="Comma-separated list of ingredients"
)
def compare_strategies(strategies: str, ingredients: str) -> None:
    """Compare multiple optimization strategies side-by-side."""
    from .agents.optimizer_agent import OptimizerAgent
    from .data_models.ingredient import Ingredient
    from .data_models.base import UnitType
    import asyncio
    
    async def run_comparison():
        try:
            # Parse ingredients
            ingredient_list = []
            for ingredient_name in ingredients.split(','):
                ingredient_name = ingredient_name.strip()
                if ingredient_name:
                    ingredient_list.append(Ingredient(name=ingredient_name, quantity=1.0, unit=UnitType.PIECES))
            
            strategy_list = [s.strip() for s in strategies.split(',')]
            
            click.echo(f"⚖️  Comparing {len(strategy_list)} strategies for {len(ingredient_list)} ingredients...")
            
            optimizer = OptimizerAgent()
            results = await optimizer.compare_strategies(
                ingredients=ingredient_list,
                strategies=strategy_list
            )
            
            click.echo("\n📊 Strategy Comparison:")
            click.echo("-" * 80)
            
            # Table header
            click.echo(f"{'Strategy':<15} {'Cost':<10} {'Stores':<8} {'Time':<8} {'Coverage':<10} {'Savings':<10}")
            click.echo("-" * 80)
            
            for strategy, result in results.items():
                if result['success']:
                    cost = result['savings_analysis']['optimized_cost']
                    stores = result['optimization_summary']['total_stores']
                    time_min = result['optimization_summary']['total_time']
                    coverage = result['optimization_summary']['coverage_percentage']
                    savings = result['savings_analysis']['savings_percentage']
                    
                    click.echo(f"{strategy:<15} ${cost:<9.2f} {stores:<8} {time_min:<8} {coverage:<9.1f}% {savings:<9.1f}%")
                else:
                    click.echo(f"{strategy:<15} {'FAILED':<50}")
            
            # Highlight best options
            valid_results = {k: v for k, v in results.items() if v['success']}
            if valid_results:
                click.echo("\n🏆 Best Options:")
                
                # Best cost
                best_cost = min(valid_results.items(), key=lambda x: x[1]['savings_analysis']['optimized_cost'])
                click.echo(f"   Lowest cost: {best_cost[0]} (${best_cost[1]['savings_analysis']['optimized_cost']:.2f})")
                
                # Best convenience (fewest stores)
                best_convenience = min(valid_results.items(), key=lambda x: x[1]['optimization_summary']['total_stores'])
                click.echo(f"   Most convenient: {best_convenience[0]} ({best_convenience[1]['optimization_summary']['total_stores']} stores)")
                
                # Best time
                best_time = min(valid_results.items(), key=lambda x: x[1]['optimization_summary']['total_time'])
                click.echo(f"   Fastest: {best_time[0]} ({best_time[1]['optimization_summary']['total_time']} minutes)")
        
        except Exception as e:
            click.echo(f"❌ Error during comparison: {e}")
    
    asyncio.run(run_comparison())


@optimize.command()
def analytics() -> None:
    """Show OptimizerAgent performance analytics."""
    from .agents.optimizer_agent import OptimizerAgent
    
    try:
        optimizer = OptimizerAgent()
        analytics = optimizer.get_optimization_analytics()
        
        click.echo("📊 OptimizerAgent Performance Analytics:")
        click.echo("")
        
        stats = analytics['optimization_stats']
        click.echo(f"Total optimizations: {stats['total_optimizations']}")
        
        if stats['total_optimizations'] > 0:
            click.echo(f"Average savings: {stats['avg_savings_percentage']:.1f}%")
            click.echo(f"Average stores reduced: {stats['avg_stores_reduced']:.1f}")
            
            # Strategy usage
            strategy_perf = stats['strategy_performance']
            if any(perf['uses'] > 0 for perf in strategy_perf.values()):
                click.echo("")
                click.echo("Strategy Usage:")
                for strategy, perf in strategy_perf.items():
                    if perf['uses'] > 0:
                        click.echo(f"  {strategy}: {perf['uses']} times")
        
        # Recommendations
        recommendations = analytics['recommendations']
        if recommendations:
            click.echo("")
            click.echo("Recommendations:")
            for rec in recommendations:
                click.echo(f"  • {rec}")
        else:
            click.echo("\n✅ No optimization recommendations at this time")
    
    except Exception as e:
        click.echo(f"❌ Error getting analytics: {e}")


@optimize.command()
@click.option(
    "--ingredients", 
    required=True,
    help="Comma-separated list of ingredients"
)
@click.option(
    "--current-method", 
    type=click.Choice(["convenience", "cost_only", "balanced"]),
    default="convenience",
    help="Current shopping method for comparison"
)
def estimate_savings(ingredients: str, current_method: str) -> None:
    """Estimate potential savings from optimization."""
    from .agents.optimizer_agent import OptimizerAgent
    from .data_models.ingredient import Ingredient
    from .data_models.base import UnitType
    import asyncio
    
    async def run_estimate():
        try:
            # Parse ingredients
            ingredient_list = []
            for ingredient_name in ingredients.split(','):
                ingredient_name = ingredient_name.strip()
                if ingredient_name:
                    ingredient_list.append(Ingredient(name=ingredient_name, quantity=1.0, unit=UnitType.PIECES))
            
            click.echo(f"💡 Estimating savings potential for {len(ingredient_list)} ingredients...")
            click.echo(f"   Current method: {current_method}")
            
            optimizer = OptimizerAgent()
            estimate = await optimizer.estimate_savings(
                ingredients=ingredient_list,
                current_shopping_method=current_method
            )
            
            if estimate.get('success', True):
                current_cost = estimate['current_cost']
                optimized_cost = estimate['optimized_cost']
                potential_savings = estimate['potential_savings']
                savings_percentage = estimate['savings_percentage']
                recommendation = estimate['recommendation']
                
                click.echo(f"✅ Savings estimate:")
                click.echo(f"   Current cost ({current_method}): ${current_cost:.2f}")
                click.echo(f"   Optimized cost: ${optimized_cost:.2f}")
                click.echo(f"   Potential savings: ${potential_savings:.2f} ({savings_percentage:.1f}%)")
                
                if recommendation == "optimization_worthwhile":
                    click.echo("   💰 Recommendation: Optimization could provide significant savings!")
                else:
                    click.echo("   ℹ️  Recommendation: Current method is already quite efficient")
            else:
                click.echo(f"❌ Estimation failed: {estimate.get('error', 'Unknown error')}")
        
        except Exception as e:
            click.echo(f"❌ Error during estimation: {e}")
    
    asyncio.run(run_estimate())


# Ingredient Matching Commands

@cli.group()
def match():
    """Ingredient-to-product matching operations using MatcherAgent."""
    pass


@match.command()
@click.option(
    "--ingredient", 
    required=True,
    help="Ingredient name to match"
)
@click.option(
    "--quantity", 
    type=float,
    default=1.0,
    help="Quantity needed"
)
@click.option(
    "--unit", 
    default="pieces",
    help="Unit of measurement"
)
@click.option(
    "--category", 
    help="Ingredient category (e.g., dairy, meat, produce)"
)
@click.option(
    "--alternatives", 
    help="Alternative names (comma-separated)"
)
@click.option(
    "--strategy", 
    type=click.Choice(["vector_only", "llm_only", "hybrid", "adaptive"]),
    default="adaptive",
    help="Matching strategy to use"
)
@click.option(
    "--confidence-threshold", 
    type=float,
    default=0.5,
    help="Minimum confidence score for matches"
)
@click.option(
    "--max-results", 
    type=int,
    default=5,
    help="Maximum number of matches to return"
)
@click.option(
    "--verbose", 
    is_flag=True,
    help="Show detailed matching information"
)
def ingredient(ingredient: str, quantity: float, unit: str, category: str, 
              alternatives: str, strategy: str, confidence_threshold: float, 
              max_results: int, verbose: bool) -> None:
    """Match a single ingredient to products."""
    from .agents.matcher_agent import MatcherAgent
    from .data_models.ingredient import Ingredient
    from .data_models.base import UnitType
    import asyncio
    
    async def run_matching():
        try:
            click.echo(f"🔍 Matching ingredient: '{ingredient}'")
            if verbose:
                click.echo(f"   Strategy: {strategy}")
                click.echo(f"   Confidence threshold: {confidence_threshold}")
                click.echo(f"   Max results: {max_results}")
            
            # Create ingredient object
            ingredient_obj = Ingredient(
                name=ingredient,
                quantity=quantity,
                unit=UnitType(unit),
                category=category,
                alternatives=alternatives.split(',') if alternatives else []
            )
            
            # Initialize matcher
            matcher = MatcherAgent()
            
            # Progress callback
            def progress_callback(message: str):
                if verbose:
                    click.echo(f"   {message}")
            
            # Perform matching
            result = await matcher.match_ingredient(
                ingredient=ingredient_obj,
                strategy=strategy,
                confidence_threshold=confidence_threshold,
                max_results=max_results
            )
            
            if result['success']:
                matches = result['matches']
                click.echo(f"✅ Found {len(matches)} matches:")
                click.echo("")
                
                for i, match in enumerate(matches, 1):
                    product = match['product']
                    confidence = match['confidence']
                    quality = match['quality'].value
                    
                    # Format price display
                    price_display = f"${product.price}"
                    if product.on_sale and product.sale_price:
                        price_display = f"${product.sale_price} (was ${product.price}) 🏷️"
                    
                    click.echo(f"{i}. {product.name}")
                    click.echo(f"   Brand: {product.brand or 'Generic'}")
                    click.echo(f"   Store: {product.store_id}")
                    click.echo(f"   Price: {price_display}")
                    click.echo(f"   Confidence: {confidence:.3f} ({quality})")
                    
                    if verbose:
                        click.echo(f"   Category: {product.category or 'N/A'}")
                        click.echo(f"   Collection method: {product.collection_method}")
                        click.echo(f"   Vector score: {match.get('vector_score', 'N/A'):.3f}")
                        click.echo(f"   LLM confidence: {match.get('llm_confidence', 'N/A'):.3f}")
                        if 'llm_reason' in match:
                            click.echo(f"   Reason: {match['llm_reason']}")
                        if 'category_warning' in match:
                            click.echo(f"   ⚠️  {match['category_warning']}")
                    click.echo("")
                
                # Show substitution suggestions
                substitutions = result.get('substitution_suggestions', [])
                if substitutions:
                    click.echo(f"🔄 Alternative suggestions ({len(substitutions)}):")
                    for i, sub in enumerate(substitutions, 1):
                        sub_product = sub['product']
                        click.echo(f"{i}. {sub_product.name} - ${sub_product.price}")
                        click.echo(f"   Type: {sub['type']}")
                        click.echo(f"   Reason: {sub['reason']}")
                        click.echo(f"   Confidence: {sub['confidence']:.3f}")
                    click.echo("")
                
                # Show human review flag
                if result.get('require_human_review', False):
                    reason = result.get('matching_metadata', {}).get('human_review_reason', 'Unknown')
                    click.echo(f"👤 Human review recommended: {reason}")
                
                # Show category analysis
                if verbose and result.get('category_analysis'):
                    category_info = result['category_analysis']
                    click.echo("📂 Category Analysis:")
                    click.echo(f"   Expected: {category_info.get('ingredient_category', 'N/A')}")
                    click.echo(f"   Found categories: {list(category_info.get('product_categories', {}).keys())}")
                    click.echo(f"   Category consistency: {category_info.get('category_consistency', False)}")
                
            else:
                click.echo(f"❌ Matching failed: {result.get('error', 'Unknown error')}")
        
        except Exception as e:
            click.echo(f"❌ Error during matching: {e}")
    
    asyncio.run(run_matching())


@match.command()
@click.option(
    "--ingredients-file", 
    type=click.Path(exists=True),
    help="File containing ingredient list (one per line or JSON)"
)
@click.option(
    "--ingredients", 
    help="Comma-separated list of ingredients"
)
@click.option(
    "--strategy", 
    type=click.Choice(["vector_only", "llm_only", "hybrid", "adaptive"]),
    default="adaptive",
    help="Matching strategy to use"
)
@click.option(
    "--confidence-threshold", 
    type=float,
    default=0.5,
    help="Minimum confidence score"
)
@click.option(
    "--max-results", 
    type=int,
    default=3,
    help="Maximum matches per ingredient"
)
@click.option(
    "--output", 
    type=click.Path(),
    help="Output file for results (JSON)"
)
def batch(ingredients_file: str, ingredients: str, strategy: str, 
          confidence_threshold: float, max_results: int, output: str) -> None:
    """Match multiple ingredients in batch."""
    from .agents.matcher_agent import MatcherAgent
    from .data_models.ingredient import Ingredient
    from .data_models.base import UnitType
    import asyncio
    import json
    
    async def run_batch_matching():
        try:
            # Parse ingredients
            ingredient_list = []
            
            if ingredients_file:
                with open(ingredients_file, 'r') as f:
                    content = f.read().strip()
                    try:
                        # Try to parse as JSON
                        data = json.loads(content)
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, str):
                                    ingredient_list.append(Ingredient(name=item, quantity=1.0, unit=UnitType.PIECES))
                                elif isinstance(item, dict):
                                    ingredient_list.append(Ingredient(**item))
                    except json.JSONDecodeError:
                        # Parse as line-separated text
                        for line in content.split('\n'):
                            line = line.strip()
                            if line:
                                ingredient_list.append(Ingredient(name=line, quantity=1.0, unit=UnitType.PIECES))
            
            elif ingredients:
                for ingredient_name in ingredients.split(','):
                    ingredient_name = ingredient_name.strip()
                    if ingredient_name:
                        ingredient_list.append(Ingredient(name=ingredient_name, quantity=1.0, unit=UnitType.PIECES))
            
            else:
                click.echo("❌ Please provide either --ingredients-file or --ingredients")
                return
            
            if not ingredient_list:
                click.echo("❌ No ingredients found")
                return
            
            click.echo(f"🔍 Batch matching {len(ingredient_list)} ingredients...")
            
            # Initialize matcher
            matcher = MatcherAgent()
            
            # Progress tracking
            results = []
            
            for i, ingredient in enumerate(ingredient_list, 1):
                click.echo(f"[{i}/{len(ingredient_list)}] Matching '{ingredient.name}'...")
                
                result = await matcher.match_ingredient(
                    ingredient=ingredient,
                    strategy=strategy,
                    confidence_threshold=confidence_threshold,
                    max_results=max_results
                )
                
                results.append({
                    'ingredient': ingredient.name,
                    'success': result['success'],
                    'matches': len(result.get('matches', [])),
                    'result': result
                })
                
                if result['success'] and result['matches']:
                    best_match = result['matches'][0]
                    click.echo(f"   ✅ Best: {best_match['product'].name} (conf: {best_match['confidence']:.3f})")
                else:
                    click.echo(f"   ❌ No matches found")
            
            # Summary
            successful = sum(1 for r in results if r['success'] and r['matches'] > 0)
            click.echo(f"\n📊 Batch matching complete:")
            click.echo(f"   Total ingredients: {len(ingredient_list)}")
            click.echo(f"   Successfully matched: {successful}")
            click.echo(f"   Failed to match: {len(ingredient_list) - successful}")
            
            # Save results if requested
            if output:
                with open(output, 'w') as f:
                    # Convert results to JSON-serializable format
                    json_results = []
                    for result in results:
                        json_result = {
                            'ingredient': result['ingredient'],
                            'success': result['success'],
                            'matches': result['matches']
                        }
                        
                        if result['result']['success']:
                            json_result['products'] = []
                            for match in result['result']['matches']:
                                product = match['product']
                                json_result['products'].append({
                                    'name': product.name,
                                    'brand': product.brand,
                                    'price': float(product.price),
                                    'store': product.store_id,
                                    'confidence': match['confidence'],
                                    'quality': match['quality'].value
                                })
                        
                        json_results.append(json_result)
                    
                    json.dump(json_results, f, indent=2)
                    click.echo(f"💾 Results saved to {output}")
        
        except Exception as e:
            click.echo(f"❌ Batch matching failed: {e}")
    
    asyncio.run(run_batch_matching())


@match.command()
@click.option(
    "--ingredient", 
    required=True,
    help="Ingredient name for substitution suggestions"
)
@click.option(
    "--max-suggestions", 
    type=int,
    default=5,
    help="Maximum number of suggestions"
)
def substitutions(ingredient: str, max_suggestions: int) -> None:
    """Get substitution suggestions for an ingredient."""
    from .agents.matcher_agent import MatcherAgent
    import asyncio
    
    async def run_substitutions():
        try:
            click.echo(f"🔄 Finding substitutions for: '{ingredient}'")
            
            matcher = MatcherAgent()
            
            suggestions = await matcher.suggest_substitutions(
                ingredient_name=ingredient,
                max_suggestions=max_suggestions
            )
            
            if suggestions:
                click.echo(f"✅ Found {len(suggestions)} substitution suggestions:")
                click.echo("")
                
                for i, suggestion in enumerate(suggestions, 1):
                    product = suggestion['product']
                    sub_type = suggestion['type']
                    reason = suggestion['reason']
                    confidence = suggestion['confidence']
                    
                    click.echo(f"{i}. {product.name}")
                    click.echo(f"   Brand: {product.brand or 'Generic'}")
                    click.echo(f"   Store: {product.store_id}")
                    click.echo(f"   Price: ${product.price}")
                    click.echo(f"   Type: {sub_type}")
                    click.echo(f"   Reason: {reason}")
                    click.echo(f"   Confidence: {confidence:.3f}")
                    click.echo("")
            else:
                click.echo("❌ No substitution suggestions found")
        
        except Exception as e:
            click.echo(f"❌ Error getting substitutions: {e}")
    
    asyncio.run(run_substitutions())


@match.command()
def analytics() -> None:
    """Show MatcherAgent performance analytics."""
    from .agents.matcher_agent import MatcherAgent
    
    try:
        matcher = MatcherAgent()
        analytics = matcher.get_matching_analytics()
        
        click.echo("📊 MatcherAgent Performance Analytics:")
        click.echo("")
        
        stats = analytics['matching_stats']
        click.echo(f"Total matches attempted: {stats['total_matches']}")
        click.echo(f"Successful matches: {stats['successful_matches']}")
        click.echo(f"Failed matches: {stats['failed_matches']}")
        
        if stats['total_matches'] > 0:
            success_rate = stats['successful_matches'] / stats['total_matches']
            click.echo(f"Success rate: {success_rate:.1%}")
        
        click.echo(f"Average confidence: {stats['avg_confidence']:.3f}")
        
        # Quality distribution
        quality_dist = stats['quality_distribution']
        if any(count > 0 for count in quality_dist.values()):
            click.echo("")
            click.echo("Quality Distribution:")
            for quality, count in quality_dist.items():
                if count > 0:
                    click.echo(f"  {quality}: {count}")
        
        # Strategy performance
        strategy_perf = stats['strategy_performance']
        if any(perf['attempts'] > 0 for perf in strategy_perf.values()):
            click.echo("")
            click.echo("Strategy Performance:")
            for strategy, perf in strategy_perf.items():
                if perf['attempts'] > 0:
                    success_rate = perf['successes'] / perf['attempts']
                    click.echo(f"  {strategy}: {success_rate:.1%} ({perf['successes']}/{perf['attempts']})")
        
        # Recommendations
        recommendations = analytics['recommendations']
        if recommendations:
            click.echo("")
            click.echo("Recommendations:")
            for rec in recommendations:
                click.echo(f"  • {rec}")
    
    except Exception as e:
        click.echo(f"❌ Error getting analytics: {e}")


@match.command()
@click.option(
    "--reset-stats", 
    is_flag=True,
    help="Reset performance statistics"
)
def test(reset_stats: bool) -> None:
    """Test MatcherAgent functionality with sample data."""
    from .agents.matcher_agent import MatcherAgent
    from .data_models.ingredient import Ingredient
    from .data_models.base import UnitType
    import asyncio
    
    async def run_test():
        try:
            click.echo("🧪 Testing MatcherAgent functionality...")
            
            # Test ingredients
            test_ingredients = [
                Ingredient(name="milk", quantity=2.0, unit=UnitType.CUPS, category="dairy"),
                Ingredient(name="bread", quantity=1.0, unit=UnitType.PIECES, category="bakery"),
                Ingredient(name="chicken breast", quantity=1.0, unit=UnitType.POUNDS, category="meat")
            ]
            
            matcher = MatcherAgent()
            
            click.echo(f"Testing with {len(test_ingredients)} sample ingredients...")
            
            for i, ingredient in enumerate(test_ingredients, 1):
                click.echo(f"\n[Test {i}] Matching '{ingredient.name}':")
                
                result = await matcher.match_ingredient(
                    ingredient=ingredient,
                    strategy="adaptive",
                    confidence_threshold=0.3,
                    max_results=2
                )
                
                if result['success']:
                    matches = result['matches']
                    click.echo(f"  ✅ Found {len(matches)} matches")
                    
                    for match in matches:
                        product = match['product']
                        confidence = match['confidence']
                        click.echo(f"    • {product.name} - ${product.price} (conf: {confidence:.3f})")
                else:
                    click.echo(f"  ❌ Failed: {result.get('error', 'Unknown error')}")
            
            # Test analytics
            analytics = matcher.get_matching_analytics()
            stats = analytics['matching_stats']
            
            click.echo(f"\n📊 Test Results:")
            click.echo(f"  Total attempts: {stats['total_matches']}")
            click.echo(f"  Successful: {stats['successful_matches']}")
            click.echo(f"  Average confidence: {stats['avg_confidence']:.3f}")
            
            click.echo("\n✅ MatcherAgent test completed successfully!")
        
        except Exception as e:
            click.echo(f"❌ Test failed: {e}")
    
    asyncio.run(run_test())


@vector.command()
@click.option(
    "--method", 
    type=click.Choice(["stealth", "human_browser", "clipboard"]),
    required=True,
    help="Test integration with specific scraping method"
)
@click.option(
    "--query", 
    default="milk",
    help="Test query for demonstration"
)
def test_integration(method: str, query: str) -> None:
    """Test vector integration with scraping layers."""
    from .vector_db import ScraperVectorIntegration
    from .data_models.base import DataCollectionMethod
    import asyncio
    
    async def run_test():
        try:
            click.echo(f"🧪 Testing {method} integration with vector database...")
            
            integration = ScraperVectorIntegration()
            
            if method == "stealth":
                # Mock stealth scraper data
                mock_data = [{
                    "name": f"Test {query.title()}",
                    "brand": "Test Brand",
                    "price": "4.99",
                    "store_id": "test_store",
                    "category": "dairy",
                    "in_stock": True
                }]
                
                products = await integration.process_stealth_scraper_results(
                    mock_data, "test_store", "test_session"
                )
                
            elif method == "human_browser":
                # Mock human browser data
                mock_data = [{
                    "name": f"Premium {query.title()}",
                    "brand": "Premium Brand",
                    "price": "6.99",
                    "store_id": "test_store",
                    "category": "dairy",
                    "description": "High quality product",
                    "in_stock": True
                }]
                
                products = await integration.process_human_browser_results(
                    mock_data, "test_store", "test_session"
                )
                
            elif method == "clipboard":
                # Mock clipboard data
                clipboard_text = f"""Premium {query.title()}
                Brand: Test Brand
                Price: $5.99
                Size: 1L
                In Stock
                https://test-store.com/product/123"""
                
                products = await integration.process_clipboard_data(
                    [clipboard_text], "test_session"
                )
            
            if products:
                click.echo(f"✅ Successfully processed {len(products)} products")
                
                # Test similarity search
                similar = await integration.find_similar_products(query, limit=3)
                
                if similar:
                    click.echo(f"🔍 Found {len(similar)} similar products:")
                    for product, score in similar:
                        click.echo(f"   - {product.name} (score: {score:.3f}, method: {product.collection_method})")
                
                # Show stats
                stats = integration.get_collection_statistics()
                method_dist = stats.get('method_distribution', {})
                click.echo(f"📊 Total vectors in database: {stats.get('total_vectors', 0)}")
                if method_dist:
                    click.echo(f"   Method distribution: {method_dist}")
                    
            else:
                click.echo("❌ No products were processed")
                
        except Exception as e:
            click.echo(f"❌ Integration test failed: {e}")
    
    asyncio.run(run_test())


# ===== WORKFLOW COMMANDS =====

@cli.group()
def workflow() -> None:
    """Master workflow commands for coordinating all agents."""
    pass


@workflow.command("run-complete")
@click.option(
    "--recipes-file",
    type=click.Path(exists=True),
    help="JSON file containing recipes to process"
)
@click.option(
    "--ingredients",
    type=str,
    help="Comma-separated list of ingredients"
)
@click.option(
    "--ingredients-file", 
    type=click.Path(exists=True),
    help="File containing ingredients (one per line or JSON)"
)
@click.option(
    "--scraping-strategy",
    type=click.Choice(["auto_stealth", "human_assisted", "clipboard_manual", "hybrid", "adaptive"]),
    default="adaptive",
    help="Scraping strategy to use"
)
@click.option(
    "--matching-strategy",
    type=click.Choice(["vector_only", "llm_only", "hybrid", "adaptive"]),
    default="adaptive", 
    help="Matching strategy to use"
)
@click.option(
    "--optimization-strategy",
    type=click.Choice(["cost_only", "convenience", "balanced", "quality_first", "time_efficient", "adaptive"]),
    default="adaptive",
    help="Optimization strategy to use"
)
@click.option(
    "--stores",
    type=str,
    default="metro_ca,walmart_ca,freshco_com",
    help="Comma-separated list of store IDs to search"
)
@click.option(
    "--max-budget",
    type=float,
    help="Maximum budget for shopping"
)
@click.option(
    "--max-stores",
    type=int,
    default=3,
    help="Maximum number of stores to visit"
)
@click.option(
    "--preferred-stores",
    type=str,
    help="Comma-separated list of preferred store IDs"
)
@click.option(
    "--parallel/--sequential",
    default=True,
    help="Enable parallel processing"
)
@click.option(
    "--timeout",
    type=int,
    default=300,
    help="Workflow timeout in seconds"
)
@click.option(
    "--output",
    type=click.Path(),
    help="Save results to JSON file"
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Show detailed progress information"
)
def run_complete(
    recipes_file: Optional[str],
    ingredients: Optional[str], 
    ingredients_file: Optional[str],
    scraping_strategy: str,
    matching_strategy: str,
    optimization_strategy: str,
    stores: str,
    max_budget: Optional[float],
    max_stores: int,
    preferred_stores: Optional[str],
    parallel: bool,
    timeout: int,
    output: Optional[str],
    verbose: bool
) -> None:
    """Execute the complete grocery workflow from recipes to optimized shopping list."""
    
    async def run_workflow():
        from .workflow import GroceryWorkflow
        from .data_models import Recipe, Ingredient
        from decimal import Decimal
        import json
        
        # Parse input data
        recipes_list = []
        ingredients_list = []
        
        if recipes_file:
            with open(recipes_file) as f:
                recipes_data = json.load(f)
                for recipe_data in recipes_data:
                    recipes_list.append(Recipe(**recipe_data))
        
        if ingredients_file:
            with open(ingredients_file) as f:
                content = f.read().strip()
                try:
                    # Try JSON first
                    data = json.loads(content)
                    if isinstance(data, list):
                        ingredients_list.extend(data)
                except json.JSONDecodeError:
                    # Parse as lines
                    ingredients_list.extend(line.strip() for line in content.split('\n') if line.strip())
        
        if ingredients:
            ingredients_list.extend(ing.strip() for ing in ingredients.split(',') if ing.strip())
        
        if not recipes_list and not ingredients_list:
            click.echo("❌ Please provide recipes or ingredients")
            return
        
        # Build configuration
        config = {
            "scraping_strategy": scraping_strategy,
            "matching_strategy": matching_strategy,
            "optimization_strategy": optimization_strategy,
            "target_stores": [s.strip() for s in stores.split(',')],
            "max_stores": max_stores,
            "enable_parallel_scraping": parallel,
            "enable_parallel_matching": parallel,
            "workflow_timeout": timeout,
        }
        
        if max_budget:
            config["max_budget"] = max_budget
        
        if preferred_stores:
            config["preferred_stores"] = [s.strip() for s in preferred_stores.split(',')]
        
        # Progress callback
        def progress_callback(progress_info):
            if verbose:
                stage = progress_info.get("stage", "unknown")
                message = progress_info.get("message", "")
                if "ingredient" in progress_info:
                    click.echo(f"🔄 [{stage}] {progress_info['ingredient']}: {message}")
                else:
                    click.echo(f"🔄 [{stage}] {message}")
        
        # Initialize workflow
        click.echo("🚀 Initializing master grocery workflow...")
        workflow = GroceryWorkflow()
        
        try:
            # Execute workflow
            click.echo(f"📋 Processing {len(recipes_list)} recipes and {len(ingredients_list)} ingredients...")
            
            result = await workflow.execute(
                recipes=recipes_list or None,
                ingredients=ingredients_list or None,
                config=config,
                progress_callback=progress_callback if verbose else None
            )
            
            # Display results
            metrics = result["execution_metrics"]
            summary = result.get("workflow_summary", {})
            
            click.echo("\n✅ Workflow completed successfully!")
            click.echo(f"⏱️  Total execution time: {metrics.total_execution_time:.2f} seconds")
            click.echo(f"🥘 Ingredients processed: {summary.get('ingredients_processed', 0)}")
            click.echo(f"🛒 Products collected: {summary.get('products_collected', 0)}")
            click.echo(f"🎯 Matches found: {summary.get('matches_found', 0)}")
            
            success_rates = summary.get("success_rates", {})
            click.echo(f"📊 Success rates - Scraping: {success_rates.get('scraping', 0):.1%}, "
                      f"Matching: {success_rates.get('matching', 0):.1%}, "
                      f"Optimization: {success_rates.get('optimization', 0):.1%}")
            
            # Show optimization results
            optimization_results = result.get("optimization_results")
            if optimization_results:
                recommended_strategy = optimization_results.get("recommended_strategy", [])
                total_savings = optimization_results.get("total_savings", 0)
                
                click.echo(f"\n💰 Recommended shopping strategy ({len(recommended_strategy)} trips):")
                for trip in recommended_strategy:
                    click.echo(f"  📍 {trip.get('store_name', 'Unknown Store')}: "
                              f"{trip.get('total_items', 0)} items, "
                              f"${trip.get('total_cost', 0):.2f}")
                
                if total_savings > 0:
                    click.echo(f"💸 Estimated savings: ${total_savings:.2f} "
                              f"({optimization_results.get('savings_percentage', 0):.1f}%)")
            
            # Save to file if requested
            if output:
                with open(output, 'w') as f:
                    # Convert Decimal to float for JSON serialization
                    json_result = json.loads(json.dumps(result, default=str))
                    json.dump(json_result, f, indent=2)
                click.echo(f"💾 Results saved to {output}")
            
        except Exception as e:
            click.echo(f"❌ Workflow failed: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
    
    asyncio.run(run_workflow())


@workflow.command("status")
@click.argument("execution_id", required=False)
def status(execution_id: Optional[str]) -> None:
    """Check status of running or completed workflow executions."""
    from .workflow import GroceryWorkflow
    
    workflow = GroceryWorkflow()
    
    if execution_id:
        # Check specific execution
        status_info = workflow.get_execution_status(execution_id)
        if status_info:
            click.echo(f"📊 Execution {execution_id}:")
            click.echo(f"   Status: {status_info['status']}")
            if "current_stage" in status_info:
                click.echo(f"   Stage: {status_info['current_stage']}")
            click.echo(f"   Time: {status_info['execution_time']:.2f}s")
            if status_info.get("errors", 0) > 0:
                click.echo(f"   Errors: {status_info['errors']}")
        else:
            click.echo(f"❌ Execution {execution_id} not found")
    else:
        # Show active executions
        if workflow.active_executions:
            click.echo(f"🔄 Active executions ({len(workflow.active_executions)}):")
            for exec_id, state in workflow.active_executions.items():
                click.echo(f"   {exec_id}: {state['workflow_status']} ({state.get('current_stage', 'unknown')})")
        else:
            click.echo("💤 No active executions")
        
        # Show recent history
        recent = workflow.execution_history[-5:]  # Last 5 executions
        if recent:
            click.echo(f"\n📚 Recent executions:")
            for metrics in recent:
                click.echo(f"   {metrics.execution_id}: {metrics.status} ({metrics.total_execution_time:.2f}s)")


@workflow.command("cancel")
@click.argument("execution_id")
def cancel(execution_id: str) -> None:
    """Cancel a running workflow execution."""
    from .workflow import GroceryWorkflow
    
    workflow = GroceryWorkflow()
    
    if workflow.cancel_execution(execution_id):
        click.echo(f"✅ Cancelled execution {execution_id}")
    else:
        click.echo(f"❌ Execution {execution_id} not found or already completed")


@workflow.command("performance")
def performance() -> None:
    """Show performance summary across all workflow executions."""
    from .workflow import GroceryWorkflow
    
    workflow = GroceryWorkflow()
    summary = workflow.get_performance_summary()
    
    if "message" in summary:
        click.echo(f"ℹ️  {summary['message']}")
        return
    
    click.echo("📈 Workflow Performance Summary:")
    click.echo(f"   Total executions: {summary['total_executions']}")
    click.echo(f"   Completed: {summary['completed_executions']}")
    click.echo(f"   Avg execution time: {summary['avg_execution_time']:.2f}s")
    click.echo(f"   Avg ingredients: {summary['avg_ingredients_processed']:.1f}")
    click.echo(f"   Avg products collected: {summary['avg_products_collected']:.1f}")
    
    success_rates = summary.get('avg_success_rates', {})
    click.echo("📊 Average success rates:")
    click.echo(f"   Scraping: {success_rates.get('scraping', 0):.1%}")
    click.echo(f"   Matching: {success_rates.get('matching', 0):.1%}")
    click.echo(f"   Optimization: {success_rates.get('optimization', 0):.1%}")


@workflow.command("demo")
@click.option(
    "--scenario",
    type=click.Choice(["quick", "family-dinner", "meal-prep", "party", "multi-recipe"]),
    default="quick",
    help="Demo scenario to run"
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Show detailed progress"
)
def demo(scenario: str, verbose: bool) -> None:
    """Run workflow demonstration with predefined scenarios."""
    
    scenarios = {
        "quick": {
            "name": "Quick Shopping",
            "ingredients": ["milk", "bread", "eggs"],
            "description": "Simple 3-ingredient demo"
        },
        "family-dinner": {
            "name": "Family Dinner",
            "ingredients": ["chicken breast", "broccoli", "rice", "olive oil", "garlic", "onion"],
            "description": "6-ingredient family meal"
        },
        "meal-prep": {
            "name": "Weekly Meal Prep", 
            "ingredients": [
                "ground turkey", "sweet potatoes", "spinach", "quinoa",
                "Greek yogurt", "berries", "almonds", "avocados",
                "bell peppers", "carrots", "brown rice"
            ],
            "description": "11-ingredient meal prep list"
        },
        "party": {
            "name": "Party Shopping",
            "ingredients": [
                "chips", "dip", "soda", "beer", "cheese", "crackers",
                "grapes", "strawberries", "chocolate", "ice cream",
                "paper plates", "napkins", "plastic cups"
            ],
            "description": "13-ingredient party supplies"
        },
        "multi-recipe": {
            "name": "Multi-Recipe Demo",
            "recipes": [
                {
                    "name": "Breakfast Smoothie",
                    "servings": 2,
                    "ingredients": [
                        {"name": "banana", "quantity": 2, "unit": "pieces"},
                        {"name": "Greek yogurt", "quantity": 1, "unit": "cup"},
                        {"name": "berries", "quantity": 0.5, "unit": "cup"},
                        {"name": "honey", "quantity": 1, "unit": "tablespoon"}
                    ]
                },
                {
                    "name": "Pasta Dinner", 
                    "servings": 4,
                    "ingredients": [
                        {"name": "pasta", "quantity": 1, "unit": "pound"},
                        {"name": "ground beef", "quantity": 1, "unit": "pound"},
                        {"name": "tomato sauce", "quantity": 2, "unit": "cups"},
                        {"name": "mozzarella cheese", "quantity": 8, "unit": "ounces"}
                    ]
                }
            ],
            "description": "2 complete recipes (8 total ingredients)"
        }
    }
    
    if scenario not in scenarios:
        click.echo(f"❌ Unknown scenario: {scenario}")
        return
    
    scene = scenarios[scenario]
    click.echo(f"🎬 Running {scene['name']} demo")
    click.echo(f"📝 {scene['description']}")
    click.echo("")
    
    async def run_demo():
        from .workflow import GroceryWorkflow
        from .data_models import Recipe, Ingredient
        import json
        
        # Progress callback
        def progress_callback(progress_info):
            if verbose:
                stage = progress_info.get("stage", "unknown")
                message = progress_info.get("message", "")
                click.echo(f"🔄 [{stage}] {message}")
        
        workflow = GroceryWorkflow()
        
        try:
            config = {
                "scraping_strategy": "adaptive",
                "matching_strategy": "adaptive", 
                "optimization_strategy": "balanced",
                "target_stores": ["metro_ca", "walmart_ca"],
                "max_stores": 2,
                "workflow_timeout": 180  # 3 minutes for demo
            }
            
            recipes_list = None
            ingredients_list = None
            
            if "recipes" in scene:
                recipes_list = [Recipe(**recipe_data) for recipe_data in scene["recipes"]]
            else:
                ingredients_list = scene["ingredients"]
            
            click.echo("🚀 Starting workflow execution...")
            start_time = asyncio.get_event_loop().time()
            
            result = await workflow.execute(
                recipes=recipes_list,
                ingredients=ingredients_list,
                config=config,
                progress_callback=progress_callback if verbose else None
            )
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            # Display results
            click.echo(f"\n✅ Demo completed in {execution_time:.2f} seconds!")
            
            summary = result.get("workflow_summary", {})
            click.echo(f"📊 Results:")
            click.echo(f"   Ingredients: {summary.get('ingredients_processed', 0)}")
            click.echo(f"   Products found: {summary.get('products_collected', 0)}")
            click.echo(f"   Matches: {summary.get('matches_found', 0)}")
            
            optimization_results = result.get("optimization_results")
            if optimization_results:
                recommended = optimization_results.get("recommended_strategy", [])
                if recommended:
                    click.echo(f"   Recommended: {len(recommended)} store visits")
                    total_cost = sum(float(trip.get("total_cost", 0)) for trip in recommended)
                    click.echo(f"   Total cost: ${total_cost:.2f}")
            
            click.echo(f"\n🎯 This demo showcases the complete workflow:")
            click.echo(f"   ✓ Ingredient extraction and validation")
            click.echo(f"   ✓ Parallel product scraping across stores")
            click.echo(f"   ✓ Intelligent ingredient-to-product matching") 
            click.echo(f"   ✓ Multi-store shopping optimization")
            click.echo(f"   ✓ Performance monitoring and error handling")
            
        except Exception as e:
            click.echo(f"❌ Demo failed: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
    
    asyncio.run(run_demo())


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()