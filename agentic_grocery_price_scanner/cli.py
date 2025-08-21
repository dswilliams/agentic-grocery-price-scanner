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


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()