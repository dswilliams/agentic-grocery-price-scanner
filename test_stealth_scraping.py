#!/usr/bin/env python3
"""
Test script for stealth web scraping with anti-bot bypass.
"""

import asyncio
import logging
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agentic_grocery_price_scanner.mcps.stealth_scraper import (
    create_stealth_scraper,
    StealthConfig
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_stealth_connectivity():
    """Test stealth scraping connectivity."""
    logger.info("Testing stealth scraping connectivity...")
    
    try:
        # Configure for stealth mode
        stealth_config = StealthConfig(
            headless=False,  # Show browser for debugging
            enable_stealth=True,
            page_load_delay=(3, 6),  # Longer delays to seem more human
            scroll_delay=(2, 4),
        )
        
        scraper = await create_stealth_scraper(stealth_config=stealth_config)
        
        async with scraper:
            # Test with a simple site first
            logger.info("Testing basic connectivity with httpbin...")
            result = await scraper.scrape_url("https://httpbin.org/headers")
            
            if result.get("success"):
                logger.info("✅ Basic stealth connectivity test PASSED")
                logger.info(f"   - Page title: {result.get('title')}")
                return True
            else:
                logger.error(f"❌ Basic connectivity failed: {result.get('error')}")
                return False
                
    except Exception as e:
        logger.error(f"❌ Stealth connectivity test exception: {e}")
        return False


async def test_metro_stealth_scraping():
    """Test stealth scraping against Metro.ca."""
    logger.info("Testing Metro.ca stealth scraping...")
    
    try:
        # Configure for maximum stealth
        stealth_config = StealthConfig(
            headless=True,  # Use headless for production
            enable_stealth=True,
            browser_type="chromium",
            page_load_delay=(4, 8),  # Even longer delays
            scroll_delay=(2, 5),
            navigation_timeout=60000,  # 60 second timeout
        )
        
        scraper = await create_stealth_scraper(stealth_config=stealth_config)
        
        async with scraper:
            # First, test the main Metro.ca page
            logger.info("Testing Metro.ca main page access...")
            main_result = await scraper.scrape_url("https://www.metro.ca")
            
            if not main_result.get("success"):
                logger.error(f"❌ Failed to access Metro.ca main page: {main_result.get('error')}")
                return False, []
            
            logger.info(f"✅ Successfully accessed Metro.ca main page")
            logger.info(f"   - Title: {main_result.get('title')}")
            
            # Now try product search
            logger.info("Attempting Metro.ca product search...")
            products = await scraper.scrape_products(
                store_id="metro_ca",
                search_term="milk",
                max_products=3
            )
            
            if products:
                logger.info(f"🎉 SUCCESS! Found {len(products)} products from Metro.ca")
                
                for i, product in enumerate(products, 1):
                    logger.info(f"   Product {i}: {product.name} - ${product.price}")
                    
                return True, products
            else:
                logger.warning("⚠️  No products found - might still be blocked or page structure changed")
                return False, []
                
    except Exception as e:
        logger.error(f"❌ Metro.ca stealth scraping failed: {e}")
        return False, []


async def test_walmart_stealth_scraping():
    """Test stealth scraping against Walmart.ca."""
    logger.info("Testing Walmart.ca stealth scraping...")
    
    try:
        stealth_config = StealthConfig(
            headless=True,
            enable_stealth=True,
            page_load_delay=(3, 7),
            scroll_delay=(1, 3),
        )
        
        scraper = await create_stealth_scraper(stealth_config=stealth_config)
        
        async with scraper:
            logger.info("Attempting Walmart.ca product search...")
            products = await scraper.scrape_products(
                store_id="walmart_ca", 
                search_term="bread",
                max_products=3
            )
            
            if products:
                logger.info(f"🎉 SUCCESS! Found {len(products)} products from Walmart.ca")
                
                for i, product in enumerate(products, 1):
                    logger.info(f"   Product {i}: {product.name} - ${product.price}")
                    
                return True, products
            else:
                logger.warning("⚠️  No products found from Walmart.ca")
                return False, []
                
    except Exception as e:
        logger.error(f"❌ Walmart.ca stealth scraping failed: {e}")
        return False, []


async def test_user_agent_rotation():
    """Test user agent rotation."""
    logger.info("Testing user agent rotation...")
    
    try:
        from agentic_grocery_price_scanner.mcps.stealth_scraper import UserAgentRotator
        
        rotator = UserAgentRotator()
        
        # Test multiple user agents
        agents = [rotator.get_random_user_agent() for _ in range(3)]
        unique_agents = set(agents)
        
        logger.info(f"Generated {len(unique_agents)} unique user agents from {len(agents)} requests")
        
        for i, agent in enumerate(unique_agents, 1):
            logger.info(f"   Agent {i}: {agent[:50]}...")
            headers = rotator.get_matching_headers(agent)
            logger.info(f"           Headers: {len(headers)} entries")
        
        return len(unique_agents) > 0
        
    except Exception as e:
        logger.error(f"❌ User agent rotation test failed: {e}")
        return False


async def main():
    """Run all stealth scraping tests."""
    logger.info("Starting Stealth Web Scraping Tests")
    logger.info("=" * 50)
    
    tests = [
        ("User Agent Rotation", test_user_agent_rotation),
        ("Stealth Connectivity", test_stealth_connectivity),
        ("Metro.ca Stealth Scraping", test_metro_stealth_scraping),
        ("Walmart.ca Stealth Scraping", test_walmart_stealth_scraping),
    ]
    
    results = {}
    total_products_found = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n🔍 Running {test_name} test...")
        
        try:
            if "Scraping" in test_name:
                success, products = await test_func()
                results[test_name] = success
                if success:
                    total_products_found += len(products)
            else:
                results[test_name] = await test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("STEALTH SCRAPING TEST SUMMARY")
    logger.info("=" * 50)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    logger.info(f"Total products scraped: {total_products_found}")
    
    if passed >= 2:  # At least connectivity and user agents working
        logger.info("🎉 Stealth scraping infrastructure is working!")
        if total_products_found > 0:
            logger.info("🚀 BREAKTHROUGH: Successfully bypassed bot protection!")
        else:
            logger.info("💡 Infrastructure ready - may need site-specific tuning")
        return 0
    else:
        logger.error("❌ Stealth scraping tests failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)