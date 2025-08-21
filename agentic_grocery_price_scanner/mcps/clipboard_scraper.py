"""
Clipboard monitoring system for easy manual data entry.
Copy product info from any website and automatically parse it into Product objects.
"""

import asyncio
import json
import logging
import re
import subprocess
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
import pyperclip
from datetime import datetime

from pydantic import BaseModel, Field

from ..data_models.product import Product
from ..data_models.base import Currency, UnitType

logger = logging.getLogger(__name__)


class ClipboardProduct(BaseModel):
    """Product data extracted from clipboard."""
    
    raw_text: str
    confidence: float = Field(ge=0, le=1, description="Confidence in extraction accuracy")
    extracted_fields: Dict[str, str] = Field(default_factory=dict)
    suggested_product: Optional[Product] = None


class ClipboardMonitor:
    """Monitor clipboard for product data and automatically parse it."""
    
    def __init__(self, store_id: str = "manual_clipboard"):
        """Initialize clipboard monitor."""
        self.store_id = store_id
        self.last_clipboard = ""
        self.is_monitoring = False
        self.products_collected = []
        
        # Patterns for extracting product information
        self.patterns = {
            "price": [
                r'\$[\d,]+\.?\d{0,2}',  # $19.99, $1,234.56
                r'[\d,]+\.?\d{0,2}\s*(?:CAD|USD|\$)',  # 19.99 CAD
                r'(?:Price|Cost|Total)[:\s]*\$?[\d,]+\.?\d{0,2}',  # Price: $19.99
            ],
            "name": [
                r'^[A-Za-z].*[A-Za-z0-9]$',  # First line that looks like a product name
                r'(?:Product|Item|Name)[:\s]*([A-Za-z].*)',  # Product: Something
            ],
            "brand": [
                r'\b(?:by|from|brand)[:\s]*([A-Za-z][A-Za-z\s&-]*)',  # by Brand Name
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:Brand|Co\.?|Inc\.?|Ltd\.?)',  # Brand Name Co.
            ],
            "size": [
                r'(\d+(?:\.\d+)?)\s*(g|kg|ml|l|lbs?|oz|cups?|pieces?|count)',
                r'(?:Size|Weight|Volume)[:\s]*(\d+(?:\.\d+)?)\s*(g|kg|ml|l|lbs?|oz)',
            ],
            "store": [
                r'(?:at|from|@)\s*([A-Za-z][A-Za-z\s&-]*?)(?:\s|$)',  # at Store Name
                r'\b(Metro|Walmart|Loblaws|Sobeys|FreshCo|No Frills)\b',  # Known stores
            ]
        }
        
    def start_monitoring(self, interval: float = 1.0) -> None:
        """Start monitoring clipboard for changes."""
        self.is_monitoring = True
        logger.info("📋 Clipboard monitoring started!")
        logger.info("💡 Copy product information from any website, and I'll parse it automatically")
        logger.info("   Example: Copy text containing product name and price")
        logger.info("   The system will detect and extract product details")
        
        asyncio.create_task(self._monitor_loop(interval))
        
    async def _monitor_loop(self, interval: float) -> None:
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                current_clipboard = pyperclip.paste()
                
                if current_clipboard != self.last_clipboard and current_clipboard.strip():
                    self.last_clipboard = current_clipboard
                    
                    # Analyze clipboard content
                    clipboard_product = self._analyze_clipboard_content(current_clipboard)
                    
                    if clipboard_product.confidence > 0.3:  # Minimum confidence threshold
                        logger.info(f"\n📋 CLIPBOARD DETECTION (Confidence: {clipboard_product.confidence:.1%})")
                        logger.info(f"Raw text: {current_clipboard[:100]}...")
                        
                        if clipboard_product.suggested_product:
                            product = clipboard_product.suggested_product
                            self.products_collected.append(product)
                            
                            logger.info(f"✅ Extracted Product:")
                            logger.info(f"   Name: {product.name}")
                            logger.info(f"   Price: ${product.price}")
                            logger.info(f"   Brand: {product.brand or 'N/A'}")
                            logger.info(f"   Store: {product.store_id}")
                            
                            # Ask user if this looks correct
                            logger.info("💡 Type 'yes' to keep, 'no' to discard, or 'edit' to modify:")
                            
                        else:
                            logger.info("⚠️  Could not extract complete product info")
                            logger.info(f"Found fields: {clipboard_product.extracted_fields}")
                            
            except Exception as e:
                logger.debug(f"Clipboard monitoring error: {e}")
                
            await asyncio.sleep(interval)
            
    def _analyze_clipboard_content(self, content: str) -> ClipboardProduct:
        """Analyze clipboard content for product information."""
        content = content.strip()
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        extracted = {}
        confidence = 0.0
        
        # Extract price (high confidence indicator)
        price_text = self._extract_with_patterns(content, self.patterns["price"])
        if price_text:
            extracted["price"] = price_text
            confidence += 0.4
            
        # Extract product name
        name = self._extract_product_name(lines, content)
        if name:
            extracted["name"] = name
            confidence += 0.3
            
        # Extract brand
        brand = self._extract_with_patterns(content, self.patterns["brand"])
        if brand:
            extracted["brand"] = brand
            confidence += 0.1
            
        # Extract size/weight
        size_match = self._extract_size(content)
        if size_match:
            extracted["size"] = size_match[0]
            extracted["size_unit"] = size_match[1]
            confidence += 0.1
            
        # Extract store
        store = self._extract_with_patterns(content, self.patterns["store"])
        if store:
            extracted["store"] = store
            confidence += 0.1
            
        # Create suggested product if we have enough info
        suggested_product = None
        if extracted.get("name") and extracted.get("price"):
            try:
                price = self._extract_price(extracted["price"])
                if price > 0:
                    suggested_product = Product(
                        name=extracted["name"],
                        brand=extracted.get("brand"),
                        price=price,
                        currency=Currency.CAD,
                        store_id=extracted.get("store", self.store_id),
                        description=f"Extracted from clipboard at {datetime.now().strftime('%H:%M')}",
                        in_stock=True,
                        keywords=[word.lower() for word in extracted["name"].split() if len(word) > 2]
                    )
            except Exception as e:
                logger.debug(f"Error creating suggested product: {e}")
                
        return ClipboardProduct(
            raw_text=content,
            confidence=min(confidence, 1.0),
            extracted_fields=extracted,
            suggested_product=suggested_product
        )
        
    def _extract_with_patterns(self, content: str, patterns: List[str]) -> Optional[str]:
        """Extract text using regex patterns."""
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
            if match:
                # Return the first capturing group if it exists, otherwise the full match
                return match.group(1) if match.groups() else match.group(0)
        return None
        
    def _extract_product_name(self, lines: List[str], content: str) -> Optional[str]:
        """Extract product name using multiple strategies."""
        # Strategy 1: Look for explicit product name patterns
        for pattern in self.patterns["name"]:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                name = match.group(1) if match.groups() else match.group(0)
                if len(name.split()) >= 2:  # At least 2 words
                    return name.strip()
                    
        # Strategy 2: Use the first substantial line
        for line in lines:
            # Skip lines that look like prices, URLs, or metadata
            if (not re.search(r'\$[\d,]+\.?\d*', line) and
                not line.startswith('http') and
                not line.lower().startswith(('price', 'cost', 'total', 'qty')) and
                len(line.split()) >= 2 and
                len(line) >= 5):
                return line.strip()
                
        return None
        
    def _extract_size(self, content: str) -> Optional[Tuple[str, str]]:
        """Extract size and unit from content."""
        for pattern in self.patterns["size"]:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1), match.group(2).lower()
        return None
        
    def _extract_price(self, price_str: str) -> Decimal:
        """Extract numeric price from string."""
        try:
            # Clean the price string
            cleaned = re.sub(r'[^\d.,]', '', str(price_str))
            if not cleaned:
                return Decimal('0')
                
            # Handle decimal separators
            if ',' in cleaned and '.' in cleaned:
                # Assume comma is thousands separator
                cleaned = cleaned.replace(',', '')
            elif ',' in cleaned:
                # Check if comma is decimal separator
                parts = cleaned.split(',')
                if len(parts) == 2 and len(parts[1]) <= 2:
                    cleaned = cleaned.replace(',', '.')
                else:
                    cleaned = cleaned.replace(',', '')
                    
            return Decimal(cleaned) if cleaned else Decimal('0')
            
        except (ValueError, ArithmeticError):
            return Decimal('0')
            
    def stop_monitoring(self) -> None:
        """Stop clipboard monitoring."""
        self.is_monitoring = False
        logger.info("📋 Clipboard monitoring stopped")
        
    def get_collected_products(self) -> List[Product]:
        """Get all products collected from clipboard."""
        return self.products_collected.copy()
        
    def clear_collected_products(self) -> None:
        """Clear the collected products list."""
        self.products_collected.clear()
        logger.info("🗑️  Collected products cleared")
        
    async def manual_clipboard_session(self, max_products: int = 10) -> List[Product]:
        """Run an interactive clipboard collection session."""
        logger.info("🚀 MANUAL CLIPBOARD COLLECTION SESSION")
        logger.info("=" * 50)
        logger.info(f"Collecting up to {max_products} products from clipboard")
        logger.info("Instructions:")
        logger.info("1. Browse to any grocery website")
        logger.info("2. Copy product information (name + price minimum)")
        logger.info("3. The system will automatically detect and extract product data")
        logger.info("4. Type 'done' when finished, or wait for timeout")
        logger.info("=" * 50)
        
        self.clear_collected_products()
        self.start_monitoring(interval=0.5)  # Faster monitoring
        
        try:
            # Wait for user to collect products
            timeout = 300  # 5 minutes
            start_time = asyncio.get_event_loop().time()
            
            while len(self.products_collected) < max_products:
                current_time = asyncio.get_event_loop().time()
                if current_time - start_time > timeout:
                    logger.info(f"⏰ Session timeout after {timeout}s")
                    break
                    
                # Check for user input to end session
                try:
                    user_input = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(None, input, "Type 'done' to finish: "),
                        timeout=1.0
                    )
                    
                    if user_input.lower().strip() == 'done':
                        logger.info("✅ Session ended by user")
                        break
                        
                except asyncio.TimeoutError:
                    pass  # Continue monitoring
                    
                await asyncio.sleep(1)
                
        finally:
            self.stop_monitoring()
            
        products = self.get_collected_products()
        
        logger.info(f"\n📊 SESSION COMPLETE")
        logger.info(f"Collected {len(products)} products:")
        
        for i, product in enumerate(products, 1):
            logger.info(f"   {i}. {product.name} - ${product.price} ({product.brand or 'No brand'})")
            
        return products


# Convenience functions
async def start_clipboard_collection(
    store_id: str = "manual_clipboard",
    max_products: int = 10
) -> List[Product]:
    """Start a clipboard collection session."""
    monitor = ClipboardMonitor(store_id)
    return await monitor.manual_clipboard_session(max_products)


def quick_parse_clipboard() -> Optional[Product]:
    """Quickly parse current clipboard content for product info."""
    try:
        content = pyperclip.paste()
        if not content.strip():
            return None
            
        monitor = ClipboardMonitor()
        clipboard_product = monitor._analyze_clipboard_content(content)
        
        if clipboard_product.suggested_product and clipboard_product.confidence > 0.5:
            logger.info(f"📋 Parsed clipboard product: {clipboard_product.suggested_product.name}")
            return clipboard_product.suggested_product
        else:
            logger.info(f"⚠️  Low confidence ({clipboard_product.confidence:.1%}) or incomplete data")
            logger.info(f"Found: {clipboard_product.extracted_fields}")
            return None
            
    except Exception as e:
        logger.error(f"Error parsing clipboard: {e}")
        return None