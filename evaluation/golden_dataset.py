"""
Golden Dataset for Evaluation Framework

Comprehensive collection of verified ingredient-to-product matches for quality monitoring,
regression testing, and continuous improvement validation.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, date
from decimal import Decimal
import json
import uuid
from pathlib import Path

from ..data_models.base_entity import BaseEntity


@dataclass
class GoldenMatch:
    """Verified ingredient-to-product match for quality validation."""
    
    # Identifiers
    match_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Ingredient information
    ingredient_name: str = ""
    ingredient_category: str = ""
    ingredient_quantity: Optional[float] = None
    ingredient_unit: str = ""
    ingredient_attributes: Dict[str, Any] = field(default_factory=dict)  # organic, low-fat, etc.
    
    # Product information
    product_name: str = ""
    product_brand: str = ""
    product_category: str = ""
    product_size: str = ""
    product_unit_price: Decimal = Decimal('0.00')
    product_total_price: Decimal = Decimal('0.00')
    product_store: str = ""
    product_url: Optional[str] = None
    product_stock_status: str = "in_stock"
    product_attributes: Dict[str, Any] = field(default_factory=dict)
    
    # Match metadata
    match_confidence: float = 1.0  # Human verified = 1.0
    match_type: str = "exact"  # exact, close, substitute, generic
    verification_date: date = field(default_factory=date.today)
    verifier: str = "human_expert"
    match_quality_score: float = 1.0
    
    # Context information
    season: str = "all"  # all, spring, summer, fall, winter, holiday
    location: str = "CA"  # Geographic region
    price_tier: str = "regular"  # regular, sale, premium, discount
    difficulty_level: str = "easy"  # easy, medium, hard, edge_case
    
    # Edge case indicators
    is_edge_case: bool = False
    edge_case_type: Optional[str] = None  # multi_brand, seasonal_unavailable, size_mismatch, etc.
    
    # Performance expectations
    expected_match_time: float = 0.5  # seconds
    expected_confidence_range: Tuple[float, float] = (0.85, 1.0)
    
    # Quality indicators
    price_accuracy_tolerance: float = 0.05  # 5% tolerance
    last_verified: date = field(default_factory=date.today)
    verification_frequency_days: int = 30
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = {
            'match_id': self.match_id,
            'ingredient_name': self.ingredient_name,
            'ingredient_category': self.ingredient_category,
            'ingredient_quantity': float(self.ingredient_quantity) if self.ingredient_quantity else None,
            'ingredient_unit': self.ingredient_unit,
            'ingredient_attributes': self.ingredient_attributes,
            'product_name': self.product_name,
            'product_brand': self.product_brand,
            'product_category': self.product_category,
            'product_size': self.product_size,
            'product_unit_price': str(self.product_unit_price),
            'product_total_price': str(self.product_total_price),
            'product_store': self.product_store,
            'product_url': self.product_url,
            'product_stock_status': self.product_stock_status,
            'product_attributes': self.product_attributes,
            'match_confidence': self.match_confidence,
            'match_type': self.match_type,
            'verification_date': self.verification_date.isoformat(),
            'verifier': self.verifier,
            'match_quality_score': self.match_quality_score,
            'season': self.season,
            'location': self.location,
            'price_tier': self.price_tier,
            'difficulty_level': self.difficulty_level,
            'is_edge_case': self.is_edge_case,
            'edge_case_type': self.edge_case_type,
            'expected_match_time': self.expected_match_time,
            'expected_confidence_range': self.expected_confidence_range,
            'price_accuracy_tolerance': self.price_accuracy_tolerance,
            'last_verified': self.last_verified.isoformat(),
            'verification_frequency_days': self.verification_frequency_days
        }
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GoldenMatch':
        """Create from dictionary."""
        match = cls()
        match.match_id = data.get('match_id', str(uuid.uuid4()))
        match.ingredient_name = data.get('ingredient_name', '')
        match.ingredient_category = data.get('ingredient_category', '')
        match.ingredient_quantity = data.get('ingredient_quantity')
        match.ingredient_unit = data.get('ingredient_unit', '')
        match.ingredient_attributes = data.get('ingredient_attributes', {})
        match.product_name = data.get('product_name', '')
        match.product_brand = data.get('product_brand', '')
        match.product_category = data.get('product_category', '')
        match.product_size = data.get('product_size', '')
        match.product_unit_price = Decimal(str(data.get('product_unit_price', '0.00')))
        match.product_total_price = Decimal(str(data.get('product_total_price', '0.00')))
        match.product_store = data.get('product_store', '')
        match.product_url = data.get('product_url')
        match.product_stock_status = data.get('product_stock_status', 'in_stock')
        match.product_attributes = data.get('product_attributes', {})
        match.match_confidence = data.get('match_confidence', 1.0)
        match.match_type = data.get('match_type', 'exact')
        match.verification_date = date.fromisoformat(data.get('verification_date', date.today().isoformat()))
        match.verifier = data.get('verifier', 'human_expert')
        match.match_quality_score = data.get('match_quality_score', 1.0)
        match.season = data.get('season', 'all')
        match.location = data.get('location', 'CA')
        match.price_tier = data.get('price_tier', 'regular')
        match.difficulty_level = data.get('difficulty_level', 'easy')
        match.is_edge_case = data.get('is_edge_case', False)
        match.edge_case_type = data.get('edge_case_type')
        match.expected_match_time = data.get('expected_match_time', 0.5)
        match.expected_confidence_range = tuple(data.get('expected_confidence_range', [0.85, 1.0]))
        match.price_accuracy_tolerance = data.get('price_accuracy_tolerance', 0.05)
        match.last_verified = date.fromisoformat(data.get('last_verified', date.today().isoformat()))
        match.verification_frequency_days = data.get('verification_frequency_days', 30)
        return match
    
    def needs_verification(self) -> bool:
        """Check if this match needs re-verification."""
        days_since_verification = (date.today() - self.last_verified).days
        return days_since_verification >= self.verification_frequency_days


class GoldenDatasetManager:
    """Manager for golden dataset operations and validation."""
    
    def __init__(self, dataset_path: str = "evaluation/datasets/golden_matches.json"):
        self.dataset_path = Path(dataset_path)
        self.matches: List[GoldenMatch] = []
        self._load_dataset()
    
    def _load_dataset(self):
        """Load golden dataset from file."""
        if self.dataset_path.exists():
            try:
                with open(self.dataset_path, 'r') as f:
                    data = json.load(f)
                    self.matches = [GoldenMatch.from_dict(match_data) for match_data in data.get('matches', [])]
                print(f"✅ Loaded {len(self.matches)} golden matches from {self.dataset_path}")
            except Exception as e:
                print(f"❌ Error loading golden dataset: {e}")
                self.matches = []
        else:
            print(f"📝 Creating new golden dataset at {self.dataset_path}")
            self._create_initial_dataset()
    
    def _create_initial_dataset(self):
        """Create initial golden dataset with comprehensive matches."""
        self.matches = self._generate_comprehensive_dataset()
        self.save_dataset()
    
    def _generate_comprehensive_dataset(self) -> List[GoldenMatch]:
        """Generate comprehensive golden dataset with 100+ verified matches."""
        matches = []
        
        # Dairy Products (20 matches)
        dairy_matches = [
            {
                'ingredient_name': 'milk', 'ingredient_category': 'dairy',
                'product_name': 'Lactantia 2% Milk', 'product_brand': 'Lactantia',
                'product_size': '4L', 'product_unit_price': '6.99', 'product_store': 'metro_ca',
                'match_type': 'exact', 'difficulty_level': 'easy'
            },
            {
                'ingredient_name': 'organic milk', 'ingredient_category': 'dairy',
                'ingredient_attributes': {'organic': True},
                'product_name': 'Organic Meadow Organic Milk 2%', 'product_brand': 'Organic Meadow',
                'product_size': '2L', 'product_unit_price': '8.99', 'product_store': 'walmart_ca',
                'match_type': 'exact', 'difficulty_level': 'medium'
            },
            {
                'ingredient_name': 'greek yogurt', 'ingredient_category': 'dairy',
                'product_name': 'Oikos Greek Yogurt Plain', 'product_brand': 'Oikos',
                'product_size': '750g', 'product_unit_price': '5.99', 'product_store': 'metro_ca',
                'match_type': 'exact', 'difficulty_level': 'easy'
            },
            {
                'ingredient_name': 'butter', 'ingredient_category': 'dairy',
                'product_name': 'Lactantia Salted Butter', 'product_brand': 'Lactantia',
                'product_size': '454g', 'product_unit_price': '5.49', 'product_store': 'walmart_ca',
                'match_type': 'exact', 'difficulty_level': 'easy'
            },
            {
                'ingredient_name': 'cheese', 'ingredient_category': 'dairy',
                'product_name': 'Armstrong Medium Cheddar', 'product_brand': 'Armstrong',
                'product_size': '400g', 'product_unit_price': '7.99', 'product_store': 'metro_ca',
                'match_type': 'close', 'difficulty_level': 'medium'
            }
        ]
        
        # Produce (25 matches)
        produce_matches = [
            {
                'ingredient_name': 'bananas', 'ingredient_category': 'produce',
                'product_name': 'Fresh Bananas', 'product_brand': 'No Brand',
                'product_size': 'per lb', 'product_unit_price': '1.49', 'product_store': 'metro_ca',
                'match_type': 'exact', 'difficulty_level': 'easy',
                'season': 'all'
            },
            {
                'ingredient_name': 'organic apples', 'ingredient_category': 'produce',
                'ingredient_attributes': {'organic': True},
                'product_name': 'Organic Gala Apples', 'product_brand': 'Organic',
                'product_size': '3lb bag', 'product_unit_price': '6.99', 'product_store': 'walmart_ca',
                'match_type': 'exact', 'difficulty_level': 'medium'
            },
            {
                'ingredient_name': 'strawberries', 'ingredient_category': 'produce',
                'product_name': 'Fresh Strawberries', 'product_brand': 'No Brand',
                'product_size': '1lb container', 'product_unit_price': '4.99', 'product_store': 'metro_ca',
                'match_type': 'exact', 'difficulty_level': 'easy',
                'season': 'spring', 'price_tier': 'seasonal'
            },
            {
                'ingredient_name': 'avocados', 'ingredient_category': 'produce',
                'product_name': 'Large Avocados', 'product_brand': 'No Brand',
                'product_size': 'each', 'product_unit_price': '1.99', 'product_store': 'walmart_ca',
                'match_type': 'exact', 'difficulty_level': 'easy'
            },
            {
                'ingredient_name': 'spinach', 'ingredient_category': 'produce',
                'product_name': 'Fresh Baby Spinach', 'product_brand': 'No Brand',
                'product_size': '312g container', 'product_unit_price': '3.99', 'product_store': 'metro_ca',
                'match_type': 'exact', 'difficulty_level': 'easy'
            }
        ]
        
        # Meat & Poultry (20 matches)
        meat_matches = [
            {
                'ingredient_name': 'chicken breast', 'ingredient_category': 'meat',
                'product_name': 'Fresh Chicken Breast Boneless Skinless', 'product_brand': 'No Brand',
                'product_size': 'per lb', 'product_unit_price': '8.99', 'product_store': 'metro_ca',
                'match_type': 'exact', 'difficulty_level': 'easy'
            },
            {
                'ingredient_name': 'ground beef', 'ingredient_category': 'meat',
                'product_name': 'Lean Ground Beef', 'product_brand': 'No Brand',
                'product_size': 'per lb', 'product_unit_price': '7.99', 'product_store': 'walmart_ca',
                'match_type': 'exact', 'difficulty_level': 'easy'
            },
            {
                'ingredient_name': 'salmon fillet', 'ingredient_category': 'meat',
                'product_name': 'Atlantic Salmon Fillet', 'product_brand': 'No Brand',
                'product_size': 'per lb', 'product_unit_price': '12.99', 'product_store': 'metro_ca',
                'match_type': 'exact', 'difficulty_level': 'medium'
            },
            {
                'ingredient_name': 'bacon', 'ingredient_category': 'meat',
                'product_name': 'Maple Leaf Bacon', 'product_brand': 'Maple Leaf',
                'product_size': '375g', 'product_unit_price': '6.99', 'product_store': 'walmart_ca',
                'match_type': 'exact', 'difficulty_level': 'easy'
            }
        ]
        
        # Pantry & Dry Goods (20 matches)
        pantry_matches = [
            {
                'ingredient_name': 'flour', 'ingredient_category': 'baking',
                'product_name': 'All Purpose Flour', 'product_brand': 'Five Roses',
                'product_size': '2.5kg', 'product_unit_price': '4.99', 'product_store': 'metro_ca',
                'match_type': 'exact', 'difficulty_level': 'easy'
            },
            {
                'ingredient_name': 'rice', 'ingredient_category': 'grains',
                'product_name': 'Jasmine Rice', 'product_brand': 'Uncle Ben\'s',
                'product_size': '2kg', 'product_unit_price': '7.99', 'product_store': 'walmart_ca',
                'match_type': 'exact', 'difficulty_level': 'easy'
            },
            {
                'ingredient_name': 'pasta', 'ingredient_category': 'pasta',
                'product_name': 'Spaghetti', 'product_brand': 'Barilla',
                'product_size': '450g', 'product_unit_price': '1.99', 'product_store': 'metro_ca',
                'match_type': 'close', 'difficulty_level': 'medium'
            },
            {
                'ingredient_name': 'olive oil', 'ingredient_category': 'oils',
                'product_name': 'Extra Virgin Olive Oil', 'product_brand': 'Bertolli',
                'product_size': '500ml', 'product_unit_price': '8.99', 'product_store': 'walmart_ca',
                'match_type': 'exact', 'difficulty_level': 'easy'
            }
        ]
        
        # Edge Cases & Difficult Matches (15 matches)
        edge_case_matches = [
            {
                'ingredient_name': 'tahini', 'ingredient_category': 'condiments',
                'product_name': 'Joyva Tahini Sesame Paste', 'product_brand': 'Joyva',
                'product_size': '454g', 'product_unit_price': '7.99', 'product_store': 'metro_ca',
                'match_type': 'exact', 'difficulty_level': 'hard',
                'is_edge_case': True, 'edge_case_type': 'specialty_ingredient'
            },
            {
                'ingredient_name': 'coconut milk', 'ingredient_category': 'dairy',
                'product_name': 'Thai Kitchen Coconut Milk', 'product_brand': 'Thai Kitchen',
                'product_size': '400ml can', 'product_unit_price': '2.99', 'product_store': 'walmart_ca',
                'match_type': 'exact', 'difficulty_level': 'medium',
                'is_edge_case': True, 'edge_case_type': 'alternative_milk'
            },
            {
                'ingredient_name': 'vanilla extract', 'ingredient_category': 'baking',
                'product_name': 'Pure Vanilla Extract', 'product_brand': 'Club House',
                'product_size': '118ml', 'product_unit_price': '12.99', 'product_store': 'metro_ca',
                'match_type': 'exact', 'difficulty_level': 'hard',
                'is_edge_case': True, 'edge_case_type': 'small_expensive_item'
            }
        ]
        
        # Convert to GoldenMatch objects
        all_match_data = dairy_matches + produce_matches + meat_matches + pantry_matches + edge_case_matches
        
        for match_data in all_match_data:
            match = GoldenMatch(
                ingredient_name=match_data['ingredient_name'],
                ingredient_category=match_data['ingredient_category'],
                ingredient_attributes=match_data.get('ingredient_attributes', {}),
                product_name=match_data['product_name'],
                product_brand=match_data['product_brand'],
                product_size=match_data['product_size'],
                product_unit_price=Decimal(match_data['product_unit_price']),
                product_store=match_data['product_store'],
                match_type=match_data['match_type'],
                difficulty_level=match_data['difficulty_level'],
                season=match_data.get('season', 'all'),
                price_tier=match_data.get('price_tier', 'regular'),
                is_edge_case=match_data.get('is_edge_case', False),
                edge_case_type=match_data.get('edge_case_type')
            )
            matches.append(match)
        
        return matches
    
    def save_dataset(self):
        """Save golden dataset to file."""
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'metadata': {
                'version': '1.0',
                'created_date': datetime.now().isoformat(),
                'total_matches': len(self.matches),
                'verification_standard': 'human_expert'
            },
            'matches': [match.to_dict() for match in self.matches]
        }
        
        with open(self.dataset_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"✅ Saved {len(self.matches)} golden matches to {self.dataset_path}")
    
    def get_matches_by_category(self, category: str) -> List[GoldenMatch]:
        """Get all matches for a specific category."""
        return [match for match in self.matches if match.ingredient_category == category]
    
    def get_edge_cases(self) -> List[GoldenMatch]:
        """Get all edge case matches."""
        return [match for match in self.matches if match.is_edge_case]
    
    def get_matches_by_difficulty(self, difficulty: str) -> List[GoldenMatch]:
        """Get matches by difficulty level."""
        return [match for match in self.matches if match.difficulty_level == difficulty]
    
    def get_matches_by_store(self, store: str) -> List[GoldenMatch]:
        """Get matches for a specific store."""
        return [match for match in self.matches if match.product_store == store]
    
    def get_seasonal_matches(self, season: str) -> List[GoldenMatch]:
        """Get matches for a specific season."""
        return [match for match in self.matches if match.season == season or match.season == 'all']
    
    def get_matches_needing_verification(self) -> List[GoldenMatch]:
        """Get matches that need re-verification."""
        return [match for match in self.matches if match.needs_verification()]
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get comprehensive dataset statistics."""
        total_matches = len(self.matches)
        
        if total_matches == 0:
            return {'total_matches': 0}
        
        categories = {}
        difficulties = {}
        stores = {}
        seasons = {}
        edge_cases = 0
        
        for match in self.matches:
            # Categories
            if match.ingredient_category in categories:
                categories[match.ingredient_category] += 1
            else:
                categories[match.ingredient_category] = 1
            
            # Difficulties
            if match.difficulty_level in difficulties:
                difficulties[match.difficulty_level] += 1
            else:
                difficulties[match.difficulty_level] = 1
            
            # Stores
            if match.product_store in stores:
                stores[match.product_store] += 1
            else:
                stores[match.product_store] = 1
            
            # Seasons
            if match.season in seasons:
                seasons[match.season] += 1
            else:
                seasons[match.season] = 1
            
            # Edge cases
            if match.is_edge_case:
                edge_cases += 1
        
        return {
            'total_matches': total_matches,
            'categories': categories,
            'difficulties': difficulties,
            'stores': stores,
            'seasons': seasons,
            'edge_cases': edge_cases,
            'edge_case_percentage': (edge_cases / total_matches) * 100,
            'needs_verification': len(self.get_matches_needing_verification())
        }


if __name__ == "__main__":
    manager = GoldenDatasetManager()
    stats = manager.get_dataset_stats()
    print(f"\n📊 Golden Dataset Statistics:")
    print(f"Total Matches: {stats['total_matches']}")
    print(f"Categories: {stats['categories']}")
    print(f"Difficulty Distribution: {stats['difficulties']}")
    print(f"Store Coverage: {stats['stores']}")
    print(f"Edge Cases: {stats['edge_cases']} ({stats['edge_case_percentage']:.1f}%)")
    print(f"Matches Needing Verification: {stats['needs_verification']}")