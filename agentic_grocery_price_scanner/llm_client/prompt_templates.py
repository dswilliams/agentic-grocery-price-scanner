"""
Prompt templates for grocery-specific LLM tasks.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """Template for LLM prompts with variables."""
    template: str
    variables: List[str]
    description: str
    suggested_model: str
    expected_output: str


class PromptTemplates:
    """Collection of prompt templates for grocery price scanner tasks."""
    
    # Ingredient normalization templates
    NORMALIZE_INGREDIENT = PromptTemplate(
        template="""
Normalize the following ingredient to a standard grocery product name:

Input ingredient: "{ingredient}"

Rules:
- Convert to lowercase
- Remove unnecessary words (fresh, organic, etc.) unless they're essential
- Standardize units and quantities
- Use common grocery store naming conventions
- Keep brand names only if specifically mentioned

Return only the normalized ingredient name, nothing else.
""".strip(),
        variables=["ingredient"],
        description="Normalize ingredient names for consistent matching",
        suggested_model="qwen2.5:1.5b",
        expected_output="Normalized ingredient string"
    )
    
    MATCH_PRODUCTS = PromptTemplate(
        template="""
Given a shopping ingredient and a list of grocery products, find the best matches:

Ingredient: "{ingredient}"

Available products:
{product_list}

Return the top 3 best matches as a JSON array with this format:
{{
  "matches": [
    {{
      "product_name": "exact product name",
      "confidence": 0.95,
      "reason": "explanation for match"
    }}
  ]
}}

Consider:
- Similar ingredients (flour types, milk varieties)
- Brand alternatives
- Size/quantity differences
- Organic vs conventional options
""".strip(),
        variables=["ingredient", "product_list"],
        description="Match ingredients to available grocery products",
        suggested_model="qwen2.5:1.5b", 
        expected_output="JSON with product matches and confidence scores"
    )
    
    # Brand normalization templates
    NORMALIZE_BRAND = PromptTemplate(
        template="""
Extract and normalize the brand name from this product text:

Product: "{product_text}"

Rules:
- Extract the main brand name only
- Remove product type descriptors
- Standardize common brand name variations
- Use proper capitalization
- Return "Generic" if no specific brand is mentioned

Return only the brand name, nothing else.
""".strip(),
        variables=["product_text"],
        description="Extract and standardize brand names",
        suggested_model="qwen2.5:1.5b",
        expected_output="Standardized brand name"
    )
    
    # Decision-making templates
    SCRAPING_STRATEGY = PromptTemplate(
        template="""
Analyze the scraping situation and recommend the best strategy:

Current context:
- Target store: {store_name}
- Product query: "{query}"
- Previous attempts: {previous_attempts}
- Success rates: Layer 1 (stealth): {layer1_success}%, Layer 2 (browser): {layer2_success}%, Layer 3 (manual): {layer3_success}%
- Time constraints: {time_limit} minutes
- User preference: {user_preference}

Available strategies:
1. Stealth scraping (fast, automated, may be blocked)
2. Human-assisted browser (reliable, requires user interaction)
3. Manual clipboard collection (always works, slowest)

Recommend the best strategy and provide reasoning as JSON:
{{
  "recommended_strategy": "layer_number",
  "confidence": 0.85,
  "reasoning": "detailed explanation",
  "estimated_time": "time in minutes",
  "success_probability": 0.90,
  "fallback_plan": "alternative if recommended fails"
}}
""".strip(),
        variables=["store_name", "query", "previous_attempts", "layer1_success", 
                  "layer2_success", "layer3_success", "time_limit", "user_preference"],
        description="Intelligent scraping strategy selection",
        suggested_model="phi3.5:latest",
        expected_output="JSON with strategy recommendation and analysis"
    )
    
    OPTIMIZATION_ADVICE = PromptTemplate(
        template="""
Analyze grocery shopping data and provide optimization recommendations:

Shopping list analysis:
- Total items: {total_items}
- Estimated total cost: ${estimated_cost}
- Stores available: {store_list}
- User preferences: {preferences}
- Time available: {time_available}

Product data:
{product_data}

Price comparison summary:
{price_comparison}

Provide optimization advice as JSON:
{{
  "cost_savings": {{
    "potential_savings": "$X.XX",
    "percentage": "XX%",
    "best_store_combination": ["store1", "store2"]
  }},
  "time_optimization": {{
    "single_store_option": "store_name",
    "multi_store_route": ["store1", "store2"],
    "estimated_time": "XX minutes"
  }},
  "substitution_suggestions": [
    {{
      "original": "product_name",
      "substitute": "alternative_product",
      "savings": "$X.XX",
      "reason": "explanation"
    }}
  ],
  "priority_recommendations": [
    "recommendation 1",
    "recommendation 2"
  ]
}}
""".strip(),
        variables=["total_items", "estimated_cost", "store_list", "preferences",
                  "time_available", "product_data", "price_comparison"],
        description="Provide grocery shopping optimization advice",
        suggested_model="phi3.5:latest",
        expected_output="JSON with comprehensive optimization recommendations"
    )
    
    # Classification templates
    CLASSIFY_PRODUCT_CATEGORY = PromptTemplate(
        template="""
Classify the following product into the appropriate grocery category:

Product: "{product_name}"

Standard categories:
- Produce (fruits, vegetables)
- Dairy & Eggs
- Meat & Seafood
- Bakery
- Pantry (canned goods, dry goods)
- Frozen Foods
- Beverages
- Snacks & Candy
- Health & Beauty
- Household Items
- Other

Return only the category name, nothing else.
""".strip(),
        variables=["product_name"],
        description="Classify products into grocery categories",
        suggested_model="qwen2.5:1.5b",
        expected_output="Single category name"
    )
    
    EXTRACT_PRODUCT_INFO = PromptTemplate(
        template="""
Extract structured product information from this text:

Product text: "{product_text}"

Extract and return as JSON:
{{
  "name": "product name",
  "brand": "brand name or Generic",
  "size": "size/quantity",
  "unit": "unit type",
  "price": "price as number",
  "currency": "currency code",
  "category": "product category",
  "organic": true/false,
  "on_sale": true/false,
  "original_price": "original price if on sale"
}}

If information is not available, use null for that field.
""".strip(),
        variables=["product_text"],
        description="Extract structured data from product text",
        suggested_model="qwen2.5:1.5b",
        expected_output="JSON with structured product information"
    )
    
    # Vector search optimization
    GENERATE_SEARCH_VARIATIONS = PromptTemplate(
        template="""
Generate search query variations for better product matching:

Original query: "{original_query}"

Generate 5-7 alternative search terms that could match the same product:
- Include synonyms
- Different brand names
- Size variations
- Generic vs specific terms
- Common misspellings
- Alternative product forms

Return as JSON array:
{{
  "variations": [
    "variation 1",
    "variation 2",
    "..."
  ]
}}
""".strip(),
        variables=["original_query"],
        description="Generate query variations for better search results",
        suggested_model="qwen2.5:1.5b",
        expected_output="JSON array of search query variations"
    )
    
    # Error analysis and improvement
    ANALYZE_SCRAPING_FAILURE = PromptTemplate(
        template="""
Analyze this scraping failure and suggest improvements:

Failure details:
- Store: {store_name}
- Method: {scraping_method}
- Error: {error_message}
- HTML snippet: {html_snippet}
- Selectors used: {selectors}
- Success rate: {success_rate}%

Provide analysis and recommendations as JSON:
{{
  "failure_category": "bot_detection|selector_changed|rate_limiting|network_error|other",
  "root_cause": "detailed explanation",
  "severity": "high|medium|low",
  "recommended_fixes": [
    "specific action 1",
    "specific action 2"
  ],
  "alternative_approaches": [
    "approach 1",
    "approach 2"
  ],
  "confidence": 0.85
}}
""".strip(),
        variables=["store_name", "scraping_method", "error_message", "html_snippet",
                  "selectors", "success_rate"],
        description="Analyze scraping failures and suggest improvements",
        suggested_model="phi3.5:latest",
        expected_output="JSON with failure analysis and recommendations"
    )
    
    @classmethod
    def get_template(cls, template_name: str) -> Optional[PromptTemplate]:
        """Get a specific prompt template by name."""
        return getattr(cls, template_name, None)
    
    @classmethod
    def list_templates(cls) -> Dict[str, PromptTemplate]:
        """List all available prompt templates."""
        templates = {}
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            if isinstance(attr, PromptTemplate):
                templates[attr_name] = attr
        return templates
    
    @classmethod
    def format_template(
        cls, 
        template_name: str, 
        **kwargs
    ) -> Optional[str]:
        """Format a template with provided variables."""
        template = cls.get_template(template_name)
        if not template:
            return None
        
        try:
            return template.template.format(**kwargs)
        except KeyError as e:
            missing_var = str(e).strip("'")
            raise ValueError(f"Missing required variable '{missing_var}' for template '{template_name}'")
    
    @classmethod
    def get_model_recommendation(cls, template_name: str) -> Optional[str]:
        """Get the recommended model for a specific template."""
        template = cls.get_template(template_name)
        return template.suggested_model if template else None
    
    @classmethod
    def validate_variables(cls, template_name: str, **kwargs) -> bool:
        """Validate that all required variables are provided."""
        template = cls.get_template(template_name)
        if not template:
            return False
        
        provided_vars = set(kwargs.keys())
        required_vars = set(template.variables)
        
        return required_vars.issubset(provided_vars)