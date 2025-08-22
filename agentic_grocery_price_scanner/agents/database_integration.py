"""
Database integration for the intelligent scraper agent.
Handles saving products to both SQL and vector databases with confidence weighting.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..data_models.product import Product
from ..data_models.base import DataCollectionMethod
from ..utils.database import DatabaseManager
from ..vector_db.qdrant_client import QdrantVectorClient
from ..vector_db.scraper_integration import ScraperIntegrationService

logger = logging.getLogger(__name__)


class ScrapingDatabaseIntegrator:
    """Integrates scraping results with database systems."""
    
    def __init__(self, enable_vector_db: bool = True, enable_sql_db: bool = True):
        """Initialize database integrator."""
        self.enable_vector_db = enable_vector_db
        self.enable_sql_db = enable_sql_db
        
        # Lazy initialization
        self._db_manager = None
        self._vector_client = None
        self._integration_service = None
    
    @property
    def db_manager(self) -> DatabaseManager:
        """Get database manager (lazy initialization)."""
        if self._db_manager is None:
            self._db_manager = DatabaseManager()
        return self._db_manager
    
    @property
    def vector_client(self) -> QdrantVectorClient:
        """Get vector client (lazy initialization)."""
        if self._vector_client is None:
            self._vector_client = QdrantVectorClient()
        return self._vector_client
    
    @property
    def integration_service(self) -> ScraperIntegrationService:
        """Get integration service (lazy initialization)."""
        if self._integration_service is None:
            self._integration_service = ScraperIntegrationService()
        return self._integration_service
    
    async def save_scraping_results(
        self,
        scraping_results: Dict[str, Any],
        session_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Save complete scraping results to databases.
        
        Args:
            scraping_results: Results from intelligent scraper agent
            session_metadata: Additional session metadata
        
        Returns:
            Dictionary with save results and statistics
        """
        logger.info("Starting database integration for scraping results")
        
        products = scraping_results.get("products", [])
        if not products:
            logger.warning("No products to save")
            return {"success": True, "products_saved": 0, "warnings": ["No products to save"]}
        
        save_results = {
            "success": True,
            "products_saved": 0,
            "sql_saved": 0,
            "vector_saved": 0,
            "errors": [],
            "warnings": [],
            "metadata": {
                "save_timestamp": datetime.now().isoformat(),
                "session_metadata": session_metadata
            }
        }
        
        try:
            # Save to SQL database
            if self.enable_sql_db:
                sql_results = await self._save_to_sql_database(products, scraping_results)
                save_results["sql_saved"] = sql_results["saved_count"]
                save_results["errors"].extend(sql_results.get("errors", []))
                save_results["warnings"].extend(sql_results.get("warnings", []))
            
            # Save to vector database
            if self.enable_vector_db:
                vector_results = await self._save_to_vector_database(products, scraping_results)
                save_results["vector_saved"] = vector_results["saved_count"]
                save_results["errors"].extend(vector_results.get("errors", []))
                save_results["warnings"].extend(vector_results.get("warnings", []))
            
            # Calculate total saved (use max to avoid double counting)
            save_results["products_saved"] = max(
                save_results["sql_saved"],
                save_results["vector_saved"]
            )
            
            # Add collection method analysis
            save_results["collection_analysis"] = self._analyze_collection_methods(products)
            
            logger.info(f"Database integration completed: {save_results['products_saved']} products saved")
            
        except Exception as e:
            logger.error(f"Database integration failed: {e}", exc_info=True)
            save_results["success"] = False
            save_results["errors"].append(f"Integration failed: {str(e)}")
        
        return save_results
    
    async def _save_to_sql_database(
        self,
        products: List[Product],
        scraping_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Save products to SQL database."""
        logger.info(f"Saving {len(products)} products to SQL database")
        
        results = {
            "saved_count": 0,
            "errors": [],
            "warnings": []
        }
        
        try:
            # Ensure database is initialized
            self.db_manager.initialize()
            
            saved_count = 0
            for product in products:
                try:
                    # Add scraping session metadata
                    if hasattr(product, 'source_metadata'):
                        if product.source_metadata is None:
                            product.source_metadata = {}
                        product.source_metadata.update({
                            "scraping_session": {
                                "query": scraping_results.get("query"),
                                "stores_scraped": scraping_results.get("stores_completed", []),
                                "success_rate": scraping_results.get("success_rates", {}),
                                "save_timestamp": datetime.now().isoformat()
                            }
                        })
                    
                    # Save product
                    self.db_manager.save_product(product)
                    saved_count += 1
                    
                except Exception as e:
                    error_msg = f"Failed to save product '{product.name}': {str(e)}"
                    logger.error(error_msg)
                    results["errors"].append(error_msg)
            
            results["saved_count"] = saved_count
            logger.info(f"SQL save completed: {saved_count}/{len(products)} products saved")
            
        except Exception as e:
            error_msg = f"SQL database save failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            results["errors"].append(error_msg)
        
        return results
    
    async def _save_to_vector_database(
        self,
        products: List[Product],
        scraping_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Save products to vector database with embeddings."""
        logger.info(f"Saving {len(products)} products to vector database")
        
        results = {
            "saved_count": 0,
            "errors": [],
            "warnings": []
        }
        
        try:
            # Ensure vector database is initialized
            await self.vector_client.initialize()
            
            # Use integration service for intelligent saving
            integration_results = await self.integration_service.process_scraped_products(
                products,
                source_metadata={
                    "scraping_session": scraping_results.get("collection_metadata", {}),
                    "query": scraping_results.get("query"),
                    "stores": scraping_results.get("stores_completed", [])
                }
            )
            
            results["saved_count"] = integration_results.get("processed_count", 0)
            results["errors"].extend(integration_results.get("errors", []))
            results["warnings"].extend(integration_results.get("warnings", []))
            
            # Add vector-specific metrics
            results["embedding_stats"] = integration_results.get("embedding_stats", {})
            results["similarity_matches"] = integration_results.get("similarity_matches", [])
            
            logger.info(f"Vector save completed: {results['saved_count']}/{len(products)} products saved")
            
        except Exception as e:
            error_msg = f"Vector database save failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            results["errors"].append(error_msg)
        
        return results
    
    def _analyze_collection_methods(self, products: List[Product]) -> Dict[str, Any]:
        """Analyze collection methods used in the products."""
        method_stats = {}
        confidence_stats = {}
        
        for product in products:
            method = product.collection_method.value
            
            # Count by method
            method_stats[method] = method_stats.get(method, 0) + 1
            
            # Track confidence by method
            if method not in confidence_stats:
                confidence_stats[method] = []
            confidence_stats[method].append(product.get_collection_confidence_weight())
        
        # Calculate averages
        method_analysis = {}
        for method, count in method_stats.items():
            confidences = confidence_stats[method]
            method_analysis[method] = {
                "count": count,
                "percentage": (count / len(products)) * 100,
                "avg_confidence": sum(confidences) / len(confidences),
                "min_confidence": min(confidences),
                "max_confidence": max(confidences)
            }
        
        return {
            "total_products": len(products),
            "method_breakdown": method_analysis,
            "overall_avg_confidence": sum(
                p.get_collection_confidence_weight() for p in products
            ) / len(products)
        }
    
    async def search_similar_products(
        self,
        query: str,
        collection_method: Optional[DataCollectionMethod] = None,
        store_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for similar products using vector similarity.
        
        Args:
            query: Search query
            collection_method: Filter by collection method
            store_id: Filter by store
            limit: Maximum results
        
        Returns:
            List of similar products with similarity scores
        """
        if not self.enable_vector_db:
            logger.warning("Vector database disabled, cannot search similar products")
            return []
        
        try:
            await self.vector_client.initialize()
            
            # Build filters
            filters = {}
            if collection_method:
                filters["collection_method"] = collection_method.value
            if store_id:
                filters["store_id"] = store_id
            
            # Search with filters
            search_results = await self.vector_client.search_products(
                query_text=query,
                limit=limit,
                filters=filters,
                confidence_threshold=0.7
            )
            
            # Format results
            formatted_results = []
            for result in search_results:
                formatted_results.append({
                    "product": result["product"],
                    "similarity_score": result["score"],
                    "collection_confidence": result["product"].get_collection_confidence_weight(),
                    "combined_score": result["score"] * result["product"].get_collection_confidence_weight()
                })
            
            # Sort by combined score (similarity × collection confidence)
            formatted_results.sort(key=lambda x: x["combined_score"], reverse=True)
            
            logger.info(f"Found {len(formatted_results)} similar products for query: '{query}'")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Similar product search failed: {e}", exc_info=True)
            return []
    
    async def get_collection_statistics(self) -> Dict[str, Any]:
        """Get statistics about collected products across all methods."""
        stats = {
            "sql_stats": {},
            "vector_stats": {},
            "combined_stats": {}
        }
        
        try:
            # Get SQL statistics
            if self.enable_sql_db:
                stats["sql_stats"] = await self._get_sql_statistics()
            
            # Get vector statistics
            if self.enable_vector_db:
                stats["vector_stats"] = await self._get_vector_statistics()
            
            # Combine statistics
            stats["combined_stats"] = self._combine_statistics(
                stats["sql_stats"],
                stats["vector_stats"]
            )
            
        except Exception as e:
            logger.error(f"Failed to get collection statistics: {e}", exc_info=True)
            stats["error"] = str(e)
        
        return stats
    
    async def _get_sql_statistics(self) -> Dict[str, Any]:
        """Get statistics from SQL database."""
        try:
            # This would query the SQL database for statistics
            # Implementation depends on DatabaseManager methods
            return {
                "total_products": 0,  # Placeholder
                "by_store": {},
                "by_collection_method": {},
                "latest_update": None
            }
        except Exception as e:
            logger.error(f"SQL statistics failed: {e}")
            return {"error": str(e)}
    
    async def _get_vector_statistics(self) -> Dict[str, Any]:
        """Get statistics from vector database."""
        try:
            await self.vector_client.initialize()
            return await self.vector_client.get_collection_stats()
        except Exception as e:
            logger.error(f"Vector statistics failed: {e}")
            return {"error": str(e)}
    
    def _combine_statistics(
        self,
        sql_stats: Dict[str, Any],
        vector_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Combine statistics from both databases."""
        combined = {
            "total_products_sql": sql_stats.get("total_products", 0),
            "total_products_vector": vector_stats.get("total_products", 0),
            "synchronization_rate": 0.0,
            "collection_method_distribution": {},
            "store_distribution": {},
            "data_quality_score": 0.0
        }
        
        # Calculate synchronization rate
        sql_count = sql_stats.get("total_products", 0)
        vector_count = vector_stats.get("total_products", 0)
        
        if max(sql_count, vector_count) > 0:
            combined["synchronization_rate"] = min(sql_count, vector_count) / max(sql_count, vector_count)
        
        # Combine collection method distributions
        sql_methods = sql_stats.get("by_collection_method", {})
        vector_methods = vector_stats.get("by_collection_method", {})
        
        all_methods = set(sql_methods.keys()) | set(vector_methods.keys())
        for method in all_methods:
            combined["collection_method_distribution"][method] = {
                "sql_count": sql_methods.get(method, 0),
                "vector_count": vector_methods.get(method, 0)
            }
        
        return combined


class ScrapingSessionTracker:
    """Track scraping sessions and their database integration results."""
    
    def __init__(self):
        """Initialize session tracker."""
        self.sessions = {}
        self.current_session_id = None
    
    def start_session(self, session_id: str, metadata: Dict[str, Any]) -> None:
        """Start tracking a new scraping session."""
        self.current_session_id = session_id
        self.sessions[session_id] = {
            "start_time": datetime.now().isoformat(),
            "metadata": metadata,
            "scraping_results": None,
            "database_results": None,
            "status": "active"
        }
    
    def update_scraping_results(self, session_id: str, results: Dict[str, Any]) -> None:
        """Update scraping results for a session."""
        if session_id in self.sessions:
            self.sessions[session_id]["scraping_results"] = results
            self.sessions[session_id]["scraping_end_time"] = datetime.now().isoformat()
    
    def update_database_results(self, session_id: str, results: Dict[str, Any]) -> None:
        """Update database integration results for a session."""
        if session_id in self.sessions:
            self.sessions[session_id]["database_results"] = results
            self.sessions[session_id]["database_end_time"] = datetime.now().isoformat()
            self.sessions[session_id]["status"] = "completed"
    
    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of a scraping session."""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        scraping_results = session.get("scraping_results", {})
        database_results = session.get("database_results", {})
        
        return {
            "session_id": session_id,
            "status": session["status"],
            "start_time": session["start_time"],
            "metadata": session["metadata"],
            "products_scraped": len(scraping_results.get("products", [])),
            "products_saved": database_results.get("products_saved", 0),
            "stores_completed": len(scraping_results.get("stores_completed", [])),
            "stores_failed": len(scraping_results.get("stores_failed", [])),
            "collection_methods_used": list(
                database_results.get("collection_analysis", {})
                .get("method_breakdown", {}).keys()
            ),
            "success_rate": scraping_results.get("success_rates", {}),
            "duration_seconds": self._calculate_duration(session)
        }
    
    def _calculate_duration(self, session: Dict[str, Any]) -> Optional[float]:
        """Calculate session duration in seconds."""
        start_time_str = session.get("start_time")
        end_time_str = session.get("database_end_time") or session.get("scraping_end_time")
        
        if not start_time_str or not end_time_str:
            return None
        
        try:
            start_time = datetime.fromisoformat(start_time_str)
            end_time = datetime.fromisoformat(end_time_str)
            return (end_time - start_time).total_seconds()
        except Exception:
            return None
    
    def get_all_sessions_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all tracked sessions."""
        summaries = []
        for session_id in self.sessions:
            summary = self.get_session_summary(session_id)
            if summary:
                summaries.append(summary)
        
        # Sort by start time (most recent first)
        summaries.sort(
            key=lambda x: x.get("start_time", ""),
            reverse=True
        )
        
        return summaries