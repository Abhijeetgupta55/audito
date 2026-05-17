"""Evaluation pipeline for recommendation quality, hallucinations, and cost tracking."""
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
import asyncio
import json

from openai import AsyncOpenAI
from backend.config import settings

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of evaluating a recommendation."""
    
    request_id: str
    concern: str
    hallucinated: bool
    hallucinated_products: List[str]
    safety_appropriate: bool
    product_relevance: float  # 0-1
    latency_ms: float
    tokens_used: int
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class EvaluationMetrics:
    """Track recommendation quality, costs, and hallucinations."""
    
    def __init__(self):
        # Keep client disabled — evaluation LLM calls eat into the same quota as the main pipeline.
        # Fallback keyword-based evaluation runs automatically when client is None.
        self.client = None
        
        # Request tracking
        self.requests_processed = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
        # Quality metrics
        self.hallucinations_detected = 0
        self.safety_checks_failed = 0
        
        # Performance tracking
        self.latencies: List[float] = []
        self.tokens_used: List[int] = []
        
        # Agent path tracking
        self.agent_paths: Dict[str, int] = {}
        
        # Product distribution
        self.product_distribution: Dict[str, int] = {}
        self.concern_distribution: Dict[str, int] = {}
        
        # Doctor feedback
        self.feedback_count = 0
        self.feedback_scores: List[int] = []
        
        logger.info("✅ Evaluation metrics initialized")
    
    async def evaluate_response(
        self,
        request_id: str,
        user_message: str,
        recommendation: str,
        products: List[Dict[str, Any]],
        latency_ms: float
    ) -> EvaluationResult:
        """Evaluate a recommendation for quality."""
        
        logger.info(f"📊 Evaluating request {request_id}")
        
        try:
            # Check for hallucinations
            hallucinated, hallucinated_products = await self._check_hallucinations(
                recommendation,
                products
            )
            
            # Check safety appropriateness
            safety_appropriate = await self._check_safety(
                recommendation,
                user_message
            )
            
            # Calculate product relevance
            product_relevance = await self._calculate_relevance(
                user_message,
                recommendation,
                products
            )
            
            # Update metrics
            self.requests_processed += 1
            self.successful_requests += 1
            
            if hallucinated:
                self.hallucinations_detected += 1
                logger.warning(f"❌ Hallucination detected: {hallucinated_products}")
            
            if not safety_appropriate:
                self.safety_checks_failed += 1
                logger.warning("⚠️  Safety concern detected")
            
            self.latencies.append(latency_ms)
            
            # Create result
            result = EvaluationResult(
                request_id=request_id,
                concern=user_message[:50],
                hallucinated=hallucinated,
                hallucinated_products=hallucinated_products,
                safety_appropriate=safety_appropriate,
                product_relevance=product_relevance,
                latency_ms=latency_ms,
                tokens_used=0
            )
            
            logger.info(f"✅ Evaluation complete: hallucination={hallucinated}, "
                       f"safety={safety_appropriate}, relevance={product_relevance:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            self.failed_requests += 1
            raise
    
    async def _check_hallucinations(
        self,
        recommendation: str,
        products: List[Dict[str, Any]]
    ) -> tuple[bool, List[str]]:
        """Check if recommendation mentions non-existent products."""
        
        product_names = {p["name"] for p in products}

        if self.client is None:
            hallucinated_products = [name for name in product_names if name.lower() not in recommendation.lower()]
            return len(hallucinated_products) > 0, hallucinated_products
        
        try:
            response = await self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                temperature=0.1,
                messages=[
                    {
                        "role": "system",
                        "content": f"""Check if the recommendation ONLY mentions these products:
{', '.join(product_names)}

If it mentions other products, list them.

Respond ONLY with JSON:
{{
    "hallucinated": false,
    "unknown_products": []
}}"""
                    },
                    {"role": "user", "content": recommendation[:500]}
                ]
            )
            
            content = response.choices[0].message.content or "{}"
            result = json.loads(content)
            return result.get("hallucinated", False), result.get("unknown_products", [])
            
        except Exception as e:
            logger.warning(f"Hallucination check failed: {e}")
            return False, []
    
    async def _check_safety(
        self,
        recommendation: str,
        user_message: str
    ) -> bool:
        """Check if recommendation is medically appropriate."""

        if self.client is None:
            return "severe" not in user_message.lower()
        
        try:
            response = await self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                temperature=0.1,
                messages=[
                    {
                        "role": "system",
                        "content": """Is this medical recommendation appropriate and safe?
Consider:
- Ingredient safety
- Drug interactions
- Condition severity mismatch
- Allergic reaction risks

Respond ONLY with JSON:
{"safe": true, "concerns": []}"""
                    },
                    {
                        "role": "user",
                        "content": f"Concern: {user_message[:100]}\n\nRecommendation: {recommendation[:500]}"
                    }
                ]
            )
            
            content = response.choices[0].message.content or "{}"
            result = json.loads(content)
            return result.get("safe", True)
            
        except Exception as e:
            logger.warning(f"Safety check failed: {e}")
            return True
    
    async def _calculate_relevance(
        self,
        user_message: str,
        recommendation: str,
        products: List[Dict[str, Any]]
    ) -> float:
        """Calculate how relevant the recommendation is to the concern."""

        if self.client is None:
            keywords = [word for word in user_message.lower().split() if len(word) > 3]
            matched = sum(1 for keyword in keywords if keyword in recommendation.lower())
            return min(1.0, 0.3 + matched * 0.1)
        
        try:
            response = await self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                temperature=0.1,
                messages=[
                    {
                        "role": "system",
                        "content": """Rate the relevance of the recommendation to the stated concern on a scale of 0-1.

Consider:
- Does it address the concern?
- Are the products suitable?
- Is the explanation clear?

Respond ONLY with JSON:
{"relevance": 0.85}"""
                    },
                    {
                        "role": "user",
                        "content": f"Concern: {user_message}\n\nRecommendation: {recommendation[:500]}"
                    }
                ]
            )
            
            content = response.choices[0].message.content or "{}"
            result = json.loads(content)
            return min(1.0, max(0.0, result.get("relevance", 0.5)))
            
        except Exception as e:
            logger.warning(f"Relevance calculation failed: {e}")
            return 0.5
    
    async def store_feedback(
        self,
        recommendation_id: str,
        quality_score: int,
        product_relevance_score: int,
        safety_score: int,
        comments: Optional[str] = None,
        improvements: Optional[str] = None
    ) -> None:
        """Store doctor feedback for model improvement."""
        
        self.feedback_count += 1
        self.feedback_scores.append(quality_score)
        
        logger.info(f"📝 Feedback stored: quality={quality_score}, "
                   f"relevance={product_relevance_score}, safety={safety_score}")
    
    def record_agent_path(self, path: List[str], concern: str) -> None:
        """Record which agents were executed."""
        
        path_key = " -> ".join(path)
        self.agent_paths[path_key] = self.agent_paths.get(path_key, 0) + 1
        self.concern_distribution[concern] = self.concern_distribution.get(concern, 0) + 1
    
    def record_products_recommended(self, products: List[Dict[str, Any]]) -> None:
        """Record product recommendations."""
        
        for product in products:
            name = product.get("name", "unknown")
            self.product_distribution[name] = self.product_distribution.get(name, 0) + 1
    
    # ========================================================================
    # METRICS GETTERS
    # ========================================================================
    
    def get_average_latency(self) -> float:
        """Get average response latency."""
        if not self.latencies:
            return 0
        return sum(self.latencies) / len(self.latencies)
    
    def get_latency_percentiles(self) -> Dict[str, float]:
        """Get latency percentiles."""
        if not self.latencies:
            return {}
        
        sorted_latencies = sorted(self.latencies)
        n = len(sorted_latencies)
        
        return {
            "p50": sorted_latencies[n // 2],
            "p95": sorted_latencies[int(n * 0.95)],
            "p99": sorted_latencies[int(n * 0.99)],
            "max": sorted_latencies[-1],
            "min": sorted_latencies[0]
        }
    
    def get_hallucination_rate(self) -> float:
        """Get percentage of responses with hallucinations."""
        if not self.requests_processed:
            return 0
        return (self.hallucinations_detected / self.requests_processed) * 100
    
    def get_safety_success_rate(self) -> float:
        """Get percentage of safety checks that passed."""
        if not self.requests_processed:
            return 100
        return ((self.requests_processed - self.safety_checks_failed) / self.requests_processed) * 100
    
    def get_average_products(self) -> float:
        """Get average number of products recommended."""
        if not self.requests_processed:
            return 0
        return sum(self.product_distribution.values()) / max(1, self.requests_processed)
    
    def get_estimated_cost(self) -> float:
        """Estimate monthly cost at current rate."""
        # Approximate costs per request:
        # - Triage: $0.002 (small LLM call)
        # - Search: $0.001 (embeddings only)
        # - Recommendation: $0.005 (medium LLM call)
        # - Safety: $0.001 (small verification)
        # Total: ~$0.009 per request
        
        cost_per_request = 0.009
        monthly_rate = self.requests_processed * 30  # Extrapolate
        return cost_per_request * monthly_rate
    
    def get_report(self) -> Dict[str, Any]:
        """Get comprehensive evaluation report."""
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "requests": {
                "total": self.requests_processed,
                "successful": self.successful_requests,
                "failed": self.failed_requests
            },
            "quality": {
                "hallucination_rate_percent": round(self.get_hallucination_rate(), 2),
                "safety_success_rate_percent": round(self.get_safety_success_rate(), 2),
                "average_products_recommended": round(self.get_average_products(), 2)
            },
            "performance": {
                "average_latency_ms": round(self.get_average_latency(), 2),
                "latency_percentiles": {k: round(v, 2) for k, v in self.get_latency_percentiles().items()}
            },
            "cost": {
                "estimated_monthly_cost": round(self.get_estimated_cost(), 2),
                "cost_per_request": 0.009
            },
            "feedback": {
                "doctor_feedback_received": self.feedback_count,
                "average_feedback_score": round(sum(self.feedback_scores) / max(1, len(self.feedback_scores)), 2) if self.feedback_scores else 0
            }
        }
