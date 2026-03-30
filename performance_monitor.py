"""
Performance monitoring and profiling for the dialogue system.
Tracks API calls, response times, token usage, and costs.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import time
from functools import wraps


@dataclass
class APICallMetrics:
    """Metrics for a single API call"""
    timestamp: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    success: bool
    error: Optional[str] = None
    
    @property
    def estimated_cost(self) -> float:
        """Estimate cost based on OpenAI pricing (GPT-4o-mini)"""
        # Pricing as of 2024: $0.150 per 1M input tokens, $0.600 per 1M output tokens
        input_cost = (self.prompt_tokens / 1_000_000) * 0.150
        output_cost = (self.completion_tokens / 1_000_000) * 0.600
        return input_cost + output_cost


@dataclass
class PerformanceStats:
    """Aggregate performance statistics"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost: float = 0.0
    avg_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    
    @property
    def success_rate(self) -> float:
        return (self.successful_calls / self.total_calls * 100) if self.total_calls > 0 else 0.0


class PerformanceMonitor:
    """Monitor and analyze system performance"""
    
    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path or Path("logs/performance.jsonl")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.metrics_history: List[APICallMetrics] = []
        
    def log_api_call(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        success: bool = True,
        error: Optional[str] = None
    ):
        """Log a single API call"""
        metric = APICallMetrics(
            timestamp=datetime.utcnow().isoformat(),
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=latency_ms,
            success=success,
            error=error
        )
        
        self.metrics_history.append(metric)
        
        # Write to JSONL
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "timestamp": metric.timestamp,
                "model": metric.model,
                "prompt_tokens": metric.prompt_tokens,
                "completion_tokens": metric.completion_tokens,
                "total_tokens": metric.total_tokens,
                "latency_ms": metric.latency_ms,
                "estimated_cost": metric.estimated_cost,
                "success": metric.success,
                "error": metric.error
            }) + "\n")
    
    def get_stats(self, last_n: Optional[int] = None) -> PerformanceStats:
        """Get aggregate statistics"""
        history = self.metrics_history[-last_n:] if last_n else self.metrics_history
        
        if not history:
            return PerformanceStats()
        
        stats = PerformanceStats()
        stats.total_calls = len(history)
        stats.successful_calls = sum(1 for m in history if m.success)
        stats.failed_calls = sum(1 for m in history if not m.success)
        stats.total_tokens = sum(m.total_tokens for m in history)
        stats.total_prompt_tokens = sum(m.prompt_tokens for m in history)
        stats.total_completion_tokens = sum(m.completion_tokens for m in history)
        stats.total_cost = sum(m.estimated_cost for m in history)
        
        latencies = [m.latency_ms for m in history if m.success]
        if latencies:
            stats.avg_latency_ms = sum(latencies) / len(latencies)
            stats.min_latency_ms = min(latencies)
            stats.max_latency_ms = max(latencies)
        
        return stats
    
    def generate_report(self) -> str:
        """Generate a human-readable performance report"""
        stats = self.get_stats()
        
        report = []
        report.append("=" * 60)
        report.append("PERFORMANCE REPORT")
        report.append("=" * 60)
        report.append(f"Total API calls: {stats.total_calls}")
        report.append(f"Success rate: {stats.success_rate:.1f}%")
        report.append(f"Failed calls: {stats.failed_calls}")
        report.append("")
        report.append("Token Usage:")
        report.append(f"  Prompt tokens: {stats.total_prompt_tokens:,}")
        report.append(f"  Completion tokens: {stats.total_completion_tokens:,}")
        report.append(f"  Total tokens: {stats.total_tokens:,}")
        report.append("")
        report.append("Latency:")
        report.append(f"  Average: {stats.avg_latency_ms:.0f} ms")
        report.append(f"  Min: {stats.min_latency_ms:.0f} ms")
        report.append(f"  Max: {stats.max_latency_ms:.0f} ms")
        report.append("")
        report.append(f"Estimated total cost: ${stats.total_cost:.4f}")
        report.append("=" * 60)
        
        return "\n".join(report)


def monitor_api_call(monitor: Optional[PerformanceMonitor] = None):
    """Decorator to monitor API calls"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if monitor is None:
                return func(*args, **kwargs)
            
            start_time = time.time()
            success = True
            error = None
            result = None
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                latency_ms = (time.time() - start_time) * 1000
                
                # Try to extract token usage from result
                prompt_tokens = 0
                completion_tokens = 0
                model = "unknown"
                
                if result and hasattr(result, 'usage'):
                    prompt_tokens = result.usage.prompt_tokens
                    completion_tokens = result.usage.completion_tokens
                    model = getattr(result, 'model', 'unknown')
                
                monitor.log_api_call(
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=latency_ms,
                    success=success,
                    error=error
                )
        
        return wrapper
    return decorator

