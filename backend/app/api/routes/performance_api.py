"""Performance API — benchmark, profile, analyze, optimize."""
import json
from pathlib import Path
from fastapi import APIRouter
from app.schemas.responses import SuccessResponse
from app.performance.benchmark import BenchmarkEngine
from app.performance.registry import PerformanceRegistry
from app.performance.budget import BudgetEvaluator
from app.performance.analyzer import PerformanceAnalyzer
from app.performance.optimization import OptimizationEngine, OptimizationRegistry
from app.performance.report import generate_baseline, generate_scorecard, generate_certificate

router = APIRouter(prefix="/performance", tags=["performance"])

_registry = PerformanceRegistry()
_analyzer = PerformanceAnalyzer(_registry)
_budget_eval = BudgetEvaluator()
_opt_engine = OptimizationEngine()
_opt_registry = OptimizationRegistry()


@router.get("/baseline", response_model=SuccessResponse)
async def get_baseline():
    engine = BenchmarkEngine()
    results = engine.run_all()
    budget_eval = _budget_eval.evaluate(results)
    _registry.record(f"bench_{results['generated_at']}", results)
    baseline = generate_baseline(results, budget_eval)
    engine.save(results, Path("data/performance/performance_baseline.json"))
    return SuccessResponse(message=f"Benchmark: {results['total_duration']}s", data=baseline)


@router.get("/registry", response_model=SuccessResponse)
async def get_registry():
    history = _registry.history()
    return SuccessResponse(message=f"{len(history)} benchmarks recorded.", data={"entries": history})


@router.get("/budgets", response_model=SuccessResponse)
async def get_budgets():
    engine = BenchmarkEngine()
    results = engine.run_all()
    budget_eval = _budget_eval.evaluate(results)
    return SuccessResponse(message=f"Budgets: {budget_eval['status']}", data=budget_eval)


@router.post("/profile", response_model=SuccessResponse)
async def run_profile():
    engine = BenchmarkEngine()
    results = engine.run_all()
    budget_eval = _budget_eval.evaluate(results)
    _registry.record(f"profile_{results['generated_at']}", results)
    scorecard = generate_scorecard(results, budget_eval)
    return SuccessResponse(message=f"Score: {scorecard['overall']}/100", data={
        "benchmark": results, "budgets": budget_eval, "scorecard": scorecard,
    })


@router.get("/comparison", response_model=SuccessResponse)
async def get_comparison():
    engine = BenchmarkEngine()
    current = engine.run_all()
    comparison = _analyzer.compare(current)
    return SuccessResponse(message=f"Comparison: {comparison.get('status', 'ok')}", data=comparison)


@router.get("/trends", response_model=SuccessResponse)
async def get_trends():
    trends = _analyzer.analyze_trends()
    return SuccessResponse(message=f"Trend: {trends.get('trend', 'unknown')}", data=trends)


@router.get("/score", response_model=SuccessResponse)
async def get_score():
    engine = BenchmarkEngine()
    results = engine.run_all()
    budget_eval = _budget_eval.evaluate(results)
    score = _analyzer.score(results, budget_eval)
    return SuccessResponse(message=f"Score: {score['overall']}/100 ({score['grade']})", data=score)


@router.get("/recommendations", response_model=SuccessResponse)
async def get_recommendations():
    engine = BenchmarkEngine()
    results = engine.run_all()
    budget_eval = _budget_eval.evaluate(results)
    recs = _analyzer.recommend(results, budget_eval)
    return SuccessResponse(message=f"{len(recs)} recommendations", data={"recommendations": recs})


@router.get("/optimizations", response_model=SuccessResponse)
async def get_optimizations():
    accepted = _opt_registry.list_accepted()
    rejected = _opt_registry.list_rejected()
    return SuccessResponse(message=f"{len(accepted)} accepted, {len(rejected)} rejected", data={
        "accepted": accepted, "rejected": rejected,
    })


@router.get("/optimization-registry", response_model=SuccessResponse)
async def get_optimization_registry():
    entries = _opt_registry._load()
    return SuccessResponse(message=f"{len(entries)} optimization(s) recorded.", data={"entries": entries})


@router.get("/optimization-manifest", response_model=SuccessResponse)
async def get_optimization_manifest():
    entries = _opt_registry._load()
    accepted = [e for e in entries if e.get("accepted")]
    return SuccessResponse(message=f"Latest optimization manifest.", data={
        "total": len(entries), "accepted": len(accepted),
        "latest": entries[-1] if entries else None,
    })


@router.post("/optimize", response_model=SuccessResponse)
async def run_optimization():
    engine = BenchmarkEngine()
    callbacks = {}
    result = _opt_engine.evaluate_candidate(
        "noop_test", lambda: None, lambda: None, callbacks,
    )
    return SuccessResponse(message=f"Optimization: {'accepted' if result['accepted'] else 'rejected'}", data=result)


@router.post("/benchmark-compare", response_model=SuccessResponse)
async def benchmark_compare():
    engine = BenchmarkEngine()
    results = engine.run_all()
    budget_eval = _budget_eval.evaluate(results)
    scorecard = _analyzer.score(results, budget_eval)
    cert = generate_certificate(results, scorecard, budget_eval)
    with open(Path("data/performance/performance_certificate.json"), "w") as f:
        json.dump(cert, f, indent=2)
    return SuccessResponse(message=f"Certificate: {cert['overall_status']}", data=cert)
