"""
Dataset Analysis Engine — statistics, quality scoring, and reporting.

Reads a validated dataset and produces:
- Dataset statistics (tokens, lengths, distributions)
- Quality score (0–100) with weighted factors
- Recommendations for improvement

Deterministic: same dataset always produces the same analysis.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from app.core import get_logger
from app.engines.dataset.validation import ValidationReport, ValidationStatus

logger = get_logger("app.engines.dataset.analysis")


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class DatasetStatistics:
    """Computed statistics for a dataset."""

    sample_count: int = 0
    total_chars: int = 0
    avg_prompt_chars: float = 0.0
    avg_response_chars: float = 0.0
    avg_prompt_tokens: float = 0.0
    avg_response_tokens: float = 0.0
    min_tokens: int = 0
    max_tokens: int = 0
    median_tokens: int = 0
    p95_tokens: int = 0
    prompt_char_lengths: List[int] = field(default_factory=list)
    response_char_lengths: List[int] = field(default_factory=list)
    token_counts: List[int] = field(default_factory=list)
    duplicate_prompt_count: int = 0
    duplicate_exact_count: int = 0
    empty_prompt_count: int = 0
    empty_response_count: int = 0
    instruction_count: int = 0
    input_count: int = 0
    longest_sample_tokens: int = 0
    shortest_sample_tokens: int = 0
    estimated_training_tokens: int = 0
    estimated_adapter_size_mb: float = 0.0
    estimated_training_minutes: float = 0.0


@dataclass
class QualityReport:
    """Quality score and analysis."""

    score: int                                         # 0–100
    grade: str                                         # Excellent/Good/Needs Improvement/Poor
    strengths: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    factor_scores: Dict[str, int] = field(default_factory=dict)


# ------------------------------------------------------------------
# Token Estimator (simple word-based, no ML dependency)
# ------------------------------------------------------------------

class TokenEstimator:
    """Estimates token counts without loading a real tokenizer.

    Uses a simple word-based heuristic: ~1.3 tokens per word on average
    for English text. This is a documented approximation — the real
    tokenizer in Sprint 2 will provide exact counts.

    Formula: tokens ≈ words × 1.3
    Source: empirical average for Llama-family tokenizers on English text.
    """

    # Average tokens-per-word ratio for Llama tokenizers on English
    TOKENS_PER_WORD = 1.3
    # Bytes per token (very rough, for adapter size estimation)
    BYTES_PER_TOKEN = 2.0
    # Tokens per second for a consumer GPU (RTX 3090/4090-class)
    TOKENS_PER_SECOND = 8000

    @classmethod
    def estimate(cls, text: str) -> int:
        """Estimate token count for a string."""
        words = len(text.split())
        if words == 0:
            return 0
        return max(1, int(words * cls.TOKENS_PER_WORD))

    @classmethod
    def estimate_batch(cls, texts: List[str]) -> List[int]:
        """Estimate token counts for a list of strings."""
        return [cls.estimate(t) for t in texts]


# ------------------------------------------------------------------
# Analysis Engine
# ------------------------------------------------------------------

class AnalysisEngine:
    """Computes statistics and quality scores for validated datasets.

    Reads the uploaded JSONL file, computes statistics using the
    TokenEstimator, and generates a quality score based on weighted
    factors. Never modifies the original dataset.
    """

    def __init__(self):
        self._token_estimator = TokenEstimator()

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def compute_statistics(
        self,
        dataset_path: Path,
        validation_report: ValidationReport,
    ) -> DatasetStatistics:
        """Compute comprehensive dataset statistics.

        Args:
            dataset_path: Path to the JSONL dataset file.
            validation_report: The already-run validation report.

        Returns:
            DatasetStatistics with all computed metrics.
        """
        records = self._load_dataset(dataset_path)
        stats = DatasetStatistics()
        stats.sample_count = len(records)

        if not records:
            return stats

        # Compute per-sample metrics
        prompt_chars = []
        response_chars = []
        all_tokens = []
        seen_prompts = set()
        seen_exact = set()

        for rec in records:
            instruction = rec.get("instruction", "")
            inp = rec.get("input", "")
            output = rec.get("output", "")

            prompt_text = f"{instruction} {inp}".strip()
            prompt_chars.append(len(prompt_text))
            response_chars.append(len(output))

            # Token estimates
            prompt_tokens = TokenEstimator.estimate(prompt_text)
            response_tokens = TokenEstimator.estimate(output)
            total_tokens = prompt_tokens + response_tokens
            all_tokens.append(total_tokens)

            # Duplicate tracking
            prompt_key = json.dumps(
                {"instruction": instruction, "input": inp},
                sort_keys=True,
                ensure_ascii=False,
            )
            exact_key = json.dumps(rec, sort_keys=True, ensure_ascii=False)
            seen_prompts.add(prompt_key)
            seen_exact.add(exact_key)

            # Track counts
            if not instruction.strip():
                stats.empty_prompt_count += 1
            if not output.strip():
                stats.empty_response_count += 1
            if inp.strip():
                stats.input_count += 1
            stats.instruction_count += 1

        # Aggregate statistics
        sorted_tokens = sorted(all_tokens)
        n = len(sorted_tokens)

        stats.total_chars = sum(prompt_chars) + sum(response_chars)
        stats.avg_prompt_chars = round(sum(prompt_chars) / n, 1)
        stats.avg_response_chars = round(sum(response_chars) / n, 1)
        stats.avg_prompt_tokens = round(
            sum(TokenEstimator.estimate(p) for p in [f"{r.get('instruction', '')} {r.get('input', '')}".strip() for r in records]) / n, 1
        )
        stats.avg_response_tokens = round(sum(TokenEstimator.estimate(r.get("output", "")) for r in records) / n, 1)
        stats.min_tokens = sorted_tokens[0]
        stats.max_tokens = sorted_tokens[-1]
        stats.median_tokens = sorted_tokens[n // 2]
        stats.p95_tokens = sorted_tokens[int(n * 0.95)] if n > 1 else sorted_tokens[-1]
        stats.longest_sample_tokens = sorted_tokens[-1]
        stats.shortest_sample_tokens = sorted_tokens[0]
        stats.prompt_char_lengths = prompt_chars
        stats.response_char_lengths = response_chars
        stats.token_counts = all_tokens

        # Duplicates
        stats.duplicate_prompt_count = n - len(seen_prompts)
        stats.duplicate_exact_count = n - len(seen_exact)

        # Estimates
        # Training tokens ≈ average tokens per sample × samples × epochs (assuming 3)
        stats.estimated_training_tokens = sum(all_tokens) * 3
        # Adapter size ≈ trainable_params × bytes_per_param (rough LoRA estimate)
        # For rank=16, alpha=32: ~10M params × 2 bytes ≈ 20MB
        stats.estimated_adapter_size_mb = round(
            (stats.estimated_training_tokens * TokenEstimator.BYTES_PER_TOKEN) / (1024 * 1024 * 100),
            1,
        )
        # Training time ≈ total_tokens / tokens_per_second / 60
        stats.estimated_training_minutes = round(
            sum(all_tokens) * 3 / TokenEstimator.TOKENS_PER_SECOND / 60, 1
        )

        return stats

    # ------------------------------------------------------------------
    # Quality Scoring
    # ------------------------------------------------------------------

    def compute_quality(
        self,
        stats: DatasetStatistics,
        validation_report: ValidationReport,
    ) -> QualityReport:
        """Compute a quality score (0–100) with weighted factors.

        Scoring methodology:
        - Schema validity: 25 points
        - Encoding: 10 points
        - Prompt quality: 20 points
        - Response quality: 20 points
        - Uniqueness: 15 points
        - Sequence balance: 10 points

        Each factor contributes its weighted share. Deductions are
        proportional to the severity and volume of issues found.

        Args:
            stats: Computed dataset statistics.
            validation_report: The validation report.

        Returns:
            QualityReport with score, grade, and recommendations.
        """
        strengths: List[str] = []
        warnings_list: List[str] = []
        recommendations: List[str] = []
        factor_scores: Dict[str, int] = {}

        # ---- Schema (25 points) ----
        schema_result = self._find_result(validation_report, "schema")
        if schema_result and schema_result.status == ValidationStatus.PASS:
            factor_scores["schema"] = 25
            strengths.append("Schema is valid — all records have the expected format.")
        elif schema_result and schema_result.status == ValidationStatus.FAIL:
            invalid = schema_result.details.get("invalid_rows", [])
            factor_scores["schema"] = max(0, 25 - len(invalid) * 5)
            warnings_list.append(f"{len(invalid)} records have invalid schema.")
            recommendations.append("Fix or remove records with missing keys.")
        else:
            factor_scores["schema"] = 20  # WARNING

        # ---- Encoding (10 points) ----
        encoding_result = self._find_result(validation_report, "encoding")
        if encoding_result and encoding_result.status == ValidationStatus.PASS:
            factor_scores["encoding"] = 10
            strengths.append("UTF-8 encoding confirmed.")
        else:
            factor_scores["encoding"] = 0
            warnings_list.append("Encoding issues detected.")
            recommendations.append("Re-save the file with UTF-8 encoding.")

        # ---- Prompt quality (20 points) ----
        prompt_result = self._find_result(validation_report, "prompt")
        empty_p = len(prompt_result.details.get("empty_prompts", [])) if prompt_result else 0
        if empty_p == 0:
            factor_scores["prompt"] = 20
            strengths.append("All prompts are non-empty and well-formed.")
        else:
            ratio = empty_p / max(stats.sample_count, 1)
            factor_scores["prompt"] = max(0, 20 - int(ratio * 40))
            warnings_list.append(f"{empty_p} empty prompt(s).")
            recommendations.append("Remove records with empty prompts or add meaningful instructions.")

        # ---- Response quality (20 points) ----
        response_result = self._find_result(validation_report, "response")
        empty_r = len(response_result.details.get("empty_responses", [])) if response_result else 0
        if empty_r == 0:
            factor_scores["response"] = 20
            strengths.append("All responses are non-empty.")
        else:
            ratio = empty_r / max(stats.sample_count, 1)
            factor_scores["response"] = max(0, 20 - int(ratio * 40))
            warnings_list.append(f"{empty_r} empty response(s).")
            recommendations.append("Filter out records with empty responses before training.")

        # ---- Uniqueness (15 points) ----
        dup_ratio = stats.duplicate_exact_count / max(stats.sample_count, 1)
        if dup_ratio == 0:
            factor_scores["uniqueness"] = 15
            strengths.append("No duplicate records detected.")
        elif dup_ratio < 0.1:
            factor_scores["uniqueness"] = 12
            warnings_list.append(f"{stats.duplicate_exact_count} exact duplicate(s) found.")
        elif dup_ratio < 0.3:
            factor_scores["uniqueness"] = 8
            warnings_list.append(f"Significant duplication ({stats.duplicate_exact_count} records).")
            recommendations.append("Consider deduplicating before training.")
        else:
            factor_scores["uniqueness"] = 3
            warnings_list.append(f"High duplication ({stats.duplicate_exact_count} records, {int(dup_ratio * 100)}%).")
            recommendations.append("Deduplicate the dataset to avoid overfitting on repeated samples.")

        # ---- Sequence balance (10 points) ----
        if stats.sample_count > 0:
            # Check for extreme variance in response lengths
            if stats.response_char_lengths:
                mean_rl = sum(stats.response_char_lengths) / len(stats.response_char_lengths)
                cv = (statistics_stdev(stats.response_char_lengths) / mean_rl) if mean_rl > 0 else 0
                if cv < 0.5:
                    factor_scores["sequence"] = 10
                    strengths.append("Response lengths are well-balanced.")
                elif cv < 1.0:
                    factor_scores["sequence"] = 7
                    warnings_list.append("Response lengths vary significantly.")
                else:
                    factor_scores["sequence"] = 4
                    warnings_list.append("Response lengths are highly variable.")
                    recommendations.append("Consider balancing short and long responses.")
            else:
                factor_scores["sequence"] = 10
        else:
            factor_scores["sequence"] = 10

        # Compute final score
        total_score = sum(factor_scores.values())
        total_score = max(0, min(100, total_score))

        # Determine grade
        if total_score >= 90:
            grade = "Excellent"
        elif total_score >= 75:
            grade = "Good"
        elif total_score >= 50:
            grade = "Needs Improvement"
        else:
            grade = "Poor"

        return QualityReport(
            score=total_score,
            grade=grade,
            strengths=strengths,
            warnings=warnings_list,
            recommendations=recommendations,
            factor_scores=factor_scores,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_dataset(self, path: Path) -> List[Dict[str, Any]]:
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    @staticmethod
    def _find_result(report: ValidationReport, validator_name: str):
        for r in report.results:
            if r.validator == validator_name:
                return r
        return None


# ------------------------------------------------------------------
# Statistics helper (no numpy dependency)
# ------------------------------------------------------------------

def statistics_stdev(values: List[float]) -> float:
    """Compute population standard deviation without numpy."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5


# Singleton
analysis_engine = AnalysisEngine()
