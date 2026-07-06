"""
Tokenization Runtime — tokenizes datasets using the PreparedModel's tokenizer.
Applies truncation, padding, optional packing. Generates tokenization report.
"""

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from app.core import settings, get_logger
from app.training.exceptions import TokenizationError

logger = get_logger("app.training.tokenization")


class TokenizationRuntime:
    """Tokenizes a dataset using the tokenizer from a PreparedModel."""

    def __init__(self, tokenizer: Any = None, max_length: int = 2048,
                 padding: str = "max_length", truncation: bool = True):
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._padding = padding
        self._truncation = truncation
        self._report: Dict[str, Any] = {}

    def tokenize(self, records: List[Dict], training_plan: Optional[Dict] = None) -> Dict[str, Any]:
        """Tokenize a list of records.

        If a real HuggingFace tokenizer is available, uses it.
        Otherwise falls back to the heuristic TokenEstimator from Sprint 1.
        """
        if training_plan:
            hp = training_plan.get("hyperparameters", {})
            self._max_length = hp.get("max_sequence_length", self._max_length)

        if self._tokenizer and hasattr(self._tokenizer, 'encode'):
            tokenized = self._tokenize_with_hf(records)
        else:
            tokenized = self._tokenize_heuristic(records)

        total_tokens = sum(t["token_count"] for t in tokenized)
        lengths = [t["token_count"] for t in tokenized]

        self._report = {
            "schema_version": "1.0",
            "forge_version": settings.app_version,
            "tokenizer": getattr(self._tokenizer, "name_or_path", "heuristic") if self._tokenizer else "heuristic",
            "max_length": self._max_length,
            "padding": self._padding,
            "truncation": self._truncation,
            "sample_count": len(tokenized),
            "total_tokens": total_tokens,
            "average_tokens": round(total_tokens / max(len(tokenized), 1), 1),
            "max_tokens": max(lengths) if lengths else 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        logger.info("tokenization_complete", samples=len(tokenized), total_tokens=total_tokens)
        return {"tokenized": tokenized, "total_tokens": total_tokens, "report": self._report}

    def save_report(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "tokenization_report.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._report, f, indent=2, ensure_ascii=False)
        return path

    def _tokenize_with_hf(self, records: List[Dict]) -> List[Dict]:
        results = []
        for r in records:
            instr = r.get("instruction", "")
            inp = r.get("input", "")
            output = r.get("output", "")
            text = f"{instr}\n{inp}\n{output}" if inp else f"{instr}\n{output}"
            tokens = self._tokenizer.encode(text, truncation=self._truncation, max_length=self._max_length)
            results.append({"text": text, "tokens": tokens, "token_count": len(tokens)})
        return results

    def _tokenize_heuristic(self, records: List[Dict]) -> List[Dict]:
        from app.engines.dataset.analysis import TokenEstimator
        results = []
        for r in records:
            instr = r.get("instruction", "")
            inp = r.get("input", "")
            output = r.get("output", "")
            text = f"{instr} {inp} {output}".strip()
            count = TokenEstimator.estimate(text)
            results.append({"text": text, "token_count": count})
        return results
