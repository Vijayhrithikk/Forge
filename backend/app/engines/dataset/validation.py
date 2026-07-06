"""
Validation Pipeline — modular, composable dataset validation.

Every validator owns a single responsibility. Each returns a
standardized result: PASS, WARNING, or FAIL with explanation.
The pipeline is deterministic — same dataset always produces
the same validation outcome.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from app.core import get_logger

logger = get_logger("app.engines.dataset.validation")


# ------------------------------------------------------------------
# Result types
# ------------------------------------------------------------------

class ValidationStatus(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


@dataclass
class ValidationResult:
    """Output from a single validator."""

    validator: str                    # Validator name (e.g. "encoding")
    status: ValidationStatus          # PASS / WARNING / FAIL
    message: str                      # Human-readable summary
    details: Dict[str, Any] = field(default_factory=dict)  # Specific findings

    @property
    def passed(self) -> bool:
        return self.status == ValidationStatus.PASS

    @property
    def failed(self) -> bool:
        return self.status == ValidationStatus.FAIL


@dataclass
class ValidationReport:
    """Aggregated results from the full validation pipeline."""

    project_id: str
    dataset_path: str
    total_records: int
    results: List[ValidationResult] = field(default_factory=list)
    passed_count: int = 0
    warning_count: int = 0
    failed_count: int = 0

    @property
    def is_valid(self) -> bool:
        """Dataset passes if no validators FAILed."""
        return self.failed_count == 0

    @property
    def summary(self) -> str:
        return (
            f"Passed: {self.passed_count}, "
            f"Warnings: {self.warning_count}, "
            f"Failed: {self.failed_count}"
        )


# ------------------------------------------------------------------
# Abstract base
# ------------------------------------------------------------------

class BaseValidator(ABC):
    """Abstract validator — one responsibility per subclass."""

    name: str = "base"

    @abstractmethod
    def validate(self, records: List[Dict[str, Any]]) -> ValidationResult:
        """Run validation on the parsed dataset records.

        Args:
            records: List of parsed JSONL records (dicts).

        Returns:
            ValidationResult with status and details.
        """
        ...


# ------------------------------------------------------------------
# Validators
# ------------------------------------------------------------------

class EncodingValidator(BaseValidator):
    """Validates that the file is UTF-8 encoded and readable."""

    name = "encoding"

    def validate(self, records: List[Dict[str, Any]]) -> ValidationResult:
        # Encoding is validated during upload — this is a pass-through
        # that confirms the loaded records are valid unicode.
        return ValidationResult(
            validator=self.name,
            status=ValidationStatus.PASS,
            message="File is valid UTF-8.",
            details={"encoding": "utf-8"},
        )


class SchemaValidator(BaseValidator):
    """Validates that every record conforms to the expected schema.

    Supported format: { "instruction": str, "input": str, "output": str }
    "input" is optional; "instruction" and "output" are required.
    """

    name = "schema"
    REQUIRED_KEYS = {"instruction", "output"}
    OPTIONAL_KEYS = {"input"}
    ALLOWED_KEYS = REQUIRED_KEYS | OPTIONAL_KEYS

    def validate(self, records: List[Dict[str, Any]]) -> ValidationResult:
        invalid_rows: List[Dict[str, Any]] = []
        extra_key_rows: List[int] = []

        for i, record in enumerate(records):
            keys = set(record.keys())

            # Check required keys
            missing = self.REQUIRED_KEYS - keys
            if missing:
                invalid_rows.append({
                    "line": i + 1,
                    "reason": f"Missing required key(s): {', '.join(sorted(missing))}",
                })
                continue

            # Check for unknown keys
            unknown = keys - self.ALLOWED_KEYS
            if unknown:
                extra_key_rows.append(i + 1)

        if invalid_rows:
            return ValidationResult(
                validator=self.name,
                status=ValidationStatus.FAIL,
                message=f"{len(invalid_rows)} record(s) have invalid schema.",
                details={"invalid_rows": invalid_rows},
            )
        elif extra_key_rows:
            return ValidationResult(
                validator=self.name,
                status=ValidationStatus.WARNING,
                message=f"{len(extra_key_rows)} record(s) contain unexpected keys (ignored).",
                details={"extra_key_rows": extra_key_rows},
            )
        else:
            return ValidationResult(
                validator=self.name,
                status=ValidationStatus.PASS,
                message="All records match the expected schema.",
                details={},
            )


class PromptValidator(BaseValidator):
    """Validates instruction/content quality in prompts."""

    name = "prompt"

    def validate(self, records: List[Dict[str, Any]]) -> ValidationResult:
        empty_prompts: List[int] = []
        short_prompts: List[int] = []
        long_prompts: List[int] = []

        for i, record in enumerate(records):
            instruction = record.get("instruction", "")
            inp = record.get("input", "")
            combined = f"{instruction} {inp}".strip()

            if not combined:
                empty_prompts.append(i + 1)
            elif len(combined) < 10:
                short_prompts.append(i + 1)
            elif len(combined) > 4096:
                long_prompts.append(i + 1)

        details = {
            "empty_prompts": empty_prompts,
            "short_prompts": short_prompts,
            "long_prompts": long_prompts,
        }

        if empty_prompts:
            return ValidationResult(
                validator=self.name,
                status=ValidationStatus.FAIL,
                message=f"{len(empty_prompts)} record(s) have empty prompts.",
                details=details,
            )

        warnings = len(short_prompts) + len(long_prompts)
        if warnings > 0:
            parts = []
            if short_prompts:
                parts.append(f"{len(short_prompts)} very short prompt(s)")
            if long_prompts:
                parts.append(f"{len(long_prompts)} very long prompt(s) (>4096 chars)")
            return ValidationResult(
                validator=self.name,
                status=ValidationStatus.WARNING,
                message=f"Prompt quality issues: {', '.join(parts)}.",
                details=details,
            )

        return ValidationResult(
            validator=self.name,
            status=ValidationStatus.PASS,
            message="All prompts are well-formed.",
            details=details,
        )


class ResponseValidator(BaseValidator):
    """Validates output/response quality."""

    name = "response"

    def validate(self, records: List[Dict[str, Any]]) -> ValidationResult:
        empty_responses: List[int] = []
        short_responses: List[int] = []
        long_responses: List[int] = []

        for i, record in enumerate(records):
            output = record.get("output", "")
            if not output or not output.strip():
                empty_responses.append(i + 1)
            elif len(output) < 5:
                short_responses.append(i + 1)
            elif len(output) > 8192:
                long_responses.append(i + 1)

        details = {
            "empty_responses": empty_responses,
            "short_responses": short_responses,
            "long_responses": long_responses,
        }

        if empty_responses:
            return ValidationResult(
                validator=self.name,
                status=ValidationStatus.FAIL,
                message=f"{len(empty_responses)} record(s) have empty responses.",
                details=details,
            )

        warnings = len(short_responses) + len(long_responses)
        if warnings > 0:
            parts = []
            if short_responses:
                parts.append(f"{len(short_responses)} very short response(s)")
            if long_responses:
                parts.append(f"{len(long_responses)} very long response(s) (>8192 chars)")
            return ValidationResult(
                validator=self.name,
                status=ValidationStatus.WARNING,
                message=f"Response quality issues: {', '.join(parts)}.",
                details=details,
            )

        return ValidationResult(
            validator=self.name,
            status=ValidationStatus.PASS,
            message="All responses are well-formed.",
            details=details,
        )


class DuplicateValidator(BaseValidator):
    """Detects duplicate prompts and exact duplicate records."""

    name = "duplicate"

    def validate(self, records: List[Dict[str, Any]]) -> ValidationResult:
        prompt_map: Dict[str, List[int]] = {}
        exact_map: Dict[str, List[int]] = {}

        for i, record in enumerate(records):
            # Exact match: full record as JSON string
            exact_key = json.dumps(record, sort_keys=True, ensure_ascii=False)
            exact_map.setdefault(exact_key, []).append(i + 1)

            # Prompt match: instruction + input
            prompt_key = json.dumps(
                {"instruction": record.get("instruction", ""), "input": record.get("input", "")},
                sort_keys=True,
                ensure_ascii=False,
            )
            prompt_map.setdefault(prompt_key, []).append(i + 1)

        # Gather duplicates
        exact_dupes = [(k, v) for k, v in exact_map.items() if len(v) > 1]
        prompt_dupes = [(k, v) for k, v in prompt_map.items() if len(v) > 1]

        details = {
            "exact_duplicates": len(exact_dupes),
            "prompt_duplicates": len(prompt_dupes),
            "exact_duplicate_groups": [
                {"lines": lines} for _, lines in exact_dupes[:10]
            ],
            "prompt_duplicate_groups": [
                {"lines": lines} for _, lines in prompt_dupes[:10]
            ],
        }

        total_issues = len(exact_dupes) + len(prompt_dupes)

        if total_issues > 0 and len(exact_dupes) / max(len(records), 1) > 0.3:
            return ValidationResult(
                validator=self.name,
                status=ValidationStatus.WARNING,
                message=(
                    f"Found {len(exact_dupes)} exact duplicate(s) and "
                    f"{len(prompt_dupes)} duplicate prompt(s). "
                    f"Over 30% of the dataset is duplicated."
                ),
                details=details,
            )

        if total_issues > 0:
            return ValidationResult(
                validator=self.name,
                status=ValidationStatus.WARNING,
                message=(
                    f"Found {len(exact_dupes)} exact duplicate(s) and "
                    f"{len(prompt_dupes)} duplicate prompt(s)."
                ),
                details=details,
            )

        return ValidationResult(
            validator=self.name,
            status=ValidationStatus.PASS,
            message="No duplicates detected.",
            details=details,
        )


class SequenceValidator(BaseValidator):
    """Analyzes output sequence lengths for potential issues."""

    name = "sequence"

    def validate(self, records: List[Dict[str, Any]]) -> ValidationResult:
        lengths = []
        for i, record in enumerate(records):
            output = record.get("output", "")
            lengths.append({
                "line": i + 1,
                "output_chars": len(output),
                "output_words": len(output.split()),
            })

        char_lengths = [l["output_chars"] for l in lengths]
        very_short = [l for l in lengths if l["output_chars"] < 10]
        very_long = [l for l in lengths if l["output_chars"] > 4096]

        if very_short and len(very_short) / max(len(records), 1) > 0.5:
            return ValidationResult(
                validator=self.name,
                status=ValidationStatus.WARNING,
                message=f"{len(very_short)} record(s) have very short outputs (<10 chars).",
                details={
                    "very_short_count": len(very_short),
                    "very_long_count": len(very_long),
                    "min_chars": min(char_lengths) if char_lengths else 0,
                    "max_chars": max(char_lengths) if char_lengths else 0,
                    "avg_chars": sum(char_lengths) / len(char_lengths) if char_lengths else 0,
                },
            )

        return ValidationResult(
            validator=self.name,
            status=ValidationStatus.PASS,
            message="Sequence lengths are within acceptable ranges.",
            details={
                "min_chars": min(char_lengths) if char_lengths else 0,
                "max_chars": max(char_lengths) if char_lengths else 0,
                "avg_chars": round(sum(char_lengths) / len(char_lengths), 1) if char_lengths else 0,
            },
        )


# ------------------------------------------------------------------
# Pipeline coordinator
# ------------------------------------------------------------------

class ValidationPipeline:
    """Coordinates the execution of all validators in order.

    The pipeline is deterministic — the same dataset always produces
    the same report. Validators are run sequentially; each gets the
    full record list.
    """

    def __init__(self):
        self._validators: List[BaseValidator] = [
            EncodingValidator(),
            SchemaValidator(),
            PromptValidator(),
            ResponseValidator(),
            DuplicateValidator(),
            SequenceValidator(),
        ]

    def validate(
        self,
        dataset_path: Path,
        project_id: str,
        on_progress=None,
    ) -> ValidationReport:
        """Run the full validation pipeline on a dataset.

        Args:
            dataset_path: Path to the uploaded JSONL file.
            project_id: The owning project.
            on_progress: Optional callback(validator_name, status).

        Returns:
            ValidationReport aggregating all validator results.
        """
        logger.info("validation_started", project_id=project_id, path=str(dataset_path))

        # Load records into memory
        records = self._load_dataset(dataset_path)
        total = len(records)

        logger.info("dataset_loaded", project_id=project_id, records=total)

        # Run validators
        results: List[ValidationResult] = []
        for validator in self._validators:
            logger.info(
                "validator_running",
                project_id=project_id,
                validator=validator.name,
            )
            result = validator.validate(records)
            results.append(result)

            logger.info(
                "validator_completed",
                project_id=project_id,
                validator=validator.name,
                status=result.status.value,
            )

            self._emit_progress(on_progress, validator.name, result.status.value)

        # Build report
        report = ValidationReport(
            project_id=project_id,
            dataset_path=str(dataset_path),
            total_records=total,
            results=results,
            passed_count=sum(1 for r in results if r.status == ValidationStatus.PASS),
            warning_count=sum(1 for r in results if r.status == ValidationStatus.WARNING),
            failed_count=sum(1 for r in results if r.status == ValidationStatus.FAIL),
        )

        logger.info(
            "validation_completed",
            project_id=project_id,
            summary=report.summary,
        )

        return report

    def _load_dataset(self, path: Path) -> List[Dict[str, Any]]:
        """Load all records from a JSONL file into memory."""
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    @staticmethod
    def _emit_progress(on_progress, validator: str, status: str) -> None:
        """Emit progress callback if registered."""
        if on_progress:
            try:
                import asyncio
                asyncio.ensure_future(
                    on_progress({
                        "stage": f"validating_{validator}",
                        "status": status,
                    })
                )
            except Exception:
                pass


# Singleton
validation_pipeline = ValidationPipeline()
