"""Validation Dataset Generator — deterministic 100-300 instruction samples."""
import json, hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any
from app.core import settings

INSTRUCTIONS = [
    {"instruction": "Explain what machine learning is.", "input": "", "output": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."},
    {"instruction": "What is the capital of France?", "input": "", "output": "The capital of France is Paris."},
    {"instruction": "Write a short poem about coding.", "input": "", "output": "Lines of logic, clean and bright, building worlds in black and white."},
    {"instruction": "Convert this sentence to past tense: I am running.", "input": "", "output": "I was running."},
    {"instruction": "What is 15 + 27?", "input": "", "output": "15 + 27 = 42."},
    {"instruction": "Define 'algorithm' in one sentence.", "input": "", "output": "An algorithm is a step-by-step procedure for solving a problem or accomplishing a task."},
    {"instruction": "Write a haiku about AI.", "input": "", "output": "Silicon dreams wake, neural pathways learn and grow, machines think anew."},
    {"instruction": "Translate 'Hello' to Spanish.", "input": "", "output": "Hola."},
    {"instruction": "What is the boiling point of water?", "input": "", "output": "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at sea level."},
    {"instruction": "Summarize the plot of Romeo and Juliet.", "input": "", "output": "Two young lovers from feuding families fall in love, marry in secret, and ultimately die tragically due to misunderstanding and fate."},
]

def generate_validation_dataset(sample_count: int = 100, output_dir: Path = Path("data")) -> Dict[str, Any]:
    records = []
    for i in range(sample_count):
        template = INSTRUCTIONS[i % len(INSTRUCTIONS)]
        records.append({**template, "instruction": f"{template['instruction']} (sample {i+1})"})
    path = output_dir / "validation_dataset.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    dataset_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "1.0", "forge_version": settings.app_version,
        "dataset_hash": dataset_hash, "sample_count": len(records),
        "generation_method": "deterministic_template", "created_at": datetime.now(timezone.utc).isoformat(),
    }
    mf_path = output_dir / "validation_dataset_manifest.json"
    with open(mf_path, "w") as f: json.dump(manifest, f, indent=2)
    return {"records": records, "hash": dataset_hash, "count": len(records), "path": str(path), "manifest": manifest}
