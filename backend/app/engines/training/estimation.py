"""
Estimation Engine — VRAM, training time, adapter size, and throughput estimators.

All estimates are approximate. Assumptions are documented.
Never present estimates as exact values.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any

from app.core import get_logger
from app.engines.training.registry import model_registry, ModelEntry
from app.engines.training.config import TrainingConfig

logger = get_logger("app.engines.training.estimation")


# ------------------------------------------------------------------
# Estimation assumptions (documented)
# ------------------------------------------------------------------

# Bytes per parameter for different precisions (approximate)
BYTES_PER_PARAM = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
}

# LoRA memory overhead: ~2× for adapter weights + optimizer states
LORA_MEMORY_FACTOR = 2.5

# Optimizer state memory (AdamW stores 2 momentum buffers = 2× params)
OPTIMIZER_STATE_FACTOR = {
    "adamw": 2.0,
    "paged_adamw": 1.0,   # 8-bit states
    "adafactor": 0.5,
    "sgd": 1.0,
}

# Activation memory: rough estimate as fraction of model memory
ACTIVATION_MEMORY_FACTOR = 0.15

# Safety buffer (10% of total)
SAFETY_BUFFER_FACTOR = 0.10

# Training throughput assumptions (tokens/second for fp16/bf16 on consumer GPUs)
# These are rough averages — actual values vary by model and hardware.
GPU_THROUGHPUT: Dict[str, float] = {
    "NVIDIA RTX 4090 24GB": 12000,
    "NVIDIA RTX 4080 16GB": 8000,
    "NVIDIA RTX 4060 8GB": 4000,
    "NVIDIA RTX 3060 12GB": 3500,
    "NVIDIA Tesla T4 16GB": 2500,
    "NVIDIA Tesla P100 16GB": 3000,
    "NVIDIA RTX 3050 6GB": 2000,
    "NVIDIA A100 40GB": 25000,
    "NVIDIA A100 80GB": 30000,
}


@dataclass
class EstimationResult:
    """Aggregated estimation results."""

    # VRAM
    model_memory_gb: float = 0.0
    lora_memory_gb: float = 0.0
    optimizer_memory_gb: float = 0.0
    activation_memory_gb: float = 0.0
    safety_buffer_gb: float = 0.0
    total_vram_gb: float = 0.0

    # Adapter
    trainable_parameters: int = 0
    adapter_size_mb: float = 0.0

    # Training
    estimated_tokens_per_step: int = 0
    estimated_steps: int = 0
    estimated_throughput_tokens_per_sec: float = 0.0
    estimated_duration_minutes: float = 0.0
    estimated_duration_display: str = ""

    # Metadata
    assumptions: List[str] = field(default_factory=list)


class EstimationEngine:
    """Estimates VRAM, training time, and adapter size for a training config.

    All estimates are approximate. The engine documents its assumptions
    so users understand the limitations.
    """

    def estimate(
        self,
        config: TrainingConfig,
        dataset_sample_count: int = 0,
        dataset_avg_tokens: float = 0,
    ) -> EstimationResult:
        """Produce a complete estimation for the given training config.

        Args:
            config: The validated training configuration.
            dataset_sample_count: Number of training samples.
            dataset_avg_tokens: Average tokens per sample.

        Returns:
            EstimationResult with all estimates and documented assumptions.
        """
        result = EstimationResult()

        try:
            model = model_registry.get_model(config.model_id)
        except KeyError:
            return result

        precision = config.hyperparams.precision
        bytes_per_param = BYTES_PER_PARAM.get(precision, 2)
        lora_rank = config.lora.rank
        target_modules = config.lora.target_modules or model.lora_defaults.get("target_modules", [])
        eff_batch = config.hyperparams.effective_batch_size()

        # ---- 1. Model memory ----
        # Base model loaded in specified precision
        result.model_memory_gb = (model.parameters * bytes_per_param) / (1024 ** 3)

        # ---- 2. LoRA memory ----
        # LoRA adds small trainable matrices. Estimated as:
        # rank × hidden_dim × num_modules × 2 (A+B matrices) × LORA_MEMORY_FACTOR
        # For a ~3B model with hidden_dim ~3072 and 7 target modules:
        trainable_params = lora_rank * 3072 * len(target_modules) * 2  # rough estimate
        result.trainable_parameters = trainable_params
        lora_mem = (trainable_params * bytes_per_param * LORA_MEMORY_FACTOR) / (1024 ** 3)
        result.lora_memory_gb = round(lora_mem, 2)

        # ---- 3. Optimizer memory ----
        opt_factor = OPTIMIZER_STATE_FACTOR.get(config.hyperparams.optimizer, 2.0)
        result.optimizer_memory_gb = round(
            (trainable_params * bytes_per_param * opt_factor) / (1024 ** 3), 2
        )

        # ---- 4. Activation memory ----
        result.activation_memory_gb = round(
            result.model_memory_gb * ACTIVATION_MEMORY_FACTOR, 2
        )

        # ---- 5. Total with safety buffer ----
        subtotal = (
            result.model_memory_gb
            + result.lora_memory_gb
            + result.optimizer_memory_gb
            + result.activation_memory_gb
        )
        result.safety_buffer_gb = round(subtotal * SAFETY_BUFFER_FACTOR, 2)
        result.total_vram_gb = round(subtotal + result.safety_buffer_gb, 2)

        # ---- 6. Adapter size ----
        # Final adapter file size (safetensors, compressed)
        result.adapter_size_mb = round(
            (trainable_params * bytes_per_param) / (1024 * 1024), 1
        )

        # ---- 7. Training time ----
        if dataset_sample_count > 0 and dataset_avg_tokens > 0:
            total_tokens = dataset_sample_count * dataset_avg_tokens
            tokens_per_epoch = total_tokens
            total_training_tokens = tokens_per_epoch * config.hyperparams.epochs

            # Steps per epoch
            steps_per_epoch = max(1, dataset_sample_count // eff_batch)
            total_steps = steps_per_epoch * config.hyperparams.epochs

            result.estimated_tokens_per_step = eff_batch * config.hyperparams.max_sequence_length
            result.estimated_steps = total_steps

            # Throughput for the recommended GPU
            gpu_name = model.recommended_gpu
            throughput = GPU_THROUGHPUT.get(gpu_name, 5000)
            if precision == "fp32":
                throughput *= 0.6  # FP32 is slower
            result.estimated_throughput_tokens_per_sec = throughput

            # Duration
            seconds = total_training_tokens / throughput
            result.estimated_duration_minutes = round(seconds / 60, 1)

            if seconds < 60:
                result.estimated_duration_display = f"< 1 minute"
            elif seconds < 3600:
                result.estimated_duration_display = f"{seconds / 60:.0f} minutes"
            else:
                result.estimated_duration_display = f"{seconds / 3600:.1f} hours"

        # ---- Assumptions ----
        result.assumptions = [
            f"Model: {model.name} ({model.parameters_display} params) loaded in {precision}",
            f"LoRA rank={lora_rank}, target modules={len(target_modules)}",
            f"Effective batch size: {eff_batch}",
            f"GPU: {model.recommended_gpu} (~{GPU_THROUGHPUT.get(model.recommended_gpu, 5000)} tok/s)",
            "Estimates are approximate. Actual values depend on hardware, data, and framework overhead.",
            "VRAM estimates include model + LoRA + optimizer + activations + 10% safety buffer.",
            f"Optimizer memory factor: {OPTIMIZER_STATE_FACTOR.get(config.hyperparams.optimizer, 2.0)}× (for {config.hyperparams.optimizer})",
        ]

        logger.info(
            "estimation_complete",
            model=model.id,
            total_vram_gb=result.total_vram_gb,
            adapter_size_mb=result.adapter_size_mb,
            duration_minutes=result.estimated_duration_minutes,
        )

        return result

    def compatible_gpus(self, total_vram_gb: float) -> List[Dict[str, Any]]:
        """Find compatible GPUs for the estimated VRAM requirement."""
        results = []
        for gpu_id, gpu in model_registry._gpus.items():
            if gpu.vram_gb >= total_vram_gb:
                status = "compatible"
            elif gpu.vram_gb >= total_vram_gb * 0.85:
                status = "limited"
            else:
                status = "unsupported"

            results.append({
                "id": gpu_id,
                "name": gpu.name,
                "vram_gb": gpu.vram_gb,
                "status": status,
                "headroom_gb": round(gpu.vram_gb - total_vram_gb, 1),
                "notes": gpu.notes,
            })

        return results


# Singleton
estimation_engine = EstimationEngine()
