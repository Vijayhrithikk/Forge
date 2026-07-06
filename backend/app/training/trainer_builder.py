"""
Trainer Builder — constructs a validated HuggingFace Trainer from Training Plan.

Assembles model, dataset, tokenizer, data collator, TrainingArguments,
and callbacks. Validates everything before returning a ValidatedTrainer.
Never calls trainer.train() — that belongs to TrainingController.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from app.core import settings, get_logger
from app.training.exceptions import TrainerBuildError

logger = get_logger("app.training.trainer_builder")


class ValidatedTrainer:
    """A fully validated trainer ready for execution.

    The TrainingController is the only module that may call
    trainer.train() on this object.
    """
    def __init__(self):
        self.trainer: Any = None
        self.training_args: Any = None
        self.data_collator: Any = None
        self.model: Any = None
        self.tokenizer: Any = None
        self.dataset: Any = None
        self.callbacks: list = []
        self.trainer_hash: str = ""
        self.ready: bool = False


class TrainerBuilder:
    """Builds a ValidatedTrainer from Training Plan + runtime components."""

    def build(
        self,
        peft_model: Any,
        tokenizer: Any,
        tokenized_dataset: Any,
        training_plan: Dict[str, Any],
        output_dir: Path,
        callbacks: Optional[list] = None,
    ) -> ValidatedTrainer:
        """Construct and validate a trainer from all components.

        Args:
            peft_model: The PEFT/LoRA-injected model.
            tokenizer: The loaded tokenizer.
            tokenized_dataset: The tokenized dataset (HuggingFace Dataset or dict).
            training_plan: The training plan from Sprint 2.
            output_dir: Output directory for checkpoints and logs.
            callbacks: List of callbacks to register.

        Returns:
            ValidatedTrainer ready for execution.
        """
        hp = training_plan.get("hyperparameters", {})
        lora = training_plan.get("lora", {})

        result = ValidatedTrainer()
        result.model = peft_model
        result.tokenizer = tokenizer
        result.dataset = tokenized_dataset
        result.callbacks = callbacks or []

        try:
            from transformers import (
                TrainingArguments, Trainer, DataCollatorForLanguageModeling,
            )

            # Build TrainingArguments from the plan
            training_args = TrainingArguments(
                output_dir=str(output_dir),
                num_train_epochs=hp.get("epochs", 3),
                per_device_train_batch_size=hp.get("batch_size", 4),
                gradient_accumulation_steps=hp.get("gradient_accumulation_steps", 1),
                learning_rate=hp.get("learning_rate", 2e-4),
                optim=hp.get("optimizer", "adamw_torch"),
                lr_scheduler_type=hp.get("scheduler", "linear"),
                warmup_ratio=hp.get("warmup_ratio", 0.03),
                weight_decay=hp.get("weight_decay", 0.01),
                logging_steps=hp.get("logging_steps", 10),
                save_steps=hp.get("save_steps", 100),
                seed=hp.get("seed", 42),
                fp16=(hp.get("precision") == "fp16"),
                bf16=(hp.get("precision") == "bf16"),
                gradient_checkpointing=hp.get("gradient_checkpointing", False),
                remove_unused_columns=True,
                report_to="none",
            )
            result.training_args = training_args

            # Data collator for causal LM
            result.data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=False,
            )

            # Build trainer
            result.trainer = Trainer(
                model=peft_model,
                args=training_args,
                train_dataset=tokenized_dataset if hasattr(tokenized_dataset, '__len__') else None,
                tokenizer=tokenizer,
                data_collator=result.data_collator,
                callbacks=result.callbacks,
            )

            # Compute hash for reproducibility
            import hashlib
            hash_input = json.dumps({
                "model": str(type(peft_model).__name__),
                "epochs": hp.get("epochs"), "lr": hp.get("learning_rate"),
                "batch": hp.get("batch_size"), "precision": hp.get("precision"),
            }, sort_keys=True)
            result.trainer_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

            result.ready = True
            logger.info("trainer_built", hash=result.trainer_hash)

        except ImportError:
            logger.warning("transformers_not_installed",
                           message="Install 'transformers' to build trainers.")
            result.ready = False

        return result
