"""Inference Validator — validates model inference with adapter."""
import time
from typing import Dict, Any
from app.core import get_logger

logger = get_logger("app.validation.inference_validator")


class InferenceValidator:
    def validate(self, model=None, tokenizer=None) -> Dict[str, Any]:
        t0 = time.time()
        results = {"status": "SKIPPED", "reason": ""}

        if model is None or tokenizer is None:
            results["reason"] = "Model or tokenizer not provided. Cannot validate inference."
            return results

        try:
            prompt = "Explain what LoRA fine-tuning is in one sentence."
            inputs = tokenizer(prompt, return_tensors="pt")
            # Move inputs to the same device as the model
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            model.eval()
            with __import__('torch').no_grad():
                outputs = model.generate(**inputs, max_new_tokens=20)
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            latency = round(time.time() - t0, 2)
            results = {
                "status": "PASS", "prompt": prompt, "generated": generated,
                "latency_seconds": latency, "tokens": int(outputs.shape[1]),
            }
            logger.info("inference_validation_pass", latency=latency)
        except Exception as e:
            results["status"] = "FAIL"
            results["error"] = str(e)
            logger.error("inference_validation_failed", error=str(e))

        return results
