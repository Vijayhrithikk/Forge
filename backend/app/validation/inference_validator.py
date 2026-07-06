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
            prompt = "Hello, how are you?"
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(**inputs, max_new_tokens=10)
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            latency = round(time.time() - t0, 2)
            results = {
                "status": "PASS", "prompt": prompt, "generated": generated,
                "latency_seconds": latency, "tokens": outputs.shape[1],
            }
            logger.info("inference_validation_pass", latency=latency)
        except Exception as e:
            results["status"] = "FAIL"
            results["error"] = str(e)
            logger.error("inference_validation_failed", error=str(e))

        return results
