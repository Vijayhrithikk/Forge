"""
Tokenizer Preview Engine — estimates tokens and visualizes tokenization.

Provides a simple, educational tokenization preview that is reusable
by Sprint 2 (Training Engine). Uses a word-based heuristic for now;
Sprint 2 will replace with the actual HuggingFace tokenizer.

This engine exists to teach users how tokenization works before
they start training.
"""

from typing import List, Dict, Any, Tuple


class TokenizerPreview:
    """Educational tokenizer preview using heuristic token estimation.

    Simulates tokenization by splitting on whitespace and punctuation
    boundaries. This is NOT the real HuggingFace tokenizer — it is a
    preview tool that produces approximately correct results.

    The interface is designed to be drop-in replaceable with a real
    HuggingFace tokenizer in Sprint 2.
    """

    # Common subword split patterns (approximates BPE behavior)
    PUNCTUATION_SPLITS = {
        ".", ",", "!", "?", ";", ":", "'", '"', "(", ")", "[", "]",
        "{", "}", "<", ">", "/", "\\", "-", "_", "+", "=", "*", "&",
        "^", "%", "$", "#", "@", "~", "`",
    }

    def __init__(self, tokenizer_name: str = "heuristic (word-based)"):
        self._name = tokenizer_name

    @property
    def name(self) -> str:
        return self._name

    def tokenize(self, text: str) -> Dict[str, Any]:
        """Tokenize a text string and return tokens with metadata.

        Args:
            text: The input text to tokenize.

        Returns:
            Dict with original text, tokens list, count, and metadata.
        """
        if not text:
            return {
                "original": text,
                "tokens": [],
                "token_count": 0,
                "char_count": 0,
                "word_count": 0,
                "tokenizer": self._name,
            }

        tokens = self._split_into_tokens(text)

        return {
            "original": text,
            "tokens": tokens,
            "token_count": len(tokens),
            "char_count": len(text),
            "word_count": len(text.split()),
            "tokenizer": self._name,
        }

    def estimate_cost(self, text: str) -> Dict[str, Any]:
        """Estimate the cost of training on this text.

        Provides very rough estimates for GPU memory, adapter size,
        and training time. These are documented approximations and
        should not be treated as exact.

        Args:
            text: The text to estimate for.

        Returns:
            Dict with estimated tokens, cost, and assumptions.
        """
        token_count = len(self._split_into_tokens(text))

        # Rough estimates based on LoRA rank=16 training
        return {
            "token_count": token_count,
            "estimated_adapter_size_kb": round(token_count * 2 / 1024, 2),
            "estimated_training_seconds": round(token_count / 8000, 2),
            "assumptions": [
                "~1.3 tokens per word (English text, Llama tokenizer)",
                "LoRA rank=16, alpha=32",
                "~8000 tokens/second on consumer GPU (RTX 4090)",
                "Estimates are rough; actual values depend on model and hardware",
            ],
        }

    def preview_sample(self, text: str) -> Dict[str, Any]:
        """Generate a full preview for one sample.

        Suitable for display in the Dataset Inspector UI.

        Args:
            text: The sample text to preview.

        Returns:
            Dict with tokenization breakdown and cost estimate.
        """
        tokenization = self.tokenize(text)
        cost = self.estimate_cost(text)

        return {
            **tokenization,
            "estimated_cost": cost,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _split_into_tokens(self, text: str) -> List[str]:
        """Split text into approximate subword tokens.

        This is a simplified BPE-like tokenization:
        1. Split on whitespace
        2. Further split words at punctuation boundaries
        3. Split long words into chunks

        Not exact, but produces reasonable estimates for English text.
        """
        tokens = []
        for word in text.split():
            sub_tokens = self._split_word(word)
            tokens.extend(sub_tokens)
        return tokens

    def _split_word(self, word: str) -> List[str]:
        """Split a single word into approximate subword tokens."""
        if len(word) <= 4:
            return [word]

        result = []
        current = ""

        for char in word:
            if char in self.PUNCTUATION_SPLITS:
                if current:
                    result.extend(self._chunk_long_word(current))
                    current = ""
                result.append(char)
            else:
                current += char

        if current:
            result.extend(self._chunk_long_word(current))

        return result if result else [word]

    def _chunk_long_word(self, word: str, max_chunk: int = 8) -> List[str]:
        """Split long words into smaller chunks (approximates BPE merging)."""
        if len(word) <= max_chunk:
            return [word]
        chunks = []
        for i in range(0, len(word), max_chunk):
            chunks.append(word[i : i + max_chunk])
        return chunks


# Singleton
tokenizer_preview = TokenizerPreview()
