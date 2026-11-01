"""
Perceptual Frame Hashing & Visual Similarity Engine.

Provides deterministic local-first perceptual image hashing for video frame deduplication,
slide transition detection, and visual similarity comparison:
- dHash (Difference Hash): 64-bit gradient-based difference fingerprint
- aHash (Average Hash): 64-bit luminance mean-comparison fingerprint
- pHash (Perceptual Hash): 64-bit frequency-domain Discrete Cosine Transform (DCT) fingerprint

Truth Classifications:
- Algorithm implementation: REAL (deterministic mathematical visual computation)
- Semantic understanding: HEURISTIC (perceptual similarity, NOT semantic comprehension)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image


class PerceptualHasher:
    """
    Deterministic perceptual hasher for video frame comparison and deduplication.
    Runs entirely on CPU using Pillow and NumPy without external network dependencies.
    """

    def __init__(self, hash_size: int = 8):
        self.hash_size = hash_size

    @staticmethod
    def get_classification() -> str:
        """Returns truth classification of the perceptual hashing subsystem."""
        return "REAL"

    @staticmethod
    def get_semantic_classification() -> str:
        """Explicitly acknowledges that perceptual hash measures visual similarity, not semantic understanding."""
        return "HEURISTIC"

    def _load_image(self, image: Union[Image.Image, np.ndarray, Path, str]) -> Image.Image:
        """Coerces diverse image representations into a Pillow RGB Image."""
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        elif isinstance(image, np.ndarray):
            # Check shape and dtype
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            if len(image.shape) == 2:
                return Image.fromarray(image, mode="L").convert("RGB")
            elif len(image.shape) == 3:
                if image.shape[2] == 4:
                    return Image.fromarray(image, mode="RGBA").convert("RGB")
                elif image.shape[2] == 3:
                    return Image.fromarray(image, mode="RGB")
            raise ValueError(f"Unsupported numpy image array shape: {image.shape}")
        elif isinstance(image, (str, Path)):
            p = Path(image)
            if not p.exists():
                raise FileNotFoundError(f"Image file does not exist: {p}")
            return Image.open(p).convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

    def compute_dhash(
        self,
        image: Union[Image.Image, np.ndarray, Path, str],
        hash_size: Optional[int] = None,
    ) -> str:
        """
        Computes difference hash (dHash).
        Resizes to (hash_size + 1, hash_size), converts to grayscale,
        and computes adjacent horizontal pixel gradient comparisons.
        Returns a 16-character hexadecimal string representing the 64-bit hash.
        """
        size = hash_size or self.hash_size
        img = self._load_image(image).convert("L")
        # Resize to (width = size + 1, height = size)
        resized = img.resize((size + 1, size), Image.Resampling.BILINEAR)
        pixels = np.array(resized, dtype=np.float32)

        # Compare adjacent columns: pixel[col+1] > pixel[col]
        diff = pixels[:, 1:] > pixels[:, :-1]
        bits = diff.flatten()

        # Convert boolean array to hex string
        return self._bits_to_hex(bits)

    def compute_ahash(
        self,
        image: Union[Image.Image, np.ndarray, Path, str],
        hash_size: Optional[int] = None,
    ) -> str:
        """
        Computes average hash (aHash).
        Resizes to (hash_size, hash_size), converts to grayscale,
        and compares each pixel against the mean pixel luminance.
        Returns a 16-character hexadecimal string.
        """
        size = hash_size or self.hash_size
        img = self._load_image(image).convert("L")
        resized = img.resize((size, size), Image.Resampling.BILINEAR)
        pixels = np.array(resized, dtype=np.float32)

        avg = pixels.mean()
        diff = pixels > avg
        bits = diff.flatten()

        return self._bits_to_hex(bits)

    def compute_phash(
        self,
        image: Union[Image.Image, np.ndarray, Path, str],
        hash_size: Optional[int] = None,
        highfreq_factor: int = 4,
    ) -> str:
        """
        Computes perceptual hash (pHash) via Discrete Cosine Transform (DCT).
        Resizes to (size * factor, size * factor), computes 2D DCT in frequency domain,
        extracts the top-left low-frequency quadrant, and compares against median.
        Resistant to minor scaling, gamma shifts, and compression artifacts.
        Returns a 16-character hexadecimal string.
        """
        size = hash_size or self.hash_size
        img_size = size * highfreq_factor
        img = self._load_image(image).convert("L")
        resized = img.resize((img_size, img_size), Image.Resampling.BILINEAR)
        pixels = np.array(resized, dtype=np.float32)

        # 2D Discrete Cosine Transform (Type-II) via NumPy matrix multiplication
        dct = self._dct_2d(pixels)

        # Extract top-left low-frequency coefficients excluding DC component at [0, 0]
        low_freq = dct[:size, :size]
        med = np.median(low_freq[1:, 1:])  # exclude DC from median calculation

        diff = low_freq > med
        bits = diff.flatten()

        return self._bits_to_hex(bits)

    @staticmethod
    def _dct_2d(a: np.ndarray) -> np.ndarray:
        """Fast 2D DCT-II computation using precomputed orthonormal cosine basis."""
        n, m = a.shape
        # Compute 1D DCT matrix for rows
        i_idx = np.arange(n).reshape(-1, 1)
        j_idx = np.arange(n).reshape(1, -1)
        c_n = np.sqrt(2.0 / n) * np.cos((np.pi * (2 * j_idx + 1) * i_idx) / (2.0 * n))
        c_n[0, :] = 1.0 / np.sqrt(n)

        # If square, reuse basis; otherwise compute column basis
        if n == m:
            c_m = c_n
        else:
            i_m = np.arange(m).reshape(-1, 1)
            j_m = np.arange(m).reshape(1, -1)
            c_m = np.sqrt(2.0 / m) * np.cos((np.pi * (2 * j_m + 1) * i_m) / (2.0 * m))
            c_m[0, :] = 1.0 / np.sqrt(m)

        return c_n @ a @ c_m.T

    @staticmethod
    def _bits_to_hex(bits: np.ndarray) -> str:
        """Packs a boolean/binary array into a hex string."""
        # Pad to multiple of 8 if necessary
        pad_len = (8 - (len(bits) % 8)) % 8
        if pad_len > 0:
            bits = np.pad(bits, (0, pad_len), constant_values=False)

        byte_arr = np.packbits(bits)
        return bytes(byte_arr).hex()

    @staticmethod
    def hamming_distance(hash1: str, hash2: str) -> int:
        """
        Calculates the bitwise Hamming distance between two hex hashes.
        Returns the number of differing bits (0 to 64 for 64-bit hashes).
        """
        if not hash1 or not hash2:
            return 64
        try:
            val1 = int(hash1, 16)
            val2 = int(hash2, 16)
            xor_val = val1 ^ val2
            return bin(xor_val).count("1")
        except ValueError:
            return 64

    @classmethod
    def perceptual_similarity(cls, hash1: str, hash2: str, max_bits: int = 64) -> float:
        """
        Computes normalized perceptual similarity in [0.0, 1.0].
        1.0 means identical perceptual hash, 0.0 means completely orthogonal.
        """
        dist = cls.hamming_distance(hash1, hash2)
        return max(0.0, min(1.0, 1.0 - (dist / float(max_bits))))

    @classmethod
    def is_duplicate_frame(
        cls,
        hash1: Union[str, Image.Image, np.ndarray, Path],
        hash2: Union[str, Image.Image, np.ndarray, Path],
        max_hamming_distance: int = 6,
    ) -> bool:
        """
        Determines whether two frames are perceptual duplicates.
        Accepts either 16-character hex hash strings or Image/array objects directly.
        Default threshold: <= 6 bits distance (~90.6% visual similarity).
        """
        hasher = cls()
        h1_str = hash1 if (isinstance(hash1, str) and len(hash1) == 16) else hasher.compute_dhash(hash1)
        h2_str = hash2 if (isinstance(hash2, str) and len(hash2) == 16) else hasher.compute_dhash(hash2)
        dist = cls.hamming_distance(h1_str, h2_str)
        return dist <= max_hamming_distance


# Convenience module-level functions
_default_hasher = PerceptualHasher()
compute_dhash = _default_hasher.compute_dhash
compute_ahash = _default_hasher.compute_ahash
compute_phash = _default_hasher.compute_phash
hamming_distance = PerceptualHasher.hamming_distance
perceptual_similarity = PerceptualHasher.perceptual_similarity
is_duplicate_frame = PerceptualHasher.is_duplicate_frame
