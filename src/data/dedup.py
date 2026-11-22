"""Near-duplicate removal via perceptual hashing (``imagehash``) with a numpy fallback.

Duplicates are grouped with union-find on pairwise Hamming distance. Each group later
becomes one *split group* so duplicates can never straddle train/val/test (leakage-safe).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from PIL import Image

from src.utils import get_logger

logger = get_logger(__name__)

try:  # optional dependency
    import imagehash

    _HAS_IMAGEHASH = True
except ImportError:  # pragma: no cover
    _HAS_IMAGEHASH = False


def perceptual_hash(path: Path | str, hash_size: int = 8, color_bits: bool = True) -> np.ndarray:
    """Return a flat boolean hash vector for an image.

    The luminance pHash (``hash_size**2`` bits) is concatenated with a coarse *colour* signature
    (3 channels x 4x4 thumbnail, thresholded at the global mean) because a grayscale-only pHash
    collides for different-coloured items of similar shape (e.g. a glass vs a plastic item).
    """
    with Image.open(path) as im:
        im = im.convert("RGB")
        if _HAS_IMAGEHASH:
            bits = imagehash.phash(im, hash_size=hash_size).hash.flatten().astype(bool)
        else:  # fallback: difference hash on a (hash_size+1, hash_size) grayscale thumbnail
            g = np.asarray(im.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS), dtype=np.float32)
            bits = (g[:, 1:] > g[:, :-1]).flatten()
        if color_bits:
            t = np.asarray(im.resize((4, 4), Image.BOX), dtype=np.float32)  # (4, 4, 3)
            bits = np.concatenate([bits, (t > t.mean()).flatten()])
        return bits


def hamming_matrix(hashes: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distances for an ``(N, bits)`` boolean matrix."""
    h = hashes.astype(np.int16)
    # d(a,b) = |a| + |b| - 2 a.b  for binary vectors
    ones = h.sum(axis=1)
    return ones[:, None] + ones[None, :] - 2 * (h @ h.T)


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)  # smallest index is the representative


@dataclass
class DedupResult:
    """Outcome of near-duplicate detection."""

    kept: List[int] = field(default_factory=list)            # indices to keep (one per cluster)
    removed: List[int] = field(default_factory=list)         # indices dropped
    cluster_of: List[int] = field(default_factory=list)      # cluster id (representative index) per item
    clusters: Dict[int, List[int]] = field(default_factory=dict)


def find_near_duplicates(paths: Sequence[Path | str], max_distance: int = 4, hash_size: int = 8) -> DedupResult:
    """Cluster near-duplicate images.

    Args:
        paths: Image file paths.
        max_distance: Items with Hamming distance ``<=`` this are duplicates.
        hash_size: pHash size (bits = hash_size**2).

    Returns:
        :class:`DedupResult`; the first member of each cluster is kept.
    """
    n = len(paths)
    if n == 0:
        return DedupResult()
    hashes = np.stack([perceptual_hash(p, hash_size) for p in paths])
    dist = hamming_matrix(hashes)
    uf = _UnionFind(n)
    ii, jj = np.where(np.triu(dist <= max_distance, k=1))
    for a, b in zip(ii.tolist(), jj.tolist()):
        uf.union(a, b)
    cluster_of = [uf.find(i) for i in range(n)]
    clusters: Dict[int, List[int]] = {}
    for i, c in enumerate(cluster_of):
        clusters.setdefault(c, []).append(i)
    kept = sorted(clusters.keys())
    removed = sorted(i for i in range(n) if i not in set(kept))
    logger.info("Dedup: %d images -> %d unique (%d near-duplicates removed)", n, len(kept), len(removed))
    return DedupResult(kept=kept, removed=removed, cluster_of=cluster_of, clusters=clusters)
