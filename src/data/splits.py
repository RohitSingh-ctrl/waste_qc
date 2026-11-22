"""Leakage-safe, group-aware train/val/test splitting."""
from __future__ import annotations

from typing import Dict, Hashable, List, Mapping, Optional, Sequence

import numpy as np


def grouped_split(groups: Sequence[Hashable], ratios: Mapping[str, float], seed: int = 42,
                  labels: Optional[Sequence[int]] = None) -> Dict[str, List[int]]:
    """Split item indices into train/val/test so that *no group crosses splits*.

    Groups (scene id / near-duplicate cluster) are shuffled and greedily assigned to the split
    that is furthest below its target size. If ``labels`` is given, a group's majority label is
    used to stratify assignment (best effort) so rare classes appear in every split when possible.

    Args:
        groups: Group id per item.
        ratios: e.g. ``{"train": .7, "val": .15, "test": .15}``.
        seed: RNG seed.
        labels: Optional per-item label for best-effort stratification.

    Returns:
        ``{split_name: [item indices]}``.
    """
    rng = np.random.default_rng(seed)
    total = float(sum(ratios.values()))
    names = list(ratios.keys())
    targets = {k: ratios[k] / total for k in names}

    group_items: Dict[Hashable, List[int]] = {}
    for i, g in enumerate(groups):
        group_items.setdefault(g, []).append(i)
    gids = list(group_items.keys())
    rng.shuffle(gids)

    def _major(g: Hashable) -> int:
        if labels is None:
            return 0
        vals, cnt = np.unique([labels[i] for i in group_items[g]], return_counts=True)
        return int(vals[np.argmax(cnt)])

    # process groups label by label so each label is spread across splits
    gids.sort(key=_major)
    counts: Dict[int, Dict[str, int]] = {}
    out: Dict[str, List[int]] = {k: [] for k in names}
    for g in gids:
        lab = _major(g)
        c = counts.setdefault(lab, {k: 0 for k in names})
        tot = sum(c.values()) + 1
        # deficit = target share - current share ; pick the largest deficit
        split = max(names, key=lambda k: targets[k] - c[k] / tot + rng.uniform(0, 1e-6))
        out[split].extend(group_items[g])
        c[split] += len(group_items[g])
    return {k: sorted(v) for k, v in out.items()}


def assert_no_leakage(groups: Sequence[Hashable], split_indices: Mapping[str, Sequence[int]]) -> None:
    """Raise ``AssertionError`` if any group id appears in more than one split."""
    seen: Dict[Hashable, str] = {}
    for split, idxs in split_indices.items():
        for i in idxs:
            g = groups[i]
            if g in seen and seen[g] != split:
                raise AssertionError(f"Leakage: group {g!r} appears in both '{seen[g]}' and '{split}'")
            seen[g] = split


def leakage_report(groups: Sequence[Hashable], split_indices: Mapping[str, Sequence[int]]) -> Dict[str, int]:
    """Return counts of items and unique groups per split plus number of leaked groups."""
    rep: Dict[str, int] = {}
    sets = {}
    for split, idxs in split_indices.items():
        rep[f"{split}_items"] = len(idxs)
        sets[split] = {groups[i] for i in idxs}
        rep[f"{split}_groups"] = len(sets[split])
    names = list(sets)
    leaked = set()
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            leaked |= sets[names[a]] & sets[names[b]]
    rep["leaked_groups"] = len(leaked)
    return rep
