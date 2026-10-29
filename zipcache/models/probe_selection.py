"""Probe-token selection utilities used by ZipCache's salient-token estimator."""

import math

import torch


def select_probe_ids(q_len, probe_ratio=0.10, recent_share=0.50, device=None):
    """Select a fixed-budget mixture of recent and random probe positions.

    Args:
        q_len: Number of prompt/query tokens available during prefill.
        probe_ratio: Approximate fraction of query positions used as probes.
        recent_share: Fraction of the probe budget allocated to the recent tail
            (0.0 means all random probes; 1.0 means all recent probes).
        device: Optional device for the returned index tensor. ``None`` keeps
            the original ZipCache behavior and creates indices on CPU.

    The default intentionally reproduces the original implementation's
    independent rounding: recent_count = ceil(0.05 * q_len), random_count =
    floor(0.05 * q_len). The random portion is sampled with replacement and,
    when recent probes are present, only from positions before the recent tail.
    The total budget is computed once and therefore stays constant as
    ``recent_share`` changes.
    """
    if isinstance(q_len, bool) or not isinstance(q_len, int) or q_len <= 0:
        raise ValueError("q_len must be a positive integer")

    try:
        probe_ratio = float(probe_ratio)
        recent_share = float(recent_share)
    except (TypeError, ValueError) as exc:
        raise ValueError("probe_ratio and recent_share must be numeric") from exc

    if not math.isfinite(probe_ratio) or not 0.0 < probe_ratio <= 1.0:
        raise ValueError("probe_ratio must be in the interval (0, 1]")
    if not math.isfinite(recent_share) or not 0.0 <= recent_share <= 1.0:
        raise ValueError("recent_share must be in the interval [0, 1]")

    half_budget = q_len * (probe_ratio / 2.0)
    total_probe_count = math.floor(half_budget) + math.ceil(half_budget)
    recent_count = math.floor(total_probe_count * recent_share + 0.5)
    random_count = total_probe_count - recent_count
    recent_start = q_len - recent_count

    recent_ids = torch.arange(recent_start, q_len, dtype=torch.long, device=device)
    if random_count:
        # Keep the original with-replacement sampling behavior. Excluding the
        # selected recent tail avoids overlap between the two probe groups.
        random_ids = torch.randint(
            0, recent_start, (random_count,), dtype=torch.long, device=device
        )
    else:
        random_ids = torch.empty(0, dtype=torch.long, device=device)

    return torch.cat((recent_ids, random_ids))
