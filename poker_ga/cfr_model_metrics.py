"""Standalone report on how "simplifiable" a trained AdvantageNet (see
cfr_networks.py) is, without retraining anything -- prints, for a given
checkpoint:

  1. Total SHAP importance per feature (reusing cfr_networks.
     mean_shap_contributions_for_samples, the same metric cfr_explorer.py
     shows interactively) -- lowest-ranked features are candidates to drop
     from the model's input entirely.
  2. Total pairwise SHAP *interaction* strength per feature (see
     _pairwise_interaction_strength below) -- features with a near-zero
     total across every partner have an effect that's already close to
     purely additive, and are candidates to structurally forbid from
     interacting with anything (e.g. a single extra additive term, rather
     than letting the MLP's fully-connected layers give it a say in every
     other feature's response).
  3. For every feature with *some* real interaction, how concentrated that
     interaction is on just its top 1-2 partners -- a feature whose
     interaction mass is almost entirely with one or two others is a
     candidate to be merged with exactly those partners (e.g. a small
     joint lookup table over just that handful of features) rather than
     left free to interact with the other ~30.

Run with:
    python poker_ga/cfr_model_metrics.py --checkpoint-path cfr_runs/checkpoint_latest

Everything here is computed directly against the checkpoint's own saved
reservoir (cfr_reservoir.py) -- the same realistic, CFR-visitation-weighted
sample of situations cfr_explorer.py uses -- so results reflect the actual
distribution of situations the net was trained on, not an arbitrary or
uniform one.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

import cfr_features
import cfr_networks
import cfr_reservoir

DEFAULT_CHECKPOINT_PATH = os.path.join("cfr_runs", "checkpoint_latest")

# How many reservoir rows to draw the whole report's explain/background
# pools from -- large enough that every masking-pattern stratum (street,
# mainly) is well represented, small enough that sampling from it stays fast.
DEFAULT_POOL_SIZE = 100_000


def _print_importance_table(contributions: list[tuple[str, float]]) -> None:
    total = sum(value for _, value in contributions) or 1.0
    print(f"{'Feature':38} {'Mean |SHAP|':>12} {'% of total':>11} {'Cumulative %':>13}")
    cumulative = 0.0
    for key, value in contributions:
        share = 100.0 * value / total
        cumulative += share
        print(f"{key:38} {value:12.5f} {share:10.1f}% {cumulative:12.1f}%")


def _print_isolation_candidates(
    feature_keys: tuple[str, ...], total_interaction: np.ndarray, weight: np.ndarray, threshold_fraction: float,
) -> list[str]:
    has_data = weight.sum(axis=1) > 0
    order = np.argsort(total_interaction)
    ranked = [i for i in order if has_data[i]]
    if not ranked:
        print("(no features had any jointly-unmasked interaction partner)")
        return []
    cutoff = total_interaction[ranked[-1]] * threshold_fraction if total_interaction[ranked[-1]] > 0 else 0.0
    candidates = []
    print(f"{'Feature':38} {'Total interaction':>18}")
    for i in ranked:
        flag = " <- weak interactions overall" if total_interaction[i] <= cutoff else ""
        print(f"{feature_keys[i]:38} {total_interaction[i]:18.5f}{flag}")
        if total_interaction[i] <= cutoff:
            candidates.append(feature_keys[i])
    return candidates


def _print_merge_candidates(
    feature_keys: tuple[str, ...], interaction: np.ndarray, total_interaction: np.ndarray,
    weight: np.ndarray, share_threshold: float,
) -> list[tuple[str, list[str], float]]:
    candidates = []
    n = len(feature_keys)
    for i in range(n):
        if total_interaction[i] <= 0:
            continue
        partners = [j for j in range(n) if weight[i, j] > 0 and j != i]
        if not partners:
            continue
        partners.sort(key=lambda j: -interaction[i, j])
        top = partners[:2]
        top_sum = sum(interaction[i, j] for j in top)
        share = top_sum / total_interaction[i]
        if share >= share_threshold:
            partner_desc = ", ".join(f"{feature_keys[j]} ({interaction[i, j]:.5f})" for j in top)
            candidates.append((feature_keys[i], [feature_keys[j] for j in top], share))
            print(f"{feature_keys[i]:38} top partner(s): {partner_desc}  ({100 * share:.0f}% of its total interaction)")
    if not candidates:
        print("(no feature had its interaction that concentrated on 1-2 partners)")
    return candidates


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-path", type=str, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--pool-size", type=int, default=DEFAULT_POOL_SIZE, help="Reservoir rows to sample from.")
    parser.add_argument("--importance-sample-size", type=int, default=2_500)
    parser.add_argument("--importance-background-size", type=int, default=500)
    parser.add_argument("--importance-nsamples", type=int, default=250, help="shap.GradientExplainer's own nsamples.")
    parser.add_argument("--interaction-sample-size", type=int, default=80, help="Explain rows per interaction stratum.")
    parser.add_argument("--interaction-background-size", type=int, default=20, help="Baseline rows per interaction stratum.")
    parser.add_argument(
        "--isolation-threshold", type=float, default=0.1,
        help="A feature's total interaction <= this fraction of the strongest feature's total is flagged as a weak-interaction (isolation) candidate.",
    )
    parser.add_argument(
        "--merge-share-threshold", type=float, default=0.7,
        help="A feature whose top 1-2 partners capture at least this fraction of its total interaction is flagged as a merge candidate.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)

    print(f"Loading checkpoint: {args.checkpoint_path}")
    net, config = cfr_networks.load(args.checkpoint_path)
    reservoir = cfr_reservoir.ReservoirBuffer.load(args.checkpoint_path, rng=rng)
    n = min(args.pool_size, len(reservoir))
    idx = rng.choice(len(reservoir), size=n, replace=False)
    pool = reservoir.features[idx]
    print(f"Sampled {n} of {len(reservoir)} reservoir rows across {len(config.feature_keys)} features.\n")

    print("=" * 100)
    print("1) FEATURE IMPORTANCE -- total SHAP contribution per feature (lowest = best removal candidates)")
    print("=" * 100)
    contributions = cfr_networks.mean_shap_contributions_for_samples(
        net, pool, pool, config.feature_keys, rng,
        sample_size=args.importance_sample_size, background_size=args.importance_background_size,
        nsamples=args.importance_nsamples,
    )
    folded = cfr_features.fold_child_contributions(contributions)
    _print_importance_table(folded)
    removal_candidates = [key for key, _ in folded[-5:]]
    print(f"\nWeakest 5 features by total SHAP importance: {', '.join(removal_candidates)}")

    print("\n" + "=" * 100)
    print("2) INTERACTION STRENGTH -- total pairwise SHAP interaction effect per feature (lowest = best isolation candidates)")
    print("=" * 100)
    interaction, weight = cfr_networks.pairwise_interaction_strength(
        net, pool, pool, config.feature_keys, rng,
        sample_size=args.interaction_sample_size, background_size=args.interaction_background_size,
    )
    total_interaction = interaction.sum(axis=1)
    isolation_candidates = _print_isolation_candidates(
        config.feature_keys, total_interaction, weight, args.isolation_threshold,
    )

    print("\n" + "=" * 100)
    print("3) MERGE CANDIDATES -- features whose interaction is concentrated on just 1-2 partners")
    print("=" * 100)
    merge_candidates = _print_merge_candidates(
        config.feature_keys, interaction, total_interaction, weight, args.merge_share_threshold,
    )

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"Candidates to drop entirely (least SHAP importance): {removal_candidates}")
    print(f"Candidates to isolate from all interactions (weakest total interaction): {isolation_candidates}")
    if merge_candidates:
        print("Candidates to merge with their dominant partner(s):")
        for key, partners, share in merge_candidates:
            print(f"  {key} -> {partners} ({100 * share:.0f}% of its interaction)")
    else:
        print("Candidates to merge with their dominant partner(s): none above threshold")


if __name__ == "__main__":
    main()
