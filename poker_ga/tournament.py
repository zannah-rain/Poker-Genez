"""Final tournament: re-evaluates a generation with more rounds/hands than
the per-generation GA fitness pass (which is deliberately cheap so evolution
stays fast), producing a higher-confidence ranking. Also exports the top
genomes as human-readable strategy reports plus performance statistics.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import Executor, as_completed
from dataclasses import dataclass, field

import numpy as np
from tqdm import tqdm

from features import FEATURE_GROUPS, FEATURE_SPECS, group_of
from game import GameConfig
from genome import WEIGHT_ALPHABET
from player import Player
from simulate import SimConfig, _executor_scope, run_session


@dataclass
class PlayerStats:
    player_id: int
    label: str = ""
    sessions_played: int = 0
    sessions_won: int = 0
    busts: int = 0
    total_net_chips: float = 0.0
    total_hands_survived: int = 0
    session_results: list = field(default_factory=list)

    @property
    def mean_net_chips(self) -> float:
        return self.total_net_chips / max(self.sessions_played, 1)

    @property
    def bust_rate(self) -> float:
        return self.busts / max(self.sessions_played, 1)

    @property
    def win_rate(self) -> float:
        return self.sessions_won / max(self.sessions_played, 1)

    @property
    def mean_hands_survived(self) -> float:
        return self.total_hands_survived / max(self.sessions_played, 1)

    def std_net_chips(self) -> float:
        return float(np.std(self.session_results)) if self.session_results else 0.0

    def bb_per_100(self, big_blind: float) -> float:
        if self.total_hands_survived == 0:
            return 0.0
        return (self.total_net_chips / big_blind) / self.total_hands_survived * 100.0


def _play_one_tournament_table(
    table_players: list[Player],
    game_config: GameConfig,
    backfill_pool: list[Player],
    rng: np.random.Generator,
) -> dict:
    """One table's worth of work for run_final_tournament. Deliberately a
    plain module-level function (not a closure/lambda) so it can be pickled
    and sent to a worker process."""
    return run_session(table_players, game_config, rng, backfill_pool=backfill_pool)


def run_final_tournament(
    players: list[Player],
    game_config: GameConfig,
    sim_config: SimConfig,
    rng: np.random.Generator,
    show_progress: bool = True,
    num_workers: int = 1,
    executor: Executor | None = None,
) -> dict[int, PlayerStats]:
    """Runs many rounds of random re-seating (more than a normal GA
    generation) and accumulates detailed per-player statistics, using
    simulate.run_session (which refills busted seats with a fresh player
    from the whole population rather than ending the session early).

    See simulate.run_generation's docstring for what `num_workers`/`executor`
    do and why a parallel run (num_workers > 1) doesn't reproduce the exact
    same hands as a sequential one (num_workers=1, the default) for the same
    seed -- the same trade-off applies here, table matches being the
    parallel unit for the same reasons."""
    stats = {p.player_id: PlayerStats(player_id=p.player_id, label=p.label) for p in players}

    # This is usually the single slowest step of a run (many more rounds
    # than a per-generation fitness pass, for a low-variance final ranking),
    # so a progress bar over its fixed table count is worth the noise.
    tables_per_round = math.ceil(len(players) / sim_config.table_size)
    total_tables = sim_config.rounds_per_generation * tables_per_round
    progress = tqdm(total=total_tables, desc="final tournament", unit="table", disable=not show_progress)

    def _accumulate(result: dict) -> None:
        for pid, net in result["net"].items():
            s = stats[pid]
            s.sessions_played += 1
            s.total_net_chips += net
            s.total_hands_survived += result["hands_survived"][pid]
            s.session_results.append(net)
            if result["busted"][pid]:
                s.busts += 1
            if result["winner_id"] == pid:
                s.sessions_won += 1
        progress.update(1)

    if num_workers <= 1:
        for _ in range(sim_config.rounds_per_generation):
            order = rng.permutation(len(players))
            shuffled = [players[i] for i in order]
            for start in range(0, len(shuffled), sim_config.table_size):
                table = shuffled[start : start + sim_config.table_size]
                if len(table) < 2:
                    progress.update(1)
                    continue
                result = run_session(table, game_config, rng, backfill_pool=players)
                _accumulate(result)
    else:
        tables = []
        for _ in range(sim_config.rounds_per_generation):
            order = rng.permutation(len(players))
            shuffled = [players[i] for i in order]
            for start in range(0, len(shuffled), sim_config.table_size):
                table = shuffled[start : start + sim_config.table_size]
                if len(table) >= 2:
                    tables.append(table)
                else:
                    progress.update(1)

        table_rngs = rng.spawn(len(tables))
        with _executor_scope(executor, num_workers) as pool:
            futures = [
                pool.submit(_play_one_tournament_table, table, game_config, players, table_rng)
                for table, table_rng in zip(tables, table_rngs)
            ]
            for future in as_completed(futures):
                _accumulate(future.result())

    progress.close()
    return stats


def rank_players(
    players: list[Player],
    stats: dict[int, PlayerStats],
    sparsity_penalty: float = 0.0,
) -> list[Player]:
    """Ranks by mean net chips, minus `sparsity_penalty` per nonzero feature
    weight -- the same complexity penalty applied during evolution (see
    main.py's --sparsity-penalty), so the genomes exported as "best" reflect
    the same simplicity preference used to select for them, not just raw
    chip performance."""
    def score(p: Player) -> float:
        return stats[p.player_id].mean_net_chips - sparsity_penalty * p.genome.nonzero_weight_count()

    return sorted(players, key=score, reverse=True)


def _children_by_parent() -> dict[str, dict[int, "FeatureSpec"]]:
    """Maps parent feature key -> {value_table index -> its linked indicator spec}."""
    out: dict = {}
    for spec in FEATURE_SPECS:
        if spec.linked_to is not None:
            out.setdefault(spec.linked_to, {})[spec.linked_value_index] = spec
    return out


def _axis_combined(weight_by_key: dict, spec, children: dict, index: int, point: float) -> float:
    """Combined (general + specific) raw (pre-clip) contribution for one
    value-table row, on one axis (V or L)."""
    general = weight_by_key[spec.key] * point
    child = children.get(index)
    specific = weight_by_key[child.key] if child is not None else 0.0
    return general + specific


def describe_genome(player: Player, stats: PlayerStats, game_config: GameConfig, rank: int) -> str:
    """Renders a genome's weights as a human-readable strategy report."""
    g = player.genome
    v_weight_by_key = {spec.key: float(w) for spec, w in zip(FEATURE_SPECS, g.weights_v)}
    l_weight_by_key = {spec.key: float(w) for spec, w in zip(FEATURE_SPECS, g.weights_l)}
    children_by_parent = _children_by_parent()

    lines = []
    name = player.label or f"Player {player.player_id}"
    lines.append(f"# Strategy Report: {name} (final rank #{rank})")
    lines.append("")
    lines.append("## Performance (final tournament)")
    lines.append(f"- Sessions played: {stats.sessions_played}")
    lines.append(f"- Mean net chips / session: {stats.mean_net_chips:+.1f} (stddev {stats.std_net_chips():.1f})")
    lines.append(f"- Win rate (finished a session on top): {stats.win_rate:.1%}")
    lines.append(f"- Bust rate: {stats.bust_rate:.1%}")
    lines.append(f"- Mean hands survived per session: {stats.mean_hands_survived:.1f}")
    lines.append(f"- Win rate: {stats.bb_per_100(game_config.big_blind):+.2f} bb/100 hands")
    lines.append(f"- Best single session: {max(stats.session_results, default=0.0):+.1f}")
    lines.append(f"- Worst single session: {min(stats.session_results, default=0.0):+.1f}")
    lines.append("")

    lines.append("## How this genome decides")
    lines.append(
        "Every decision computes two numbers from the features below "
        f"({len(FEATURE_SPECS)} total, organized by theme -- see the group breakdowns for "
        "how the count gets that high while staying readable):"
    )
    lines.append("")
    lines.append(
        "- **V (showdown value)** ~ roughly this player's equity against the range that "
        "continues, calibrated to read as a 0-100 percentage."
    )
    lines.append(
        "- **L (leverage)** ~ roughly how much of villain's range folds to a bet here (fold "
        "equity, shaped by blockers, initiative, board texture, position, SPR), also "
        "calibrated to 0-100."
    )
    lines.append("")
    lines.append(
        "Each is a linear, clipped sum of `weight x feature value` across all features "
        "below -- one weight per feature feeds V, a separate weight feeds L, so the two axes "
        "can capture near-independent signals (e.g. a nut flush blocker is almost pure L "
        "with ~0 V; a set on a dry board is almost pure V with low L). They combine "
        "non-convexly into one action score:"
    )
    lines.append("")
    lines.append("`A = max(V - theta_value,  L - theta_bluff - kappa x V)`")
    lines.append("")
    lines.append("- `A > 0` -> bet/raise, sized at `(A / 100) x pot` (e.g. A=50 -> half-pot, A=100 -> pot-sized)")
    lines.append("- `elif V > theta_call` -> call/check")
    lines.append("- `else` -> fold/check")
    lines.append("")
    lines.append(f"- theta_value (value needed to raise for value): {g.theta_value:.1f}")
    lines.append(f"- theta_bluff (leverage needed to raise as a bluff): {g.theta_bluff:.1f}")
    lines.append(f"- theta_call (value needed to call rather than fold): {g.theta_call:.1f}")
    lines.append(
        f"- bias_v / bias_l (baseline V/L before any feature is considered): "
        f"{g.bias_v:.1f} / {g.bias_l:.1f}"
    )
    lines.append(f"- kappa (how much V suppresses the bluff term): {g.kappa:.3f}")
    lines.append(
        f"- Decision noise (random amount added to V and L independently each decision): "
        f"{g.noise_std:.2f} points. Higher values mean more unpredictable/bluffy play; near "
        "zero means fully deterministic."
    )
    alphabet_str = ", ".join(str(float(x)) for x in WEIGHT_ALPHABET)
    lines.append(
        f"- Nonzero feature weights: {g.nonzero_weight_count()} of "
        f"{len(FEATURE_SPECS) * 2} possible (weights are quantized to [{alphabet_str}], "
        "and fitness penalizes nonzero ones, so most should land on exactly 0 -- the tables "
        "below are the actual cheat sheet, not an approximation of one)."
    )
    lines.append("")
    lines.append(
        "The per-feature weights below are on the pre-clip (raw) scale, not directly in "
        "V/L percentage points -- they combine additively with every other active feature's "
        "weight before the final linear 0-100 calibration (offset by 50, then clipped) is "
        "applied, so sign and relative magnitude are meaningful, but a single weight doesn't "
        "translate to \"N points of V\" on its own."
    )
    lines.append("")

    active_spots = g.active_gto_spots()
    lines.append("## GTO Chart Overrides")
    if active_spots:
        lines.append(
            "This genome memorizes an exact chart for the spot(s) below -- whenever the "
            "situation matches, it plays straight off the chart instead of the V/L rule above "
            "(checked in the order listed; the first matching spot wins). Anything not listed "
            "in a spot's ranges is a fold."
        )
        lines.append("")
        for spot in active_spots:
            lines.append(f"### {spot.label}")
            lines.append(f"- **When this applies:** {spot.matcher.describe()}")
            lines.append("")
            lines.append("| action | range |")
            lines.append("|---|---|")
            for action_token, range_str in spot.action_ranges:
                lines.append(f"| {action_token} | {range_str} |")
            lines.append(f"| *(anything else)* | {spot.default_action} |")
            lines.append("")
    else:
        lines.append("This genome doesn't trust any memorized chart -- every decision goes through the V/L rule above.")
        lines.append("")

    parents = [s for s in FEATURE_SPECS if s.kind != "boolean"]
    standalone_booleans = [s for s in FEATURE_SPECS if s.kind == "boolean" and s.linked_to is None]

    def parent_influence(spec) -> float:
        children = children_by_parent.get(spec.key, {})
        return max(
            max(
                abs(_axis_combined(v_weight_by_key, spec, children, i, point)),
                abs(_axis_combined(l_weight_by_key, spec, children, i, point)),
            )
            for i, (point, _label) in enumerate(spec.value_table)
        )

    parents_sorted = sorted(parents, key=parent_influence, reverse=True)
    standalone_sorted = sorted(
        standalone_booleans,
        key=lambda s: -max(abs(v_weight_by_key[s.key]), abs(l_weight_by_key[s.key])),
    )

    lines.append("## Feature Groups")
    lines.append(
        "Features are grouped by theme so related ones sit together. Within each group: "
        "boolean (yes/no) features first as a compact table (V weight, L weight), most "
        "influential first; then one breakdown table per multi-value feature, combining each "
        "value's general (linear) weight and its own exact indicator feature's weight into "
        "one raw (pre-clip) number per axis."
    )
    lines.append("")

    for group in FEATURE_GROUPS:
        group_booleans = [s for s in standalone_sorted if group_of(s) == group]
        group_parents = [s for s in parents_sorted if group_of(s) == group]
        if not group_booleans and not group_parents:
            continue

        lines.append(f"### {group}")
        lines.append("")

        if group_booleans:
            lines.append("| feature | V weight | L weight |")
            lines.append("|---|---|---|")
            for spec in group_booleans:
                vw = v_weight_by_key[spec.key]
                lw = l_weight_by_key[spec.key]
                lines.append(f"| {spec.label} | {vw:+.3f} | {lw:+.3f} |")
            lines.append("")

        for spec in group_parents:
            vw = v_weight_by_key[spec.key]
            lw = l_weight_by_key[spec.key]
            children = children_by_parent.get(spec.key, {})
            note = (
                " (continuous -- bucketed to the nearest of 5 representative points)"
                if spec.kind == "continuous" else ""
            )
            lines.append(f"#### {spec.label} — general weights V {vw:+.3f} / L {lw:+.3f}{note}")
            lines.append("")
            lines.append("| value | normalized value | V weight | L weight |")
            lines.append("|---|---|---|---|")
            for i, (point, label) in enumerate(spec.value_table):
                v_combined = _axis_combined(v_weight_by_key, spec, children, i, point)
                l_combined = _axis_combined(l_weight_by_key, spec, children, i, point)
                lines.append(f"| {label} | {point:.3f} | {v_combined:+.3f} | {l_combined:+.3f} |")
            lines.append("")

    lines.append("## Feature Reference")
    lines.append(
        "Precise definition of every generalized/standalone feature above, grouped the same "
        "way. Exact per-value indicator features aren't re-listed individually here -- their "
        "meaning is exactly the table row they're folded into above (the \"specific "
        "adjustment\" column)."
    )
    lines.append("")
    for group in FEATURE_GROUPS:
        group_booleans = [s for s in standalone_sorted if group_of(s) == group]
        group_parents = [s for s in parents_sorted if group_of(s) == group]
        if not group_booleans and not group_parents:
            continue
        lines.append(f"### {group}")
        lines.append("")
        for spec in group_booleans:
            lines.append(f"- **{spec.label}** — {spec.description}")
        for spec in group_parents:
            if spec.kind == "categorical":
                note = (
                    " Each value above also has its own exact indicator feature for a "
                    "non-linear, value-specific adjustment (the \"specific adjustment\" "
                    "column above)."
                )
            else:
                note = (
                    " Each representative value above also has an approximate \"nearest "
                    "value\" indicator feature for a non-linear adjustment (the \"specific "
                    "adjustment\" column above)."
                )
            lines.append(f"- **{spec.label}** — {spec.description}{note}")
        lines.append("")

    return "\n".join(lines)


def export_top_n(
    ranked_players: list[Player],
    stats: dict[int, PlayerStats],
    game_config: GameConfig,
    n: int,
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    summary_lines = ["# Final Tournament Leaderboard", ""]
    summary_lines.append(
        "| rank | player | mean net chips/session | win rate | bust rate | bb/100 | nonzero wts |"
    )
    summary_lines.append("|---|---|---|---|---|---|---|")
    for rank, p in enumerate(ranked_players[:n], start=1):
        s = stats[p.player_id]
        name = p.label or f"Player {p.player_id}"
        summary_lines.append(
            f"| {rank} | {name} | {s.mean_net_chips:+.1f} | {s.win_rate:.1%} "
            f"| {s.bust_rate:.1%} | {s.bb_per_100(game_config.big_blind):+.2f} "
            f"| {p.genome.nonzero_weight_count()} |"
        )
    with open(os.path.join(out_dir, "leaderboard.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    for rank, p in enumerate(ranked_players[:n], start=1):
        s = stats[p.player_id]
        report = describe_genome(p, s, game_config, rank)
        base = f"rank{rank:02d}_player{p.player_id}"
        with open(os.path.join(out_dir, f"{base}_strategy.md"), "w", encoding="utf-8") as f:
            f.write(report)
        p.genome.save(os.path.join(out_dir, f"{base}_genome.json"))
