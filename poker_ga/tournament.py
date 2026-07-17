"""Final tournament: re-evaluates a generation with more rounds/hands than
the per-generation GA fitness pass (which is deliberately cheap so evolution
stays fast), producing a higher-confidence ranking. Also exports the top
genomes as human-readable strategy reports plus performance statistics.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from features import FEATURE_GROUPS, FEATURE_SPECS, group_of
from game import GameConfig, SeatState, play_hand
from player import Player
from simulate import SimConfig


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


def run_session_detailed(players: list[Player], game_config: GameConfig, rng: np.random.Generator) -> dict:
    """Like simulate.run_session, but also tracks per-player hands survived,
    who busted, and who ended the session on top (the "winner")."""
    seats = [SeatState(player=p, stack=game_config.starting_stack) for p in players]
    starting_ids = [p.player_id for p in players]
    initial_seat_count = len(seats)
    hands_survived = {pid: 0 for pid in starting_ids}
    busted = {pid: False for pid in starting_ids}

    button_idx = 0
    hands_played = 0
    while (
        len(seats) > 1
        and hands_played < game_config.max_hands_per_session
        and (initial_seat_count - len(seats)) < game_config.busts_before_table_ends
    ):
        alive_before = {s.player.player_id for s in seats}
        play_hand(seats, button_idx % len(seats), game_config, rng)
        hands_played += 1
        for pid in alive_before:
            hands_survived[pid] += 1
        seats = [s for s in seats if s.stack > 1e-9]
        remaining = {s.player.player_id for s in seats}
        for pid in alive_before - remaining:
            busted[pid] = True
        if seats:
            button_idx = (button_idx + 1) % len(seats)

    final_stack = {s.player.player_id: s.stack for s in seats}
    net = {pid: final_stack.get(pid, 0.0) - game_config.starting_stack for pid in starting_ids}
    winner_id = max(seats, key=lambda s: s.stack).player.player_id if seats else None

    return {
        "net": net,
        "hands_survived": hands_survived,
        "busted": busted,
        "winner_id": winner_id,
    }


def run_final_tournament(
    players: list[Player],
    game_config: GameConfig,
    sim_config: SimConfig,
    rng: np.random.Generator,
) -> dict[int, PlayerStats]:
    """Runs many rounds of random re-seating (more than a normal GA
    generation) and accumulates detailed per-player statistics."""
    stats = {p.player_id: PlayerStats(player_id=p.player_id, label=p.label) for p in players}

    for _ in range(sim_config.rounds_per_generation):
        order = rng.permutation(len(players))
        shuffled = [players[i] for i in order]
        for start in range(0, len(shuffled), sim_config.table_size):
            table = shuffled[start : start + sim_config.table_size]
            if len(table) < 2:
                continue
            result = run_session_detailed(table, game_config, rng)
            for p in table:
                pid = p.player_id
                s = stats[pid]
                s.sessions_played += 1
                s.total_net_chips += result["net"][pid]
                s.total_hands_survived += result["hands_survived"][pid]
                s.session_results.append(result["net"][pid])
                if result["busted"][pid]:
                    s.busts += 1
                if result["winner_id"] == pid:
                    s.sessions_won += 1

    return stats


def rank_players(players: list[Player], stats: dict[int, PlayerStats]) -> list[Player]:
    return sorted(players, key=lambda p: stats[p.player_id].mean_net_chips, reverse=True)


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
    lines.append(f"- kappa (how much V suppresses the bluff term): {g.kappa:.3f}")
    lines.append(
        f"- Decision noise (random amount added to V and L independently each decision): "
        f"{g.noise_std:.2f} points. Higher values mean more unpredictable/bluffy play; near "
        "zero means fully deterministic."
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
    summary_lines.append("| rank | player | mean net chips/session | win rate | bust rate | bb/100 |")
    summary_lines.append("|---|---|---|---|---|---|")
    for rank, p in enumerate(ranked_players[:n], start=1):
        s = stats[p.player_id]
        name = p.label or f"Player {p.player_id}"
        summary_lines.append(
            f"| {rank} | {name} | {s.mean_net_chips:+.1f} | {s.win_rate:.1%} "
            f"| {s.bust_rate:.1%} | {s.bb_per_100(game_config.big_blind):+.2f} |"
        )
    with open(os.path.join(out_dir, "leaderboard.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    for rank, p in enumerate(ranked_players[:n], start=1):
        s = stats[p.player_id]
        report = describe_genome(p, s, game_config, rank)
        base = f"rank{rank:02d}_player{p.player_id}"
        with open(os.path.join(out_dir, f"{base}_strategy.md"), "w", encoding="utf-8") as f:
            f.write(report)
        p.genome.save(os.path.join(out_dir, f"{base}_genome.npy"))
