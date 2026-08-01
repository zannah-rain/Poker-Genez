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

import strategy
from features import group_of
from game import GameConfig
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


# Sort priority for the Strategy Rules section: broad "what spot am I in"
# context features print first, "what just happened this street" trigger
# features print last, so rules sharing the same context but differing only
# on a trigger (e.g. facing a bet or not) end up as adjacent lines --
# readable as "in this spot: checked to -> raise; raised to -> fold."
# Anything not listed falls in the middle. This governs *report order*
# only, not evaluation order -- see the note printed above the section.
_REPORT_GROUP_PRIORITY = {
    "Table & Game State Features": 0,
    "Hole Card Characteristics": 1,
    "Board / Flop Characteristics": 2,
    "Made Hand Features": 3,
    "Draw Features": 4,
    "Stack & Pot Features": 5,
    "Opponent Tendency Features": 6,
    "Betting Behaviour Features": 7,
}
_UNGROUPED_CONDITION_PRIORITY = 50
_WILDCARD_CONDITION_KEY = (99, "", 99)


def _condition_phrase(spec, bucket_index: int, num_buckets: int, thresholds) -> str:
    label = strategy.describe_bucket(spec, bucket_index, num_buckets, thresholds)
    return label if spec.kind == "boolean" else f"{spec.label} = {label}"


def _rule_sort_key(condition_features, condition_buckets) -> tuple:
    keyed = []
    for fi, bucket in zip(condition_features, condition_buckets):
        if fi == strategy.WILDCARD:
            keyed.append(_WILDCARD_CONDITION_KEY)
            continue
        spec = strategy.TOP_LEVEL_FEATURES[int(fi)]
        priority = _REPORT_GROUP_PRIORITY.get(group_of(spec), _UNGROUPED_CONDITION_PRIORITY)
        keyed.append((priority, spec.key, int(bucket)))
    return tuple(sorted(keyed))


def describe_genome(player: Player, stats: PlayerStats, game_config: GameConfig, rank: int) -> str:
    """Renders a genome's rule-based strategy as a human-readable report."""
    g = player.genome

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

    total_conditions = strategy.NUM_RULES * strategy.CONDITIONS_PER_RULE
    raise_pct = strategy.RAISE_SIZE_ALPHABET[g.raise_size_idx] * 100
    lines.append("## How this genome decides")
    lines.append(
        "Every decision buckets a small set of features into 2-3 groups (see Feature Buckets "
        "below for this genome's own cutoffs), then checks a fixed list of rules -- first full "
        "match wins -- and plays that rule's action. No match at all defaults to Fold, the "
        "same way a real range chart's blank squares are a fold."
    )
    lines.append("")
    lines.append(f"- **Raise size:** always raises to {raise_pct:.0f}% of pot (one shared size for every Raise rule).")
    lines.append(
        f"- **Decision noise:** {g.bucket_noise_std:.3f} -- small randomness applied to a feature's "
        "reading before it's bucketed, so a hand right at a threshold occasionally falls on "
        "either side (a cheap, natural mixed strategy at the margins). Near zero means fully "
        "deterministic."
    )
    lines.append(
        f"- **Active rule conditions:** {g.nonzero_weight_count()} of {total_conditions} possible -- "
        "how many (feature, threshold) facts this genome's strategy actually depends on; "
        "fitness penalizes nonzero ones, so the tables below are the actual cheat sheet, not "
        "an approximation of one."
    )
    lines.append("")

    active_spots = g.active_gto_spots()
    lines.append("## GTO Chart Overrides")
    if active_spots:
        lines.append(
            "This genome memorizes an exact chart for the spot(s) below -- whenever the "
            "situation matches, it plays straight off the chart instead of the rule list below "
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
        lines.append(
            "This genome doesn't trust any memorized chart -- every decision goes through the "
            "rule list below."
        )
        lines.append("")

    referenced = sorted(
        {int(fi) for fi in g.condition_features.flat if fi != strategy.WILDCARD},
        key=lambda fi: (
            _REPORT_GROUP_PRIORITY.get(group_of(strategy.TOP_LEVEL_FEATURES[fi]), _UNGROUPED_CONDITION_PRIORITY),
            strategy.TOP_LEVEL_FEATURES[fi].key,
        ),
    )
    lines.append("## Feature Buckets")
    lines.append(
        "This genome's own cutoffs for every feature its rules actually reference below -- "
        "shared by every rule, the way a real chart's category boundaries are defined once "
        "and reused."
    )
    lines.append("")
    if not referenced:
        lines.append("*(No rule references a feature -- this genome always plays its Strategy Rules' default.)*")
        lines.append("")
    for fi in referenced:
        spec = strategy.TOP_LEVEL_FEATURES[fi]
        lines.append(f"- **{spec.label}** — {spec.description}")
        if spec.kind == "boolean":
            continue
        row = strategy.bucket_gene_row(fi)
        num_buckets = int(g.num_buckets[row])
        thresholds = g.thresholds[row]
        for b in range(num_buckets):
            lines.append(f"  - Bucket {b}: {strategy.describe_bucket(spec, b, num_buckets, thresholds)}")
    lines.append("")

    lines.append("## Strategy Rules")
    lines.append(
        "Grouped here by shared situation for readability (context features like position/"
        "street first, situational triggers like facing a bet or call size last), **not** in "
        "true evaluation order -- each line is tagged with its actual rule number in brackets. "
        "When two rules could both match the same hand, the lower-numbered one wins; this only "
        "matters when rules' conditions actually overlap, which is uncommon since most rules "
        "in the same spot differ on a trigger condition (facing a bet or not, etc.) that makes "
        "them mutually exclusive in practice."
    )
    lines.append("")
    rule_order = sorted(
        range(strategy.NUM_RULES),
        key=lambda r: _rule_sort_key(g.condition_features[r], g.condition_buckets[r]),
    )
    for r in rule_order:
        phrases = []
        for c in range(strategy.CONDITIONS_PER_RULE):
            fi = int(g.condition_features[r, c])
            if fi == strategy.WILDCARD:
                continue
            spec = strategy.TOP_LEVEL_FEATURES[fi]
            row = strategy.bucket_gene_row(fi)
            num_buckets = int(g.num_buckets[row]) if row >= 0 else 2
            thresholds = g.thresholds[row] if row >= 0 else None
            phrases.append(_condition_phrase(spec, int(g.condition_buckets[r, c]), num_buckets, thresholds))
        condition_text = " AND ".join(phrases) if phrases else "*(any situation)*"
        action = strategy.ACTION_CATEGORIES[int(g.rule_actions[r])]
        lines.append(f"- `[rule {r}]` IF {condition_text} THEN **{action}**")
    lines.append("- `[default]` IF nothing above matched THEN **Fold**")
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
        "| rank | player | mean net chips/session | win rate | bust rate | bb/100 | active conditions |"
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
