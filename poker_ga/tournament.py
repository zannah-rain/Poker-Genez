"""Final tournament: re-evaluates a generation with more rounds/hands than
the per-generation GA fitness pass (which is deliberately cheap so evolution
stays fast), producing a higher-confidence ranking. Also exports the top
genomes as human-readable strategy reports plus performance statistics.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from features import FEATURE_NAMES, NUM_FEATURES
from game import GameConfig, SeatState, play_hand
from genome import ACTION_NAMES, BET_RAISE, Genome
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
    hands_survived = {pid: 0 for pid in starting_ids}
    busted = {pid: False for pid in starting_ids}

    button_idx = 0
    hands_played = 0
    while len(seats) > 1 and hands_played < game_config.max_hands_per_session:
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


def _top_features(weights: np.ndarray, feature_names: list[str], k: int = 6) -> list[tuple]:
    order = np.argsort(-np.abs(weights))[:k]
    return [(feature_names[i], float(weights[i])) for i in order]


def describe_genome(player: Player, stats: PlayerStats, game_config: GameConfig, rank: int) -> str:
    """Renders a genome's weights as a human-readable strategy report."""
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

    lines.append("## Action tendencies")
    lines.append(
        "Baseline tilt with no other information (bias term; positive favors "
        "that action by default):"
    )
    for a, name_a in enumerate(ACTION_NAMES):
        lines.append(f"- {name_a}: {g.action_bias[a]:+.3f}")
    lines.append("")

    for a, action_name in enumerate(ACTION_NAMES):
        lines.append(f"### What drives \"{action_name}\"")
        top = _top_features(g.action_weights[a], FEATURE_NAMES, k=8)
        lines.append("| feature | weight | effect |")
        lines.append("|---|---|---|")
        for feat, w in top:
            effect = f"pushes toward {action_name}" if w > 0 else f"pushes away from {action_name}"
            lines.append(f"| {feat} | {w:+.3f} | {effect} |")
        lines.append("")

    lines.append("## Bet sizing")
    base_frac = g.bet_size_fraction(np.zeros(NUM_FEATURES))
    lines.append(f"- Baseline bet size with neutral features: ~{base_frac:.2f}x pot")
    lines.append("")
    top_sizing = _top_features(g.sizing_weights, FEATURE_NAMES, k=6)
    lines.append("Features that most move bet sizing (positive = bets bigger):")
    lines.append("")
    lines.append("| feature | weight |")
    lines.append("|---|---|")
    for feat, w in top_sizing:
        lines.append(f"| {feat} | {w:+.3f} |")
    lines.append("")

    lines.append("## Exploration / bluffing")
    lines.append(
        f"- Decision noise (stddev added to action scores each decision): {g.noise_std:.3f}. "
        "Higher values mean more unpredictable/bluffy play; near zero means fully deterministic."
    )
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
    with open(os.path.join(out_dir, "leaderboard.md"), "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    for rank, p in enumerate(ranked_players[:n], start=1):
        s = stats[p.player_id]
        report = describe_genome(p, s, game_config, rank)
        base = f"rank{rank:02d}_player{p.player_id}"
        with open(os.path.join(out_dir, f"{base}_strategy.md"), "w") as f:
            f.write(report)
        p.genome.save(os.path.join(out_dir, f"{base}_genome.npy"))
