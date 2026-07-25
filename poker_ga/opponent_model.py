"""Session-scoped tracking of each player's *observed* tendencies, so a
genome's decision can condition on how an opponent has actually played
rather than only on the current hand's raw game state -- the ingredient
needed for active exploitation (and, in turn, for evolution to have any
reason to develop strategies that are hard to read/exploit in return).

Stats are kept per player_id in an OpponentModel that's created fresh for
each table session (see simulate.run_session / benchmark._play_side_match)
and threaded through play_hand/betting_round -- deliberately *not* stored on
Player itself. Player is reused as a stable identity across many sessions
(see player.py), but tables are randomly reshuffled every generation, so
"how this opponent has played" should reset with the session/table, the way
a live HUD resets rather than a permanent cross-table dossier following a
username around.

The six rate stats below (VPIP, PFR, 3-bet%, fold-to-3-bet%, postflop
aggression frequency, fold-vs-bet) are the standard HUD stats cited across
poker theory for reading preflop looseness/aggression and postflop
folding/barreling tendencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from genome import BET_RAISE, CHECK_CALL, FOLD

# The "no information yet" prior every rate stat shrinks toward when a
# player has been observed few (or zero) times -- reads as "assume average"
# rather than a spurious 0% or 100% from a tiny sample.
NEUTRAL = 0.5
# How many "virtual" neutral observations are blended in alongside the real
# ones (see _rate) -- e.g. a player who has 3-bet once in one opportunity
# reads close to NEUTRAL, not a deterministic 100%, until real observations
# pile up past this count.
CONFIDENCE_HANDS = 20.0


def _rate(hits: int, chances: int) -> float:
    """Observed hit rate, shrunk toward NEUTRAL when `chances` is small (add
    CONFIDENCE_HANDS worth of "virtual" neutral chances/hits to both sides).
    Chances == 0 correctly falls out to exactly NEUTRAL."""
    return (hits + NEUTRAL * CONFIDENCE_HANDS) / (chances + CONFIDENCE_HANDS)


@dataclass
class OpponentStats:
    """Running counts for one player, accumulated hand-by-hand within a
    session. See module docstring for why this resets every session instead
    of persisting on Player."""

    hands: int = 0
    vpip_hands: int = 0
    pfr_hands: int = 0
    three_bet_chances: int = 0  # preflop decisions facing exactly one raise, with the option to raise
    three_bet_hits: int = 0
    fold_to_three_bet_chances: int = 0  # preflop decisions facing exactly two raises, with the option to fold
    fold_to_three_bet_hits: int = 0
    postflop_bets_raises: int = 0
    postflop_calls: int = 0  # postflop calls of a nonzero amount (a free check doesn't count)
    fold_vs_bet_chances: int = 0  # postflop decisions facing a bet, with the option to fold
    fold_vs_bet_hits: int = 0

    def vpip_rate(self) -> float:
        return _rate(self.vpip_hands, self.hands)

    def pfr_rate(self) -> float:
        return _rate(self.pfr_hands, self.hands)

    def three_bet_rate(self) -> float:
        return _rate(self.three_bet_hits, self.three_bet_chances)

    def fold_to_three_bet_rate(self) -> float:
        return _rate(self.fold_to_three_bet_hits, self.fold_to_three_bet_chances)

    def aggression_freq(self) -> float:
        return _rate(self.postflop_bets_raises, self.postflop_bets_raises + self.postflop_calls)

    def fold_vs_bet_rate(self) -> float:
        return _rate(self.fold_vs_bet_hits, self.fold_vs_bet_chances)


@dataclass
class OpponentModel:
    """Owns one OpponentStats per player_id seen this session, plus the
    per-hand bookkeeping needed to count VPIP/PFR at most once per hand
    (a player can call then later face a raise and call again the same
    hand -- that's still only one "voluntarily played this hand", not two)."""

    stats: dict[int, OpponentStats] = field(default_factory=dict)
    _vpip_flagged_this_hand: set = field(default_factory=set)
    _pfr_flagged_this_hand: set = field(default_factory=set)

    def get(self, player_id: int) -> OpponentStats:
        return self.stats.setdefault(player_id, OpponentStats())

    def start_hand(self, player_ids) -> None:
        """Call once per hand, before any decisions, with every seated
        player's id (including busted-then-refilled players' new
        identities) -- resets the per-hand VPIP/PFR dedup and counts this
        hand toward each player's sample size."""
        self._vpip_flagged_this_hand = set()
        self._pfr_flagged_this_hand = set()
        for pid in player_ids:
            self.get(pid).hands += 1

    def record_action(
        self,
        player_id: int,
        is_preflop: bool,
        action: int,
        call_amount: float,
        num_raises_before: int,
        legal_actions: list,
    ) -> None:
        """Updates running counts from one decision. `num_raises_before` is
        the raise count on this street *before* this action (i.e. what the
        deciding player saw), and `legal_actions` is what was actually on
        offer -- both needed to distinguish a real "chance" (e.g. facing a
        3-bet with the option to fold) from a situation where the stat
        doesn't apply (e.g. already all-in, no fold available)."""
        s = self.get(player_id)
        facing_bet = call_amount > 1e-9

        if is_preflop:
            if action == BET_RAISE or (action == CHECK_CALL and facing_bet):
                if player_id not in self._vpip_flagged_this_hand:
                    self._vpip_flagged_this_hand.add(player_id)
                    s.vpip_hands += 1
            if action == BET_RAISE and player_id not in self._pfr_flagged_this_hand:
                self._pfr_flagged_this_hand.add(player_id)
                s.pfr_hands += 1
            if num_raises_before == 1 and BET_RAISE in legal_actions:
                s.three_bet_chances += 1
                if action == BET_RAISE:
                    s.three_bet_hits += 1
            if num_raises_before == 2 and FOLD in legal_actions:
                s.fold_to_three_bet_chances += 1
                if action == FOLD:
                    s.fold_to_three_bet_hits += 1
        else:
            if facing_bet and FOLD in legal_actions:
                s.fold_vs_bet_chances += 1
                if action == FOLD:
                    s.fold_vs_bet_hits += 1
            if action == BET_RAISE:
                s.postflop_bets_raises += 1
            elif action == CHECK_CALL and facing_bet:
                s.postflop_calls += 1


@dataclass(frozen=True)
class OpponentFeatures:
    """Exploitative feature values for one decision -- see features.py's
    Situation fields of the same names. Each averages over every currently
    active opponent."""

    opp_vpip: float = NEUTRAL
    opp_pfr: float = NEUTRAL
    opp_three_bet: float = NEUTRAL
    opp_fold_to_three_bet: float = NEUTRAL
    opp_aggression_freq: float = NEUTRAL
    opp_fold_vs_bet: float = NEUTRAL


def compute_opponent_features(
    model: "OpponentModel | None",
    opponent_ids: list,
) -> OpponentFeatures:
    """Aggregates `model`'s stats into the values a Situation needs for one
    decision. Returns all-NEUTRAL defaults if there's no model (e.g. a
    caller that doesn't opt in) or no active opponents."""
    if model is None or not opponent_ids:
        return OpponentFeatures()

    opponent_stats = [model.get(pid) for pid in opponent_ids]
    n = len(opponent_stats)
    return OpponentFeatures(
        opp_vpip=sum(s.vpip_rate() for s in opponent_stats) / n,
        opp_pfr=sum(s.pfr_rate() for s in opponent_stats) / n,
        opp_three_bet=sum(s.three_bet_rate() for s in opponent_stats) / n,
        opp_fold_to_three_bet=sum(s.fold_to_three_bet_rate() for s in opponent_stats) / n,
        opp_aggression_freq=sum(s.aggression_freq() for s in opponent_stats) / n,
        opp_fold_vs_bet=sum(s.fold_vs_bet_rate() for s in opponent_stats) / n,
    )
