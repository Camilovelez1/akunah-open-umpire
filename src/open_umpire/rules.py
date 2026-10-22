"""Table tennis scoring, driven by bounce events.

The referee only needs to know on which half of the table the ball bounced
and when. Everything else follows from one observation: when a rally ends,
the point goes against the player on whose half the ball bounced last,
because that player had to return it and didn't (or returned it out).

Exceptions handled explicitly:
- the serve must bounce first on the server's half,
- two bounces in a row on the same half end the rally immediately,
- a serve that never bounces is a point for the receiver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Side(str, Enum):
    LEFT = "left"
    RIGHT = "right"

    @property
    def other(self) -> Side:
        return Side.RIGHT if self is Side.LEFT else Side.LEFT


@dataclass(frozen=True)
class MatchConfig:
    points_to_win: int = 11
    win_by: int = 2
    serves_per_turn: int = 2
    games_to_win: int = 3
    swap_ends_each_game: bool = True
    # Seconds without a bounce after which a running rally is over.
    rally_timeout: float = 1.5
    # Seconds after a point during which bounces are ignored (ball rolling, being picked up).
    dead_time: float = 2.0


@dataclass
class PointEvent:
    winner: int
    reason: str
    time: float


@dataclass
class Snapshot:
    """Everything needed to undo a point."""

    scores: tuple[int, ...]
    games: tuple[int, ...]
    server: int
    serves_left: int
    game_first_server: int
    left_player: int
    finished: bool


@dataclass
class Match:
    """Score of a match between player 0 and player 1."""

    config: MatchConfig = field(default_factory=MatchConfig)
    names: tuple[str, str] = ("Player 1", "Player 2")
    # Points each player starts every game with (handicap hook).
    start_scores: tuple[int, int] = (0, 0)
    first_server: int = 0
    left_player: int = 0

    def __post_init__(self) -> None:
        self.scores = list(self.start_scores)
        self.games = [0, 0]
        self.finished = False
        self.history: list[Snapshot] = []
        self.server = self.first_server
        self.serves_left = self.config.serves_per_turn
        self._game_first_server = self.first_server

    def side_of(self, player: int) -> Side:
        return Side.LEFT if player == self.left_player else Side.RIGHT

    def player_at(self, side: Side) -> int:
        return self.left_player if side is Side.LEFT else 1 - self.left_player

    def add_point(self, winner: int) -> None:
        if self.finished:
            return
        self.history.append(
            Snapshot(
                tuple(self.scores), tuple(self.games), self.server, self.serves_left,
                self._game_first_server, self.left_player, self.finished,
            )
        )
        self.scores[winner] += 1
        if self._game_won(winner):
            self.games[winner] += 1
            if self.games[winner] >= self.config.games_to_win:
                self.finished = True
                return
            self.scores = list(self.start_scores)
            self._game_first_server = 1 - self._game_first_server
            self.server = self._game_first_server
            self.serves_left = self.config.serves_per_turn
            if self.config.swap_ends_each_game:
                self.left_player = 1 - self.left_player
            return
        deuce = min(self.scores) >= self.config.points_to_win - 1
        self.serves_left = 1 if deuce else self.serves_left - 1
        if self.serves_left <= 0 or deuce:
            self.server = 1 - self.server
            self.serves_left = 1 if deuce else self.config.serves_per_turn

    def undo(self) -> bool:
        if not self.history:
            return False
        snap = self.history.pop()
        self.scores = list(snap.scores)
        self.games = list(snap.games)
        self.server = snap.server
        self.serves_left = snap.serves_left
        self._game_first_server = snap.game_first_server
        self.left_player = snap.left_player
        self.finished = snap.finished
        return True

    def swap_ends(self) -> None:
        self.left_player = 1 - self.left_player

    def _game_won(self, player: int) -> bool:
        mine, theirs = self.scores[player], self.scores[1 - player]
        return mine >= self.config.points_to_win and mine - theirs >= self.config.win_by

    def to_dict(self) -> dict:
        return {
            "names": list(self.names),
            "scores": list(self.scores),
            "games": list(self.games),
            "server": self.server,
            "left_player": self.left_player,
            "finished": self.finished,
        }


class Referee:
    """Turns a stream of bounces into points on a Match."""

    def __init__(self, match: Match) -> None:
        self.match = match
        self.bounces: list[tuple[Side, float]] = []
        self.serve_started_at: float | None = None
        self.events: list[PointEvent] = []
        self._ignore_until = float("-inf")

    @property
    def in_rally(self) -> bool:
        return bool(self.bounces) or self.serve_started_at is not None

    def start_serve(self, t: float) -> None:
        """Optional: call when the serve toss/hit is seen, so a serve that never bounces counts."""
        if not self.in_rally and t >= self._ignore_until:
            self.serve_started_at = t

    def bounce(self, side: Side, t: float) -> PointEvent | None:
        if self.match.finished or t < self._ignore_until:
            return None
        server_side = self.match.side_of(self.match.server)
        if not self.bounces and side is not server_side:
            # A serve has to touch the server's own half first.
            self.bounces.append((side, t))
            return self._award(self.match.player_at(server_side.other), "serve did not bounce on server's half", t)
        if self.bounces and self.bounces[-1][0] is side:
            self.bounces.append((side, t))
            return self._award(self.match.player_at(side.other), f"double bounce on {side.value} half", t)
        self.bounces.append((side, t))
        return None

    def tick(self, t: float) -> PointEvent | None:
        """Call regularly; ends the rally after `rally_timeout` seconds without bounces."""
        timeout = self.match.config.rally_timeout
        if self.bounces:
            last_side, last_t = self.bounces[-1]
            if t - last_t >= timeout:
                return self._award(self.match.player_at(last_side.other), f"ball not returned from {last_side.value} half", t)
        elif self.serve_started_at is not None and t - self.serve_started_at >= timeout:
            receiver = 1 - self.match.server
            return self._award(receiver, "serve never bounced", t)
        return None

    def cancel_rally(self) -> None:
        self.bounces.clear()
        self.serve_started_at = None

    def _award(self, winner: int, reason: str, t: float) -> PointEvent:
        self.match.add_point(winner)
        self.cancel_rally()
        self._ignore_until = t + self.match.config.dead_time
        event = PointEvent(winner, reason, t)
        self.events.append(event)
        return event
