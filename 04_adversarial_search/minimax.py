"""Minimax with alpha-beta pruning, demonstrated on Tic-Tac-Toe.

MAX plays 'X' (wants +1), MIN plays 'O' (wants −1). Draw = 0.
"""

from __future__ import annotations

from dataclasses import dataclass

WIN_LINES = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
    (0, 4, 8), (2, 4, 6),             # diagonals
)


@dataclass(frozen=True)
class State:
    """Immutable Tic-Tac-Toe position."""

    board: tuple[str, ...] = (" ",) * 9
    to_move: str = "X"

    def legal_moves(self) -> list[int]:
        return [i for i, c in enumerate(self.board) if c == " "]

    def play(self, move: int) -> "State":
        board = list(self.board)
        board[move] = self.to_move
        return State(tuple(board), "O" if self.to_move == "X" else "X")

    def winner(self) -> str | None:
        for a, b, c in WIN_LINES:
            if self.board[a] != " " and self.board[a] == self.board[b] == self.board[c]:
                return self.board[a]
        return None

    def is_terminal(self) -> bool:
        return self.winner() is not None or " " not in self.board

    def utility(self) -> int:
        """+1 if X won, −1 if O won, 0 otherwise."""
        w = self.winner()
        return 1 if w == "X" else -1 if w == "O" else 0

    def render(self) -> str:
        b = self.board
        rows = [" | ".join(b[i : i + 3]) for i in (0, 3, 6)]
        return "\n---------\n".join(rows)


def minimax(
    state: State, alpha: float = float("-inf"), beta: float = float("inf")
) -> tuple[int, int | None]:
    """Return (value, best_move) under optimal play by both sides.

    Alpha-beta: alpha = best already guaranteed for MAX, beta = for MIN.
    A branch where alpha >= beta cannot influence the root — prune it.
    The returned value is identical to plain minimax.
    """
    if state.is_terminal():
        return state.utility(), None

    best_move: int | None = None
    if state.to_move == "X":  # MAX node
        value = float("-inf")
        for move in state.legal_moves():
            child_value, _ = minimax(state.play(move), alpha, beta)
            if child_value > value:
                value, best_move = child_value, move
            alpha = max(alpha, value)
            if alpha >= beta:
                break  # beta cutoff
    else:  # MIN node
        value = float("inf")
        for move in state.legal_moves():
            child_value, _ = minimax(state.play(move), alpha, beta)
            if child_value < value:
                value, best_move = child_value, move
            beta = min(beta, value)
            if alpha >= beta:
                break  # alpha cutoff
    return int(value), best_move


def self_play() -> State:
    """Both sides play the minimax-optimal move until the game ends."""
    state = State()
    while not state.is_terminal():
        _, move = minimax(state)
        state = state.play(move)
    return state


if __name__ == "__main__":
    value, first_move = minimax(State())
    print(f"Game value with optimal play: {value} (0 = draw)")
    print(f"Best first move for X: square {first_move}\n")

    final = self_play()
    print("Optimal self-play result:")
    print(final.render())
    print(f"\nWinner: {final.winner() or 'draw'}")
