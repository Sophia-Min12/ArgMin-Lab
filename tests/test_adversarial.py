"""Level 4 tests: minimax plays Tic-Tac-Toe perfectly.

Level 4 검증 — 미니맥스의 완전 플레이.
"""

from minimax import State, minimax, self_play


def test_optimal_tic_tac_toe_is_a_draw():
    """The classic theorem: perfect play from both sides ends in a draw."""
    value, _ = minimax(State())
    assert value == 0


def test_self_play_reaches_draw():
    final = self_play()
    assert final.is_terminal()
    assert final.winner() is None  # draw


def test_minimax_takes_immediate_win():
    """X to move with two in a row must complete the line."""
    board = ("X", "X", " ",
             "O", "O", " ",
             " ", " ", " ")
    value, move = minimax(State(board, "X"))
    assert move == 2
    assert value == 1


def test_minimax_blocks_opponent_win():
    """O to move must block X's open line."""
    board = ("X", "X", " ",
             " ", "O", " ",
             " ", " ", " ")
    _, move = minimax(State(board, "O"))
    assert move == 2
