"""Tests for the Rush Hour core: parsing, vehicles, features and moves."""

import pytest

from src.rushhour.features import blocking_cars, free_exit_cells, red_car_distance_to_exit
from src.rushhour.model import Move
from src.rushhour.moves import apply_move, is_goal, legal_moves
from src.rushhour.parser import board_to_rows, find_vehicles, parse_board_string

TINY_BOARD = (
    "......"
    "..A..."
    "XXA..."
    "......"
    "......"
    "......"
)


@pytest.fixture
def tiny_state():
    return parse_board_string(TINY_BOARD)


def test_parse_tiny_board(tiny_state):
    assert len(tiny_state.board) == 36
    assert "X" in tiny_state.board


def test_board_to_rows(tiny_state):
    rows = board_to_rows(tiny_state)
    assert len(rows) == 6
    assert all(len(row) == 6 for row in rows)
    assert rows[2] == "XXA..."


def test_parse_ignores_whitespace_and_newlines(tiny_state):
    with_newlines = "......\n..A...\nXXA...\n......\n......\n......\n"
    assert parse_board_string(with_newlines) == tiny_state


def test_vehicles_detected(tiny_state):
    vehicles = find_vehicles(tiny_state)
    assert set(vehicles) == {"A", "X"}
    assert vehicles["A"].orientation == "V"
    assert vehicles["A"].length == 2
    assert vehicles["X"].orientation == "H"
    assert vehicles["X"].length == 2


def test_features_on_tiny_board(tiny_state):
    assert blocking_cars(tiny_state) == 1
    assert red_car_distance_to_exit(tiny_state) == 4
    assert free_exit_cells(tiny_state) == 0


def test_legal_moves_include_a_down(tiny_state):
    moves = legal_moves(tiny_state)
    assert Move("A", "D", 1) in moves
    assert Move("A", "U", 1) in moves
    # X is blocked on the right by A and sits at the left edge
    assert not any(move.car == "X" for move in moves)


def test_move_str_token():
    assert str(Move("A", "D", 1)) == "1AD"
    assert str(Move("X", "R", 3)) == "3XR"


def test_apply_move_reaches_goal(tiny_state):
    # A sits on rows 1-2, so moving it up by 1 clears the exit row
    after = apply_move(tiny_state, Move("A", "U", 1))
    assert is_goal(after)
    # moving A down by 1 leaves it on rows 2-3, still blocking the exit row
    still_blocked = apply_move(tiny_state, Move("A", "D", 1))
    assert not is_goal(still_blocked)
    # moving A down by 2 also clears the exit row
    assert is_goal(apply_move(tiny_state, Move("A", "D", 2)))


def test_apply_move_does_not_mutate(tiny_state):
    board_before = tiny_state.board
    apply_move(tiny_state, Move("A", "U", 1))
    assert tiny_state.board == board_before


def test_illegal_moves_raise(tiny_state):
    with pytest.raises(ValueError):
        apply_move(tiny_state, Move("A", "L", 1))  # vertical car cannot move left
    with pytest.raises(ValueError):
        apply_move(tiny_state, Move("X", "R", 1))  # blocked by A
    with pytest.raises(ValueError):
        apply_move(tiny_state, Move("X", "L", 1))  # would leave the board
    with pytest.raises(ValueError):
        apply_move(tiny_state, Move("Z", "D", 1))  # no such vehicle


def test_malformed_board_length_raises():
    with pytest.raises(ValueError):
        parse_board_string("XX....")


def test_missing_red_car_raises():
    with pytest.raises(ValueError):
        parse_board_string("." * 34 + "AA")


def test_malformed_vehicle_raises():
    # B appears in two separate places, not a straight line
    board = "B....." "......" "XX...." "......" "......" ".....B"
    with pytest.raises(ValueError):
        find_vehicles(parse_board_string(board))
