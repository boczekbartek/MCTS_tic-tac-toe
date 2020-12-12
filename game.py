""" Tic-tac-toe game representation """
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


class states:
    """ Tic-Tac-Toe states """

    EMPTY: int = 0
    CIRCLE: int = 1
    CROSS: int = 2
    d: dict = {EMPTY: "_", CIRCLE: "o", CROSS: "x"}
    allowed_moves = (CIRCLE, CROSS)

    @classmethod
    def translate(cls, state: int) -> str:
        """ Translate state into string representation x or o """
        return cls.d[state]


@dataclass
class TicTacToe:
    """ Tic-Tac-Toe game representation """

    size: int = 3
    board: np.array = np.full(shape=(size, size), fill_value=states.EMPTY)

    def move(self, player: int, x: int, y: int):
        """ 
        Perform the move 
        
        Parameters
        ----------
        player : int
            states.CIRCLE or states.CROSS
        x : int
            x cordinate of the move, 0..(size-1)
        y : int
            y cordinate of the move, 0..(size-1)
        """
        assert (
            player in states.allowed_moves
        ), f"player={player} is invalid! Must be one of {states.allowed_moves}"
        assert 0 <= x < self.size, f"x={x} is invalid! {0} <= x < {self.size}"
        assert 0 <= y < self.size, f"y={y} is invalid! {0} <= y < {self.size}"
        assert self.board[x][y] == states.EMPTY, f"({x},{y}) is not EMPTY!"
        self.board[x][y] = player

    def get_possible_moves(self) -> List[Tuple[int, int]]:
        """
        Get coordinates of possible moves

        Returns
        -------
        List[Tuple[int,int]]
            list of (x,y) pairs that are still empty on the game board
        """
        moves = []
        for x in range(self.size):
            for y in range(self.size):
                if self.board[x][y] == states.EMPTY:
                    moves.append((x, y))
        return moves

    def evaluate_game(self) -> int:
        """ 
        Evaluate the game and find its winner, if there is any
        
        Returns
        -------
        int
            Winner of the game. states.CROSS -> x, states.CIRCLE -> o, states.EMPTY -> no winner
        """
        # check if there is a row win
        for row in self.board:
            unique = np.unique(row)
            if len(unique) == 1 and unique[0] != states.EMPTY:
                return unique[0]
        for col in self.board.T:
            unique = np.unique(col)
            if len(unique) == 1 and unique[0] != states.EMPTY:
                return unique[0]
        for cross_cords in [
            [(0, 0), (1, 1), (2, 2)],
            [(0, 2), (1, 1), (2, 0)],
        ]:
            unique = np.unique(
                [self.board[x][y] for x, y in cross_cords]
            )  # TODO numpy way
            if len(unique) == 1 and unique[0] != states.EMPTY:
                return unique[0]
        return states.EMPTY

    def __str__(self) -> str:
        s = ""
        for row in self.board:
            for state in row:
                s += states.translate(state) + " "
            s += "\n"
        return s[:-1]

    def get_empty_fields(self):
        """ Return empty fields left in the game """
        empty = []
        for x in range(self.size):
            for y in range(self.size):
                if self.board[x][y] == states.EMPTY:
                    empty.append((x, y))
        return empty

    def copy(self):
        return TicTacToe(size=self.size, board=self.board.copy())

