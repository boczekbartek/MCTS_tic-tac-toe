""" Main file for Monte Carlo Tree Search in Tic-Tac-Toe game """
__author__ = "Bartłomiej Boczek, Krzyszfor Linke"

from dataclasses import dataclass
from typing import List, Tuple
import random

import numpy as np


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
            unique = [self.board[x][y] for x, y in cross_cords]  # TODO numpy way
            if len(unique) == 1 and unique[0] != states.EMPTY:
                return unique[0]
        return states.EMPTY

    def __str__(self) -> str:
        s = ""
        for row in self.board:
            for state in row:
                s += states.translate(state) + " "
            s += "\n"
        return s


# Create the game
game = TicTacToe()

# Select first player randomly (circles or crosses)
cur_player = random.choice(states.allowed_moves)

# Get initial possible moves
moves = game.get_possible_moves()

# Play the game until there are still moves left
while len(moves) != 0:
    # Random strategy of move choice
    x, y = random.choice(moves)

    game.move(player=cur_player, x=x, y=y)

    # Check if the game already has a winner
    winner = game.evaluate_game()
    if winner != states.EMPTY:
        break

    # switch player
    cur_player = states.CROSS if cur_player == states.CIRCLE else states.CIRCLE

    # get possible moves
    moves = game.get_possible_moves()

print(game)
winner = game.evaluate_game()
print(f"Winner is: {states.translate(winner)}")
