""" Main file for Monte Carlo Tree Search in Tic-Tac-Toe game """
__author__ = "Bartłomiej Boczek, Krzyszfor Linke"

import itertools
import logging
import random
from dataclasses import dataclass
from typing import List, Tuple
from collections import defaultdict


import numpy as np

logging.basicConfig(level="INFO", format="%(message)s")


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
        empty = []
        for x in range(self.size):
            for y in range(self.size):
                if self.board[x][y] == states.EMPTY:
                    empty.append((x, y))
        return empty

    def copy(self):
        return TicTacToe(size=self.size, board=self.board.copy())


def setup_initial_state_game() -> TicTacToe:
    game = TicTacToe()
    game.board = np.array(
        [
            [states.EMPTY, states.CIRCLE, states.CROSS],
            [states.EMPTY, states.CROSS, states.CIRCLE],
            [states.CIRCLE, states.EMPTY, states.EMPTY],
        ]
    )
    return game


def follow_random_path(game: TicTacToe, player: int) -> int:
    """ Unwrap the game randomly to the end and return which player won """
    mgame = game.copy()
    moves = mgame.get_possible_moves()
    # Play the game until there are still moves left
    while len(moves) != 0:
        logging.debug(mgame)

        # Random strategy of move choice
        x, y = random.choice(moves)
        logging.debug(f"{states.translate(player)}'s move")
        mgame.move(player=player, x=x, y=y)

        # Check if the game already has a winner
        winner = mgame.evaluate_game()
        if winner != states.EMPTY:
            break

        # switch player
        player = states.CROSS if player == states.CIRCLE else states.CIRCLE
        # get possible moves
        moves = mgame.get_possible_moves()
    logging.debug(f"Final:\n{mgame}")
    winner = mgame.evaluate_game()
    logging.debug(f"There is a winner: {states.translate(winner)}")
    return winner


game = setup_initial_state_game()
cur_player = states.CROSS
other_player = states.CROSS if cur_player == states.CIRCLE else states.CIRCLE

empty_fields = game.get_empty_fields()
max_iters = 10
logging.info(f"Initial game:\n{game}")
while len(empty_fields) != 0:
    rewards = defaultdict(list)
    i = 0
    for x, y in itertools.cycle(empty_fields):
        this_game = game.copy()
        this_game.move(cur_player, x=x, y=y)
        logging.debug(f"Simuation #{i}")
        winner = follow_random_path(game=this_game, player=other_player)
        if winner == cur_player:  # you won
            reward = 1
        elif winner == states.EMPTY:  # draw, no winner
            reward = 0
        else:
            reward = -1  # you loose
        rewards[(x, y)].append(reward)
        i += 1
        if i >= max_iters:
            break
    logging.debug(f"Rewards: {rewards}")
    best_action = max(rewards.items(), key=lambda v: np.mean(v[1]))[0]
    logging.debug(f"ACTION: {best_action}")
    bx, by = best_action
    game.move(player=cur_player, x=bx, y=by)
    logging.info(
        f"Player {states.translate(cur_player)} moved to ({bx},{by}) with MCTS"
    )
    logging.info(game)
    x, y = random.choice(game.get_possible_moves())
    game.move(other_player, x=x, y=y)
    logging.info(f"Player {states.translate(other_player)} moved to ({x},{y}) randomly")
    logging.info(game)

    empty_fields = game.get_empty_fields()

winner = game.evaluate_game()

logging.info(f"Winner: {states.translate(winner)}")
