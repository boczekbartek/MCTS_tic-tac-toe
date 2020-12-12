""" Main file for Monte Carlo Tree Search in Tic-Tac-Toe game """
__author__ = "Bartłomiej Boczek, Krzyszfor Linke"

import argparse
import itertools
import logging
from os import stat
import random
from collections import defaultdict
import numpy as np

from game import TicTacToe, states
from mcts import MCTS

# Predefined initial states of the game
INITIAL_STATES = {
    "empty": np.full((3, 3), states.EMPTY),
    "assignment": np.array(
        [
            [states.EMPTY, states.CIRCLE, states.CROSS],
            [states.EMPTY, states.CROSS, states.EMPTY],
            [states.CIRCLE, states.EMPTY, states.EMPTY],
        ]
    ),
}


def main(n_rollouts: int, ini_game: str):
    # Initialize the game with one of predefined boards
    mcts_player = states.CROSS
    logging.info(f"{states.translate(mcts_player)} plays with MCTS")
    game = TicTacToe(board=INITIAL_STATES[ini_game])
    logging.info(f"Initial game:\n{game}")

    # 'x' starts
    cur_player = states.CROSS

    # Monte Carlo Tree Search loop
    empty_fields = game.get_empty_fields()
    while len(empty_fields) != 0:
        if cur_player == mcts_player:
            mcts = MCTS(
                game_state=game, n_iters=n_rollouts, player=cur_player, uct=True
            )
            best_move = mcts.run()
            x, y = best_move
        else:
            moves = game.get_possible_moves()
            x, y = random.choice(moves)

        game.move(player=cur_player, x=x, y=y)

        logging.info(f"Player '{states.translate(cur_player)}' moved to ({x},{y})")
        logging.info(game)

        winner = game.evaluate_game()
        if winner != states.EMPTY:
            break

        # Switch player
        cur_player = states.CROSS if cur_player == states.CIRCLE else states.CIRCLE

        # Check which fields are stll empty in the game

        empty_fields = game.get_empty_fields()

    winner = game.evaluate_game()
    logging.info(f"Winner: {states.translate(winner)}")
    print(f"Winner: {states.translate(winner)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Tic-tac-toe game, Monte Carlo Tree Search vs random agent."
    )
    p.add_argument(
        "--n-rollouts",
        required=True,
        type=int,
        help="Number of rollouts before taking actions.",
    )
    p.add_argument(
        "--ini-game",
        required=True,
        choices=list(INITIAL_STATES.keys()),
        help="Initial game state.",
    )
    p.add_argument("--verbose", action="store_true", help="Show more extensive logs")
    args = p.parse_args()
    logging.basicConfig(level="DEBUG" if args.verbose else "INFO", format="%(message)s")
    args_dict = vars(args)
    del args_dict["verbose"]
    main(**args_dict)
