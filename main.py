""" Main file for Monte Carlo Tree Search in Tic-Tac-Toe game """
__author__ = "Bartłomiej Boczek, Krzyszfor Linke"

import argparse
from copy import deepcopy
import itertools
import logging
from os import stat
import random
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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


def create_full_metric_mtx(metric_values, ttc: TicTacToe):
    metrics_iter = iter(metric_values)
    result = np.empty_like(ttc.board, dtype=float)
    for x in range(ttc.size):
        for y in range(ttc.size):
            result[x][y] = 0 if ttc.board[x][y] != states.EMPTY else next(metrics_iter)
    return result


def main(n_rollouts: int, ini_game: str, plot=False):
    # Initialize the game with one of predefined boards
    mcts_player = states.CROSS
    logging.info(f"{states.translate(mcts_player)} plays with MCTS")
    game = TicTacToe(board=INITIAL_STATES[ini_game]).copy()
    logging.info(f"Initial game:\n{game}")

    # 'x' starts
    cur_player = states.CROSS

    # Monte Carlo Tree Search loop
    empty_fields = game.get_empty_fields()
    i = 0
    win_save, q_save = None, None
    while len(empty_fields) != 0:
        if cur_player == mcts_player:
            mcts = MCTS(
                game_state=game, n_iters=n_rollouts, player=cur_player, uct=True
            )
            best_move, win_prob, q_values = mcts.run()
            x, y = best_move
            if i == 0:
                win_save = deepcopy(win_prob)
                q_save = deepcopy(q_values)
                win_probs_mtx = create_full_metric_mtx(win_prob, game)
                q_values_mtx = create_full_metric_mtx(q_values, game)

                if plot:
                    plt.figure()
                    sns.heatmap(win_probs_mtx, annot=True)
                    plt.title(
                        f"Winning probabilities calculated for {n_rollouts} rollouts"
                    )
                    win_probs_fname = f"win-probs.{ini_game}.png"
                    plt.savefig(win_probs_fname)

                    logging.info(f"Saving win-probs to: {win_probs_fname}")

                    logging.info(q_values_mtx)
                    plt.figure()
                    sns.heatmap(q_values_mtx, annot=True)
                    plt.title(f"Q-values calculated for {n_rollouts} rollouts")
                    q_values_fname = f"q-vals.{ini_game}.png"
                    plt.savefig(q_values_fname)
                    logging.info(f"Saving q-values to: {q_values_fname}")
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
        i += 1

    winner = game.evaluate_game()
    logging.info(f"Winner: {states.translate(winner)}")
    return winner == mcts_player, list(win_save), list(q_save)


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
