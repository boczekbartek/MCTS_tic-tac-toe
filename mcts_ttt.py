""" Main file for Monte Carlo Tree Search in Tic-Tac-Toe game """
__author__ = "Bartłomiej Boczek, Krzyszfor Linke"

import argparse
import logging
import random
from collections import defaultdict
import numpy as np

from game import TicTacToe, states

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


def follow_random_path(game: TicTacToe, player: int) -> int:
    """ Unwrap the game randomly to the end and return which player won """
    mgame = (
        game.copy()
    )  # copy the game, because we don't want to change the actual game, just simulate
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
    winner = mgame.evaluate_game()  # check the winner
    logging.debug(f"There is a winner: {states.translate(winner)}")
    return winner


def main(n_rollouts: int, ini_game: str):
    # Initialize the game with one of predefined boards
    game = TicTacToe(board=INITIAL_STATES[ini_game])
    logging.info(f"Initial game:\n{game}")

    # 'x' starts
    cur_player = states.CROSS
    other_player = states.CROSS if cur_player == states.CIRCLE else states.CIRCLE

    # Monte Carlo Tree Search loop
    empty_fields = game.get_empty_fields()
    while len(empty_fields) != 0:
        rewards = defaultdict(list)

        i = 0
        while i < n_rollouts:
            # Select random move for simulation
            field_i = np.random.choice(np.arange(0, len(empty_fields), 1), 1)[0]
            x, y = empty_fields[field_i]

            # Perform the move
            this_game = game.copy()
            this_game.move(cur_player, x=x, y=y)
            logging.debug(f"Simuation #{i}")

            # Simulate
            winner = follow_random_path(game=this_game, player=other_player)

            # Assign reward
            if winner == cur_player:  # you won
                reward = 1
            elif winner == states.EMPTY:  # draw, no winner
                reward = 0
            else:
                reward = -1  # you loose

            rewards[(x, y)].append(reward)

            i += 1
            # Iterate until we reach desired number of rollouts
            if i >= n_rollouts:
                break
        logging.debug(f"Rewards: {rewards}")

        # Best action is the one with greatest mean reward
        best_action = max(rewards.items(), key=lambda v: np.sum(v[1]) / n_rollouts)[0]
        logging.debug(f"ACTION: {best_action}")

        # Perform best action
        bx, by = best_action
        game.move(player=cur_player, x=bx, y=by)
        logging.info(
            f"Player {states.translate(cur_player)} moved to ({bx},{by}) with MCTS"
        )
        logging.info(game)

        # Perform move of the random agent
        moves = game.get_possible_moves()
        if len(moves) <= 0:
            break
        x, y = random.choice(moves)
        game.move(other_player, x=x, y=y)
        logging.info(
            f"Player {states.translate(other_player)} moved to ({x},{y}) randomly"
        )
        logging.info(game)

        # Check which fields are stll empty in the game
        empty_fields = game.get_empty_fields()

    winner = game.evaluate_game()

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
