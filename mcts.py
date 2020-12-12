from dataclasses import dataclass, field
import logging
import random
from typing import Tuple
from copy import deepcopy
from tqdm import tqdm
import numpy as np

from game import TicTacToe, states


@dataclass
class TreeNode(object):
    game_state: TicTacToe
    player: int
    parent: int = None
    children: list = field(default_factory=list)
    w: int = 0
    q: int = 0
    n: int = 0
    move: Tuple[int, int] = None

    def add_child(self, child_id: int):
        self.children.append(child_id)

    def has_children(self) -> bool:
        return len(self.children) != 0


class MCTS(object):
    def __init__(self, game_state, n_iters, player, c=np.sqrt(2), uct=True):
        self.c = c
        self.n_iters = n_iters
        self.tree: dict[tuple, TreeNode] = dict()
        self.tree[(0,)] = TreeNode(game_state=game_state, player=player)
        self.selection = self.selection_uct if uct else self.selection_rand
        self.cur_n = 0

    @staticmethod
    def ucb(w: int, n: int, c: int, total_n: int, node):
        logging.debug(f"UCB {node}| W={w}, n={n},c={c}, total_n={total_n}")
        if n == 0:
            n = 1e-6
        exploit = w / n
        explore = np.sqrt(np.log(total_n) / n)
        logging.debug(f"UCB | Exploit: {exploit}, explore: {explore}")
        return exploit + c * explore

    def selection_uct(self) -> int:
        leaf_node_found = False
        leaf_node_id = (0,)
        while not leaf_node_found:
            node_id = leaf_node_id
            if not self.tree[node_id].has_children():
                leaf_node_found = True
                leaf_node_id = node_id
            else:
                ucbs = [
                    self.ucb(
                        w=self.tree[child].w,
                        n=self.tree[child].n,
                        c=self.c,
                        total_n=self.cur_n,
                        node=self.tree[child].move,
                    )
                    for child in self.tree[node_id].children
                ]
                logging.debug(f"UCB values: {ucbs}")
                action = np.argmax(ucbs)
                leaf_node_id = node_id + (action,)
        return leaf_node_id

    def selection_rand(self) -> TreeNode:
        pass

    def expansion(self, node_id: int) -> int:
        game_state: TicTacToe = self.tree[node_id].game_state
        winner = game_state.evaluate_game()
        if winner != states.EMPTY:
            return node_id

        moves = game_state.get_possible_moves()
        children = list()
        for move_id, move in enumerate(moves):
            cur_player = self.tree[node_id].player
            state = game_state.copy()
            next_player = states.CROSS if cur_player == states.CIRCLE else states.CIRCLE

            child_id = node_id + (move_id,)
            children.append(child_id)
            state.move(cur_player, x=move[0], y=move[1])
            self.tree[child_id] = TreeNode(
                parent=node_id, game_state=state, player=next_player, move=move,
            )
            self.tree[node_id].add_child(child_id)
        rand_idx = np.random.randint(low=0, high=len(children), size=1)[0]
        logging.debug(f"Simulating game from move to: {moves[rand_idx]}")
        selected_child = children[rand_idx]
        return selected_child

    def simulation(self, node: TreeNode) -> TreeNode:
        self.cur_n += 1
        this_game: TicTacToe = self.tree[node].game_state.copy()
        player = deepcopy(self.tree[node].player)

        winner = this_game.evaluate_game()
        moves = this_game.get_possible_moves()

        while len(moves) != 0:
            # Random strategy of move choice
            x, y = random.choice(moves)
            this_game.move(player=player, x=x, y=y)

            # Check if the game already has a winner
            winner = this_game.evaluate_game()
            if winner != states.EMPTY:
                break

            # switch player
            player = states.CROSS if player == states.CIRCLE else states.CIRCLE

            # get possible moves
            moves = this_game.get_possible_moves()
        logging.debug(f"Simulation ended:\n{str(this_game)}")
        winner = this_game.evaluate_game()  # check the winner
        return winner

    def backpropagation(self, child_node_id: int, winner: int):
        player = self.tree[(0,)].player
        if winner == player:  # you won
            reward = 1
        elif winner == states.EMPTY:  # draw, no winner
            reward = 0
        else:
            reward = -1  # you loose
        logging.debug(f"Simulation reward: {reward}")
        node_id = child_node_id

        while True:
            self.tree[node_id].n += 1
            self.tree[node_id].w += reward
            self.tree[node_id].q += self.tree[node_id].w / self.tree[node_id].n
            parent_id = self.tree[node_id].parent
            if parent_id == (0,):
                self.tree[parent_id].n += 1
                self.tree[parent_id].w += reward
                self.tree[parent_id].q += (
                    self.tree[parent_id].w / self.tree[parent_id].n
                )
                break
            else:
                node_id = parent_id

    def choose_best_action(self) -> Tuple[int, int]:
        """ Select best action using q values """
        first_level_leafs = self.tree[(0,)].children
        logging.debug(f"first_level_leafs: {first_level_leafs}")
        Q_values = [self.tree[node].q for node in first_level_leafs]
        logging.debug(f"Q_values: {Q_values}")
        best_action_id = np.argmax(Q_values)
        best_leaf = first_level_leafs[best_action_id]

        best_move = self.tree[best_leaf].move
        logging.debug(f"Best move: {best_move}")
        best_q = Q_values[best_action_id]
        return best_move, best_q

    def run(self):
        for _ in tqdm(range(self.n_iters)):
            best_node_id = self.selection()
            logging.debug(f"Selected node: {best_node_id}")
            new_leaf_id = self.expansion(best_node_id)
            logging.debug(f"Expanded node: {new_leaf_id}")
            winner = self.simulation(new_leaf_id)
            logging.debug(f"Simulation winner: {states.translate(winner)}")
            self.backpropagation(new_leaf_id, winner=winner)
            logging.debug(f"Backpropagation done!")
            Q_values = [
                (self.tree[node].move, self.tree[node].q)
                for node in self.tree[(0,)].children
            ]
            logging.debug(f"Q_values: {Q_values}")

        best_action, best_q = self.choose_best_action()
        logging.debug(f"Best: action={best_action}, q={best_q}")
        return best_action
