from main import main
import logging
from tqdm import tqdm

import pickle

logging.basicConfig(level=logging.WARNING, format="%(message)s")

results = dict()

for fname, ini in [
    ("data.assignment.big.pickle", "assignment"),
    ("data.big.pickle", "empty"),
]:
    for i in tqdm(range(100), desc="Experiments"):
        for rollouts in tqdm(
            [5, 10, 20, 50, 100, 200, 300, 500, 1000, 2000, 5000, 10000],
            desc="Rollouts",
        ):
            win, win_probs, q_values = main(n_rollouts=rollouts, ini_game=ini)
            logging.info(f"{rollouts}, {i}, {win}")
            results[(rollouts, i)] = {
                "win": win,
                "probs": win_probs,
                "q_vals": q_values,
            }
            with open(fname, "wb") as fp:
                pickle.dump(results, fp)

            logging.info("Saved")
