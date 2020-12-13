# %%
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np

from collections import defaultdict


for f, plot, n in [
    ("data.assignment.big.pickle", "Assignment_game.conv.big.png", "Assignment"),
    ("data.big.pickle", "Empty_game.conv.big.png", "Empty"),
]:
    with open(f, "rb") as fd:
        data = pkl.load(fd)
    win_rate = defaultdict(list)
    for k, v in data.items():
        rolls, _ = k
        win_rate[rolls].append(v["win"])
    ks = []
    means = []
    for k, v in win_rate.items():
        ks.append(k)
        means.append(np.mean(v))
    experiments = len(win_rate[ks[0]])
    fig, ax = plt.subplots()
    ax.plot(np.dot(100, means), "-o")
    ax.set_xticks(np.arange(0, len(ks)))
    ax.set_xticklabels(ks)
    ax.set_title(f"{n} game MCTS Convergence | {experiments} experiments")
    ax.set_xlabel("N-rollouts")
    ax.set_ylabel("% wins")
    print(plot)
    fig.savefig(plot)

# %%
