#!/usr/bin/env bash
# Test mcts_ttt.py file by running it 'LOOPS' times
# Count number of occurences of results.
# usage: ./test.sh LOOPS N_ROLLOUTS

LOOPS=$1
N_ROLLOUTS=$2

if [[ $# -ne 2 ]]; then echo "usage: $0 LOOPS N_ROLLOUTS" && exit 1; fi

SCRIPT=mcts_ttt.py

>&2 echo "Running $SCRIPT $LOOPS times"
time for i in $(seq $LOOPS); do 
    python $SCRIPT --n-rollouts $N_ROLLOUTS --ini-game empty 2>/dev/null; 
done | sort | uniq -c