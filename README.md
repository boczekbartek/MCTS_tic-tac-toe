# MCTS - tic-tac-toe
tic-tac-toe Monte Carlo Tree Search

## Instsallation
How to install package depenencies.
### Prerequisties
* Python >= 3.7

### Install Python packages
``` bash
pip install -r requirements.txt
```

## Usage
* `--n-rollouts` is number of rollouts per Monte Carlo Tree Search interation (*aka* computational budget)
* `--ini-game` *empty* or *assignment*
* `--verbose` optional flag to increase number of logs

``` bash
python  mcts_ttt.py --n-rollouts N_ROLLOUTS --ini-game {empty,assignment} [--verbose]
```

For example:
bash
```
python mcts_ttt.py --n-rollouts 10 --ini-game assignment
```

Possible output
```
Initial game:
_ o x
_ x o
o _ _
Player x moved to (0,0) with MCTS
x o x
_ x o
o _ _
Player o moved to (2,1) randomly
x o x
_ x o
o o _
Player x moved to (2,2) with MCTS
x o x
_ x o
o o x
Player o moved to (1,0) randomly
x o x
o x o
o o x
Winner: x
```

## Bulk testing
To run the script **N** times you can use the `test.sh` script. The script:
* groups results and count them
* count execution time

### Usage
```bash
test.sh LOOPS N_ROLLOUTS
```

### Example
```
./test.sh 100 10
```

Possible output
```
Running mcts_ttt.py 100 times
   4 Winner: _
   2 Winner: o
  94 Winner: x

real    0m32.985s
user    0m20.224s
sys     0m6.284s
```