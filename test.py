from main import main
import logging


logging.basicConfig(level=logging.DEBUG, format="%(message)s", filename="main.log")

main(n_rollouts=10, ini_game="assignment")
