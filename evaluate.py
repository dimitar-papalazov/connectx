from dqn.dqn import DQNAgent
from ddqn.ddqn import DDQNAgent
import numpy as np
from kaggle_environments import make, evaluate
import random


def dqn(obs, config):
    input_dimensions = config.rows * config.columns
    n_actions = config.columns
    agent = DQNAgent(gamma=0.99, epsilon=0.0, alpha=0.0005, input_dimensions=input_dimensions, n_actions=n_actions,
                     memory_size=1000000, batch_size=16, epsilon_end=0.0)
    agent.load_model()
    col = agent.choose_action(obs)
    is_valid = (obs['board'][int(col)] == 0)
    if is_valid:
        return int(col)
    else:
        return random.choice([col for col in range(config.columns) if obs.board[int(col)] == 0])


def ddqn(obs, config):
    input_dimensions = config.rows * config.columns
    n_actions = config.columns
    agent = DDQNAgent(gamma=0.99, epsilon=0.0, alpha=0.0005, input_dimensions=input_dimensions, n_actions=n_actions,
                      memory_size=1000000, batch_size=16, epsilon_end=0.0)
    agent.load_model()
    col = agent.choose_action(obs)
    is_valid = (obs['board'][int(col)] == 0)
    if is_valid:
        return int(col)
    else:
        return random.choice([col for col in range(config.columns) if obs.board[int(col)] == 0])


def get_win_percentages(agent1, agent2, n_rounds=100):
    outcomes = evaluate("connectx", [agent1, agent2], num_episodes=n_rounds)
    print("Agent 1 Win Percentage:", np.round(outcomes.count([1, -1]) / len(outcomes), 2))
    print("Agent 2 Win Percentage:", np.round(outcomes.count([-1, 1]) / len(outcomes), 2))


if __name__ == '__main__':
    env = make('connectx', debug=True)
    print("DQN vs. Random")
    get_win_percentages(dqn, "random")
    print("DDQN vs. Random")
    get_win_percentages(ddqn, "random")
    print("DQN vs. Negamax")
    get_win_percentages(dqn, "negamax")
    print("DDQN vs. Negamax")
    get_win_percentages(ddqn, "negamax")
