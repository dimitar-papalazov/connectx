from dqn.dqn import DQNAgent
from ddqn.ddqn import DDQNAgent
from ppo.ppo import PPOAgent
import numpy as np
from kaggle_environments import make, evaluate


def dqn(obs, config):
    input_dimensions = config.rows * config.columns
    n_actions = config.columns
    agent = DQNAgent(gamma=0.99, epsilon=0.0, alpha=0.0005, input_dimensions=input_dimensions, n_actions=n_actions,
                     memory_size=1000000, batch_size=128, epsilon_end=0.0)
    agent.load_model()
    col = agent.choose_action(obs)
    return int(col)


def ddqn(obs, config):
    input_dimensions = config.rows * config.columns
    n_actions = config.columns
    agent = DDQNAgent(gamma=0.99, epsilon=0.0, alpha=0.0005, input_dimensions=input_dimensions, n_actions=n_actions,
                      memory_size=1000000, batch_size=128, epsilon_end=0.0)
    agent.load_model()
    col = agent.choose_action(obs)
    return int(col)


def ppo(obs, config):
    input_dimensions = config.rows * config.columns
    n_actions = config.columns
    agent = PPOAgent(n_actions=n_actions, input_dimensions=input_dimensions)
    agent.load_models()
    col = agent.choose_action(obs)[0]
    return int(col)


def get_win_percentages(agent1, agent2, n_rounds=100):
    outcomes = evaluate("connectx", [agent1, agent2], num_episodes=n_rounds)
    print("Agent 1 Win Percentage:", np.round(outcomes.count([1, -1]) / len(outcomes), 2))
    print("Agent 2 Win Percentage:", np.round(outcomes.count([-1, 1]) / len(outcomes), 2))
    print("Number of Invalid Plays by Agent 1:", outcomes.count([None, 0]))
    print("Number of Invalid Plays by Agent 2:", outcomes.count([0, None]))


if __name__ == '__main__':
    env = make('connectx', debug=True)
    # get_win_percentages(dqn, "negamax")
    # get_win_percentages(ddqn, "negamax")
    get_win_percentages(ppo, "negamax")
