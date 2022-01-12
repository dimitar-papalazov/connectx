from dqn.dqn import DQNAgent
from ddqn.ddqn import DDQNAgent
import numpy as np
from kaggle_environments import make


def train_dqn():
    env = make('connectx', debug=True)
    n_games = 10000
    config = env.configuration
    input_dimensions = config.rows * config.columns
    n_actions = config.columns
    agent = DQNAgent(gamma=0.99, epsilon=1.0, alpha=0.0005, input_dimensions=input_dimensions, n_actions=n_actions,
                     memory_size=1000000, batch_size=16)
    trainer = env.train([None, 'random'])
    scores = []
    epsilon_history = []

    for i in range(n_games):
        done = False
        score = 0
        observation = trainer.reset()
        while not done:
            action = agent.choose_action(observation)
            new_observation, reward, done, info = trainer.step(int(action))
            if done:
                if reward == 1:  # Won
                    reward = 10
                elif reward == 0:  # Lost
                    reward = -10
                elif reward is None:  # Invalid Move
                    reward = -100
                else:  # Draw
                    reward = 1
            else:
                reward = (1 / 42)
            score += reward
            agent.remember(observation, action, reward, new_observation, done)
            observation = new_observation
            agent.learn()
        epsilon_history.append(agent.epsilon)
        scores.append(score)
        avg_score = np.mean(scores[max(0, i - 100):(i + 1)])
        print('epsiode:', i + 1, "score: %.2f" % score, 'average score: %.2f' % avg_score)
        if i % 10 == 0 and i > 0:
            agent.save_model()


def train_ddqn():
    env = make('connectx', debug=True)
    episodes = 10000
    config = env.configuration
    input_dimensions = config.rows * config.columns
    n_actions = config.columns
    agent = DDQNAgent(gamma=0.99, epsilon=1.0, alpha=0.0005, input_dimensions=input_dimensions, n_actions=n_actions,
                      memory_size=1000000, batch_size=16)
    trainer = env.train([None, 'random'])
    scores = []
    epsilon_history = []

    for i in range(episodes):
        done = False
        score = 0
        observation = trainer.reset()
        while not done:
            action = agent.choose_action(observation)
            new_observation, reward, done, info = trainer.step(int(action))
            if done:
                if reward == 1:  # Won
                    reward = 10
                elif reward == 0:  # Lost
                    reward = -10
                elif reward is None:  # Invalid Move
                    reward = -100
                else:  # Draw
                    reward = 1
            else:
                reward = (1 / 42)
            score += reward
            agent.remember(observation, action, reward, new_observation, done)
            observation = new_observation
            agent.learn()
        epsilon_history.append(agent.epsilon)
        scores.append(score)
        avg_score = np.mean(scores[max(0, i - 100):(i + 1)])
        print('epsiode:', i + 1, "score: %.2f" % score, 'average score: %.2f' % avg_score)
        if i % 10 == 0 and i > 0:
            agent.save_model()


if __name__ == '__main__':
    train_dqn()
    train_ddqn()
