from dqn.dqn import DQNAgent
from ddqn.ddqn import DDQNAgent
from ppo.ppo import PPOAgent
import numpy as np
from kaggle_environments import make


def train_dqn():
    env = make('connectx', debug=True)
    n_games = 10000
    config = env.configuration
    input_dimensions = config.rows * config.columns
    n_actions = config.columns
    agent = DQNAgent(gamma=0.99, epsilon=1.0, alpha=0.0005, input_dimensions=input_dimensions, n_actions=n_actions,
                     memory_size=1000000, batch_size=16, epsilon_end=0.01)
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
                      memory_size=1000000, batch_size=16, epsilon_end=0.01)
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


def train_ppo():
    env = make('connectx', debug=True)
    N = 20
    batch_size = 16
    n_epochs = 20
    alpha = 0.0003
    episodes = 10000
    config = env.configuration
    input_dimensions = config.rows * config.columns
    n_actions = config.columns
    agent = PPOAgent(n_actions=n_actions, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs,
                     input_dimensions=input_dimensions)

    trainer = env.train([None, 'random'])

    best_score = 0
    scores = []

    learn_iterations = 0
    n_steps = 0

    for i in range(episodes):
        done = False
        score = 0
        observation = trainer.reset()
        while not done:
            action, probabilities, values = agent.choose_action(observation)
            new_observation, reward, done, info = trainer.step(action)
            n_steps += 1
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
            agent.remember(observation, action, probabilities, values, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iterations += 1
            observation = new_observation
        scores.append(score)
        average_score = np.mean(scores[-100:])
        if average_score > best_score:
            best_score = average_score
            agent.save_models()
        print('epsiode:', i + 1, "score: %.2f" % score, 'average score: %.2f' % average_score, 'time_steps', n_steps,
              'learning_steps', learn_iterations)


if __name__ == '__main__':
    # train_dqn()
    # train_ddqn()
    train_ppo()
