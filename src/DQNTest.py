import shutil
import tempfile

import gym

from DQN import DQNAgent
from SnakeMain import SnakeEnvironment

SAVE_EVERY = 50


def train(environment, model_name = None, load_from_file = False):
    tdir = tempfile.mkdtemp()
    env = gym.wrappers.Monitor(environment, tdir, force = True)
    agent = DQNAgent(env, trained_model = model_name if load_from_file else None)
    EPISODES = 50000
    action_count = 0
    for episode in range(EPISODES):
        state, reward, done = env.reset(), 0.0, False
        action = agent.action(state, reward, done, episode, action_count)
        action_count += 1
        while not done:
            # env.render()
            next_state, reward, done, _ = env.step(action)
            agent.store(state, action, reward, next_state, done)
            state = next_state
            action = agent.action(state, reward, done, episode, action_count)
            action_count += 1
        if model_name and (episode == EPISODES - 1 or episode % SAVE_EVERY == 0):
            try:
                agent.save_model(filename = model_name)
                print("saved")
            except:
                print("borked on save")
    env.close()
    shutil.rmtree(tdir)


def run(environment, model_name):
    tdir = tempfile.mkdtemp()
    env = gym.wrappers.Monitor(environment, tdir, force = True)
    agent = DQNAgent(env, trained_model = model_name)
    EPISODES = 100
    for episode in range(EPISODES):
        state, reward, done = env.reset(), 0.0, False
        action = agent.action(state, reward, done, episode, 0, training = False)
        while not done:
            # env.render()
            next_state, reward, done, _ = env.step(action)
            state = next_state
            action = agent.action(state, reward, done, episode, training = False)
    env.close()
    shutil.rmtree(tdir)


if __name__ == "__main__":
    # run(SnakeEnvironment(mode = "computer") , model_name = "Snake")
    train(SnakeEnvironment(mode = "computer") , model_name = "Snake", load_from_file = False)
