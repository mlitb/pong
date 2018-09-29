"""
    Reinforcement Learning for Pong!
"""

import os
import gym
import argparse
import numpy as np
import pickle as pkl

from typing import Dict, List


# Type shorthands
Model = Dict[np.ndarray, np.ndarray]
EpisodeBuffer = Dict[str, List]
Gradient = Dict[str, np.ndarray]


def preprocess(frame: np.ndarray) -> np.ndarray:
    """
        Preprocess 210x160x3 uint8 frame into 6400 (80x80) 1D float vector.
    """
    frame = frame[35:195] # crop
    frame = frame[::2,::2,0] # downsample by factor of 2
    frame[frame == 144] = 0 # erase background (background type 1)
    frame[frame == 109] = 0 # erase background (background type 2)
    frame[frame != 0] = 1 # everything else (paddles, ball) just set to 1
    return frame.astype(np.float).ravel()


def relu(x: np.ndarray) -> np.ndarray:
    x[x < 0] = 0
    return x


def relu_prime(x: np.ndarray) -> np.ndarray:
    x[x > 0] = 1
    return x


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime(x: np.ndarray) -> np.ndarray:
    s = sigmoid(x)
    return s * (1.0 - s)


def forward(x: np.ndarray, model: Model, episode_buffer: EpisodeBuffer) -> float:
    """
        Do a forward pass to get the probability of moving the paddle up.
    """
    ph = np.dot(model['wh'], x)
    h = relu(ph)
    py = np.dot(model['wo'], h)
    y = sigmoid(py)
    episode_buffer['x'].append(x)
    episode_buffer['ph'].append(ph)
    episode_buffer['h'].append(h)
    episode_buffer['py'].append(py)
    episode_buffer['y'].append(y)
    return y


def backward(model: Model, episode_buffer: EpisodeBuffer, episode_reward: np.ndarray) -> Gradient:
    """
        Do a backward pass to get the gradient of the network weights.
    """
    y_true = np.vstack(episode_buffer['y_true'])
    y = np.vstack(episode_buffer['y'])
    py = np.vstack(episode_buffer['py'])
    h = np.vstack(episode_buffer['h'])
    ph = np.vstack(episode_buffer['ph'])
    x = np.vstack(episode_buffer['x'])

    grad_y = y - y_true
    adv_grad_y = grad_y * episode_reward # advantage based on reward
    grad_py = y * (1.0 - y) * adv_grad_y # sigmoid prime
    grad_wo = np.dot(grad_py.T, h)
    
    # grad_logp = y - y_true
    # adv_grad_logp = grad_logp * episode_reward # advantage based on reward
    # grad_wo = np.dot(adv_grad_logp.T, h)

    # print('grad_y', grad_y.shape)
    # print('episode_reward', episode_reward.shape)
    # print('adv_grad_y', adv_grad_y.shape)
    # print('grad_py', grad_py.shape)
    # print('grad_wo', grad_wo.shape)
    # print('wo', model['wo'].shape)
    # print()

    grad_h = np.dot(grad_py, model['wo'])
    grad_ph = relu_prime(ph) * grad_h
    grad_wh = np.dot(grad_ph.T, x)
    
    # grad_h = np.dot(adv_grad_logp, model['wo'])
    # grad_ph = relu_prime(ph) * grad_h
    # grad_wh = np.dot(grad_ph.T, x)

    # print('grad_y', grad_y.shape)
    # print('grad_py', grad_py.shape)
    # print('grad_h', grad_h.shape)
    # print('grad_ph', grad_ph.shape)
    # print('grad_wh', grad_wh.shape)
    # print('wh', model['wh'].shape)
    # print()

    return {'wh': grad_wh, 'wo': grad_wo}


def normal_discounted_reward(episode_buffer: EpisodeBuffer, discount_factor: float) -> float:
    """
        Calculate the normalized and discounted reward for the current episode.
    """
    reward = episode_buffer['reward']
    discounted_reward = np.zeros((len(reward), 1))
    future_reward = 0
    for i in range(len(reward) - 1, -1, -1):
        if reward[i] != 0: # reset future reward after each score
            future_reward = 0
        discounted_reward[i][0] = reward[i] + discount_factor * future_reward
        future_reward = discounted_reward[i][0]
    discounted_reward -= np.mean(discounted_reward)
    discounted_reward /= np.std(discounted_reward)
    return discounted_reward


def main(load_fname: str, save_dir: str, render: bool) -> None:
    """
        Main training loop.
    """
    batch_size = 10
    input_layer_size = 6400
    hidden_layer_size = 200
    learning_rate = 1e-3
    discount_factor = .99
    rmsprop_decay = 0.9
    rmsprop_smoothing = 1e-5

    if load_fname is not None:
        saved = pkl.load(open(load_fname, 'rb'))
        model = saved['model']
        moving_grad_rms = saved['moving_grad_rms']
        episode_number = saved['episode_number']
        print('Resuming saved model in \'{}\'.'.format(load_fname))
    else:
        model = {
            'wh': np.random.randn(hidden_layer_size, input_layer_size) / np.sqrt(input_layer_size),
            'wo': np.random.randn(1, hidden_layer_size) / np.sqrt(hidden_layer_size),
        }
        moving_grad_rms = {
            'wh': np.zeros_like(model['wh']),
            'wo': np.zeros_like(model['wo']),
        }
        episode_number = 0

    batch_gradient_buffer = {
        'wh': np.zeros_like(model['wh']),
        'wo': np.zeros_like(model['wo']),
    }
    batch_rewards = []

    env = gym.make('Pong-v0')
    while True:
        observation = env.reset()
        prev_frame = np.zeros(input_layer_size)
        episode_done = False
        timestep = 0
        
        episode_buffer = {
            'x': [], # input vector
            'ph': [], # product of hidden layer
            'h': [], # activation of hidden layer
            'py': [], # product of output layer
            'y': [], # activation of output layer (prob of moving up)
            'y_true': [], # fake label
            'reward': [] # rewards
        }

        while not episode_done:
            if render:
                env.render()

            # generate input vector
            frame = preprocess(observation)
            x = frame - prev_frame
            prev_frame = frame

            # forward pass
            y = forward(x, model, episode_buffer)
            action, y_true = (2, 1.0) if np.random.uniform() < y else (5, 0.0)
            episode_buffer['y_true'].append(y_true)
            
            # perform action and get new observation
            observation, reward, episode_done, info = env.step(action)
            episode_buffer['reward'].append(reward)
            timestep += 1
            
            if episode_done:
                # backward pass
                episode_reward = normal_discounted_reward(episode_buffer, discount_factor)
                gradient = backward(model, episode_buffer, episode_reward)
                for key in model:
                    batch_gradient_buffer[key] += gradient[key]

                # bookeeping
                batch_rewards.append(sum(episode_buffer['reward']))
                episode_number += 1

                # training info
                # print('Episode: {}, rewards: {}'.format(episode_number, sum(episode_buffer['reward'])))

                # parameter update (rmsprop)
                if episode_number % batch_size == 0:
                    for key in model:
                        moving_grad_rms[key] = rmsprop_decay * moving_grad_rms[key] + \
                                (1 - rmsprop_decay) * (batch_gradient_buffer[key] ** 2)
                        model[key] -= batch_gradient_buffer[key] * learning_rate / \
                                (np.sqrt(moving_grad_rms[key]) + rmsprop_smoothing)
                        batch_gradient_buffer[key] = np.zeros_like(model[key])
                    
                    # training info
                    print('Batch: {}, avg_episode_reward: {}'.format(episode_number // batch_size, 
                            sum(batch_rewards) / len(batch_rewards)))
                    batch_rewards = []

                # save model
                if episode_number % 50 == 0 and save_dir is not None:
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_fname = os.path.join(save_dir, 'save_{}.pkl'.format(episode_number))
                    pkl.dump({'model': model, 'moving_grad_rms': moving_grad_rms, 
                            'episode_number': episode_number}, open(save_fname, 'wb'))
                    print('Model saved to \'{}\'!'.format(save_fname))

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an RL agent to play the mighty game of Pong.')
    parser.add_argument('-l', '--load', action="store", default=None, help='path to the saved model to load from')
    parser.add_argument('-s', '--save', action="store", default=None, help='path to the folder to save model')
    parser.add_argument('-r', '--render', action="store_true", default=False, help='whether to render the environment or not')
    args = parser.parse_args()
    main(load_fname=args.load, save_dir=args.save, render=args.render)