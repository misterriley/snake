

'''
Created on Sep 27, 2021

@author: bleem
'''
from collections import deque
import math
import os
import pickle

from numpy import random
from numpy.linalg import norm
from termcolor import colored

from SnakeMain import SnakeEnvironment
import numpy as np


def exp_raw_moment(a, tau, index):
    return math.factorial(index) * a * (tau ** (index + 1))


def exp_raw_moment_consts(a, b, index):
    return exp_raw_moment(a, 1. / b, index)


# average firing rate of neurons in Hz
K = 100

# constants for an asymmetric exponential STDP window based on Bi and Poo (1998)
W_constants = {
    "A-" :.5 / 60,  # max increase in weight for a causal spike pair
    "tau-" : .02,  # time constant for the left side of the window
    "A+" :-.2 / 60,  # max decrease in weight for an acausal spike pair
    "tau+" : .02  # time constant for the right side of the window
    }

# constants for an exponential EPSP
e_constants = {
    "A": 0.002,  # https://journals.physiology.org/doi/full/10.1152/jn.00942.2007
    "tau": .05  # https://physoc.onlinelibrary.wiley.com/doi/pdf/10.1113/jphysiol.1959.sp006159
    }

# raw moments of the STDP kernel
W_mu0 = exp_raw_moment(W_constants["A-"], W_constants["tau-"], 0) + exp_raw_moment(W_constants["A+"], W_constants["tau+"], 0)
W_mu1 = exp_raw_moment(W_constants["A-"], W_constants["tau-"], 1) + exp_raw_moment(W_constants["A+"], W_constants["tau+"], 1)
W_mu2 = exp_raw_moment(W_constants["A-"], W_constants["tau-"], 2) + exp_raw_moment(W_constants["A+"], W_constants["tau+"], 2)

# raw moments of EPSP
e_mu0 = exp_raw_moment(e_constants["A"], e_constants["tau"], 0)
e_mu1 = exp_raw_moment(e_constants["A"], e_constants["tau"], 1)
e_mu2 = exp_raw_moment(e_constants["A"], e_constants["tau"], 2)

# raw moments of the convolution between W and e
We_mu0 = W_mu0 * e_mu0
We_mu1 = W_mu1 * e_mu0 + W_mu0 * e_mu1
We_mu2 = W_mu2 * e_mu0 + 2 * W_mu1 * e_mu1 + W_mu0 * e_mu2

W_times_e_nt_mu0 = exp_raw_moment_consts(W_constants["A-"] * e_constants["A"], 1 / W_constants["tau-"] + 1 / e_constants["tau"], 0)

DP_MOMENTUM = .997
PRINT_EVERY = 100
SAVE_EVERY = 1000
TARGET_OUTPUT_VAR = (K / 2) ** 2

WEIGHT_MOMENTUM = .997


def save(obj):
    pickle.dump(obj, open("scf_save.dat", "wb"))


def load():
    if os.path.exists("scf_save.dat"):
        return pickle.load(open("scf_save.dat", "rb"))
    return None


class SCFilter:

    def __init__(self,
                 shape, n_filters):
        self.shape = shape
        self.n_filters = n_filters
        self.filters_flattened = np.reshape(np.asarray([random.rand(*shape) for _ in range(self.n_filters)]), (self.n_filters, -1))

        self.orthonorm_weights()

        self.last_input = None
        self.last_output = None
        self.num_filters_performed = 0
        self.mean_consecutive_dps = [AdamEMA(1) for _ in  range(n_filters)]

        self.gamma = 1000
        self.beta = .9
        self.burnin = 100
        self.time_step_s = 1. / 3  # saccade frequency
        self.replay_memory = 50000

        self.storage = deque(maxlen = self.replay_memory)
        self.replay_size = 2 ** 6
        self.last_delta = np.zeros(self.filters_flattened.shape)

        self.mean_angle_delta = AdamEMA(0)
        self.last_raw_delta = np.zeros(self.filters_flattened.shape)
        self.inputs_averaged = AdamEMA(np.zeros(self.filters_flattened[0].shape))
        self.output_gain_logs = AdamEMA(np.ones((self.n_filters,)) * np.log(TARGET_OUTPUT_VAR), gamma = 0.0001)
        self.outputs_averaged = AdamEMA(np.ones((self.n_filters,)))
        self.output_vars = AdamEMA(np.ones((self.n_filters,)) * TARGET_OUTPUT_VAR, gamma = 0.001 * K / 2)
        self.consecutive_gradient_dp = [AdamEMA(1) for _ in range(n_filters)]

    def orthonorm_weights(self):

        for i in range(self.n_filters):
            for j in range(i):
                self.filters_flattened[i] -= np.dot(self.filters_flattened[i], self.filters_flattened[j]) / norm(self.filters_flattened[j]) ** 2 * self.filters_flattened[j]

        for i in range(self.n_filters):
            self.filters_flattened[i] /= norm(self.filters_flattened[i])

    def learn_filter(self, input_data):

        assert input_data.shape == self.shape

        input_flattened = input_data.flatten()

        if self.last_input is not None:
            self.storage.append(MemoryData(input_flattened, self.last_input))
            self.num_filters_performed += 1

        self.last_input = input_flattened
        self.inputs_averaged.add_to_average(input_flattened)

        if self.num_filters_performed >= self.burnin:

            self.replay()

    def print_angle(self, v1, v2):
        print(self.get_angle(v1, v2))

    def get_angle(self, v1, v2):
        dp = np.dot(self.get_unit(v1).flatten(), self.get_unit(v2).flatten())
        return np.arccos(dp) * 180 / np.pi

    def replay(self):
        minibatch_idx = np.random.choice(a = len(self.storage), size = self.replay_size)
        old_weights = np.copy(self.filters_flattened)
        raw_delta_w = np.zeros(self.filters_flattened.shape)
        outputs_sum = np.zeros((self.n_filters,))
        outputs_sq = np.zeros((self.n_filters,))

        for mb_index in minibatch_idx:
            this_delta_w = np.zeros(self.filters_flattened.shape)

            # this_delta_w += (K ** 2) * W_mu0
            this_delta_w += K * W_times_e_nt_mu0 * self.filters_flattened

            cur_input = self.storage[mb_index].this_input - self.inputs_averaged.get_value()
            prev_input = self.storage[mb_index].last_input - self.inputs_averaged.get_value()

            cur_outputs = self.get_outputs(cur_input)
            prev_outputs = self.get_outputs(prev_input)

            outputs_sum += cur_outputs
            outputs_sq += np.square(cur_outputs)

            dinput_dt = (cur_input - prev_input) / self.time_step_s
            doutput_dt = (cur_outputs - prev_outputs) / self.time_step_s

            this_delta_w += We_mu0 * np.outer(cur_outputs, cur_input)
            this_delta_w += .5 * We_mu1 * (np.outer(cur_outputs, dinput_dt) - np.outer(doutput_dt, cur_input))
            this_delta_w += -.5 * We_mu2 * np.outer(doutput_dt, dinput_dt)

            raw_delta_w += this_delta_w

        sample_mean = outputs_sum / len(minibatch_idx)
        self.outputs_averaged.add_to_average(sample_mean)
        sample_var = (outputs_sq / len(minibatch_idx) - np.square(sample_mean)) * len(minibatch_idx) / (len(minibatch_idx) - 1)
        self.output_vars.add_to_average(sample_var)

        if all(self.output_vars.get_value() > 0):
            log_loss = (np.log(TARGET_OUTPUT_VAR) - np.log(sample_var))
            self.output_gain_logs.apply_gradient(log_loss)

        raw_delta_w = raw_delta_w / self.replay_size
        delta_w = raw_delta_w * (1 - self.beta) + self.last_delta * self.beta
        unnormed_filters = self.filters_flattened + self.gamma * delta_w

        for i in range(self.n_filters):
            for j in range(i):
                delta_w[i] -= 1 / (self.gamma) * np.dot(unnormed_filters[i], unnormed_filters[j]) / norm(unnormed_filters[j]) ** 2 * unnormed_filters[j]

        # perp_delta = delta_w
        perp_filters = self.filters_flattened + self.gamma * delta_w

        for i in range(self.n_filters):
            delta_w[i] += 1 / (self.gamma) * (1. - norm(perp_filters[i])) * perp_filters[i]

        self.filters_flattened += self.gamma * delta_w
        angle_change = self.get_angle(self.filters_flattened, old_weights)
        self.mean_angle_delta.add_to_average(angle_change)

        if self.num_filters_performed % PRINT_EVERY == 0:
            print("")
            print("mean angle delta: " + str(self.mean_angle_delta.get_value()))
            print("gamma: " + str(self.gamma))
            print("norm(momentum): " + str(norm(self.last_delta)))
            print("avg outputs: " + np.array_str(self.outputs_averaged.get_value(),
                                                precision = 3,
                                                suppress_small = True))
            print("gains: " + np.array_str(np.exp(self.output_gain_logs.value),
                                                precision = 3,
                                                suppress_small = True))
            print("variances: " + np.array_str(self.output_vars.get_value(),
                                                precision = 3,
                                                suppress_small = True))

        if self.num_filters_performed % SAVE_EVERY == 0:
            print("saving ...")
            save(self)
            print("saved")

        for filter_index in range(self.n_filters):
            unit_new = self.get_unit(self.filters_flattened[filter_index])
            unit_old = self.get_unit(old_weights[filter_index])
            dp1 = min(self.get_dot(unit_new, unit_old), 1)
            dp2 = min(self.get_dot(self.get_unit(self.last_raw_delta[filter_index]), self.get_unit(self.last_delta[filter_index])), 1) if self.last_raw_delta is not None else 0
            self.mean_consecutive_dps[filter_index].add_to_average(dp2)

            if self.last_raw_delta is not None:
                perp_raw_delta = raw_delta_w[filter_index] - self.get_dot(raw_delta_w[filter_index], self.filters_flattened[filter_index]) * self.filters_flattened[filter_index]
                last_perp_raw_delta = self.last_raw_delta[filter_index] - self.get_dot(raw_delta_w[filter_index], self.filters_flattened[filter_index]) * self.filters_flattened[filter_index]
                dot = self.get_dot(self.get_unit(perp_raw_delta), self.get_unit(last_perp_raw_delta))

                self.consecutive_gradient_dp[filter_index].add_to_average(dot)

            if self.num_filters_performed % PRINT_EVERY == 0:

                print("filter {} angle delta: {}".format(filter_index, np.arccos(dp1) / np.pi * 180))
                print("avg raw/corrected gradient dp (converges at 0): {}".format(self.mean_consecutive_dps[filter_index].get_value()))
                print("avg consecutive gradient dp (converges at 0): {}".format(self.consecutive_gradient_dp[filter_index].get_value()))
        self.last_delta = delta_w
        self.last_raw_delta = raw_delta_w

    def get_dot(self, v1, v2):
        return np.dot(v1, v2) if v1 is not None and v2 is not None else 0

    def get_unit(self, value):
        length = norm(value)
        return value / length if length != 0 else None

    def get_demeaned_outputs(self, input_data):
        return self.get_outputs(input_data) - self.mean_outputs.values

    def get_outputs(self, input_data):
        raw_output = np.matmul(input_data, np.transpose(self.filters_flattened))
        return np.multiply(raw_output, np.exp(self.output_gain_logs.value))


class MemoryData():

    def __init__(self, this_input, last_input):
        self.this_input = np.copy(this_input)
        self.last_input = np.copy(last_input)


class OUProcess():

    def __init__(self, beta, mu, sigma, dt, x_0 = None):
        assert beta.shape == sigma.shape
        assert beta.shape[0] == beta.shape[1] == mu.shape[0]

        self.beta = beta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        if x_0 is None:
            self.x = np.copy(self.mu)
        else:
            assert x_0.shape == mu.shape
            self.x = x_0

        self.w_mean = np.zeros(mu.shape)
        self.w_covar = self.dt * np.identity(mu.shape[0], dtype = float)

    def step(self):
        dW = random.multivariate_normal(self.w_mean, self.w_covar)
        dx = self.dt * np.matmul(self.beta, self.mu - self.x) + np.matmul(self.sigma, dW)
        self.x += dx
        return self.x


class AdamEMA():

    def __init__(self, value, beta1 = .9, beta2 = .999, epsilon = 10 ** -8, gamma = .001):
        self.value = value
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.gamma = gamma
        self.m = np.zeros(self.value.shape) if type(self.value) == np.ndarray else 0
        self.v = np.zeros(self.value.shape) if type(self.value) == np.ndarray else 0
        self.steps = 0

    def apply_gradient(self, gradients):
        self.steps += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.square(gradients)
        m_hat = self.m / (1 - self.beta1 ** self.steps)
        v_hat = self.v / (1 - self.beta2 ** self.steps)
        self.value = self.value + self.gamma * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def add_to_average(self, value):

        gradients = value - self.value
        self.apply_gradient(gradients)

    def get_value(self):
        return self.value

    def get_var(self):
        return self.v / (1 - self.beta2 ** self.steps)


class EMA():

    def __init__(self, value, momentum = .9, sq_momentum = .99):
        self.value = value
        self.momentum = momentum
        self.sq_momentum = sq_momentum
        self.sq_value = value ** 2

    def add_to_average(self, value):
        if type(value) == np.ndarray or type(self.value) == np.ndarray:
            assert value.shape == self.value.shape
        self.value = self.momentum * self.value + (1 - self.momentum) * value
        self.sq_value = self.sq_momentum * self.sq_value + (1 - self.sq_momentum) * np.square(value)

    def get_value(self):
        return self.value

    def get_sq_value(self):
        return self.sq_value

    def get_var(self):
        return self.sq_value - np.square(self.value) + self.epsilon


class MomentumObj():

    def __init__(self, values, gamma_init = .1, beta = 0.9):
        self.num_steps = 1
        self.gamma_init = gamma_init
        self.beta = beta
        self.v = np.zeros(values.shape)
        self.values = values
        self.shape = self.values.shape

    def apply_gradient(self, gradient):
        assert gradient.shape == self.values.shape
        self.v = self.beta * self.v + (1 - self.beta) * gradient
        self.values += self.gamma() * self.v
        self.num_steps += 1

    def peek_v(self, gradient):
        return self.beta * self.v + (1 - self.beta) * gradient

    def peek_delta(self, gradient):
        v_temp = self.peek_v(gradient)
        return self.gamma() * v_temp

    def step_toward(self, values):
        gradient = values - self.values
        self.apply_gradient(gradient)

    def force_to_target(self, target):
        delta = target - self.values
        self.v = delta / self.gamma()
        self.values = target
        self.num_steps += 1

    def __str__(self):
        return str(self.values)

    def gamma(self):
        return self.gamma_init


def print_red(text):
    print(colored(text, "red"))


def print_weights(scfilter, weight_shape):
    for filter_index in range(scfilter.n_filters):
        print("filter {}".format(filter_index))
        weights = np.reshape(scfilter.filters_flattened[filter_index], weight_shape)
        head_weights = weights[0]
        body_weights = weights[1]
        food_weights = weights[2]
        oob_weights = weights[3]
        print("head")
        print(head_weights)
        print("body")
        print(body_weights)
        print("food")
        print(food_weights)
        print("oob")
        print(oob_weights)


def run_snake():
    game = SnakeEnvironment('computer')
    game.reset()
    scfilter = load()
    if scfilter is None:
        scfilter = SCFilter(shape = game.observation_space.shape, n_filters = 5)
    game_index = 0
    while True:
        if game.is_over():
            game.reset()
            game_index += 1
            if game_index % PRINT_EVERY == 0:
                print_weights(scfilter, game.observation_space.shape)
                pass
        else:
            action = game.action_space.sample()
            state, _, _, _ = game.step(action)
            scfilter.learn_filter(state)


def test_scf():
    scf = SCFilter((4,), 3)
    time_index = 0
    while True:
        time_index += 1
        time = time_index * .001
        w = 1  # * np.sin(time)
        x = 2  # *np.sin(2 * time)
        y = 3  # * np.sin(3 * time)
        z = 4 + 4 * np.sin(4 * time)
        sample = np.asarray([w, x, y, z]) + .001 * np.random.rand(4)
        # print(sample)
        scf.learn_filter(sample)

        if time_index % PRINT_EVERY == 0:
            print(scf.filters_flattened)


if __name__ == "__main__":
    run_snake()

