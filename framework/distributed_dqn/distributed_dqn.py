import gym
import torch
import time
import os
import ray
import numpy as np

from tqdm import tqdm
from random import uniform, randint

import io
import base64
from IPython.display import HTML

from dqn_model import DQNModel
from dqn_model import _DQNModel
from memory_remote import ReplayBuffer_remote

import matplotlib.pyplot as plt
#%matplotlib inline
from custom_cartpole import CartPoleEnv

FloatTensor = torch.FloatTensor

########################################################################################################################################
########################################################################################################################################
# Set the Env name and action space for CartPole
ENV_NAME = 'CartPole-v0'
# Move left, Move right
ACTION_DICT = {
    "LEFT": 0,
    "RIGHT":1
}
# Register the environment
env_CartPole = CartPoleEnv()

# Set results saveing floder
result_floder = ENV_NAME
result_file = ENV_NAME + "/result_file_1.txt"
if not os.path.isdir(result_floder):
    os.mkdir(result_floder)

def plot_result(total_rewards ,learning_num, legend):
    print("\nLearning Performance:\n")
    episodes = []
    for i in range(len(total_rewards)):
        episodes.append(i * learning_num + 1)
        
    plt.figure(num = 1)
    fig, ax = plt.subplots()
    plt.plot(episodes, total_rewards)
    plt.title('performance')
    plt.legend(legend)
    plt.xlabel("Episodes")
    plt.ylabel("total rewards")
    plt.savefig("reward_4cv_4ev.png")
    #plt.show()

hyperparams_CartPole = {
    'epsilon_decay_steps' : 100000, 
    'final_epsilon' : 0.1,
    'batch_size' : 32, 
    'update_steps' : 10, 
    'memory_size' : 2000, 
    'beta' : 0.99, 
    'model_replace_freq' : 2000,
    'learning_rate' : 0.0003,
    'use_target_model': True
}
########################################################################################################################################
########################################################################################################################################
ray.shutdown()
ray.init(include_webui=False, ignore_reinit_error=True, redis_max_memory=500000000, object_store_memory=5000000000)

# Model Server
@ray.remote
class model_server(object):
    def __init__(self, env, memory, hyper_params, training_episodes, test_interval, action_space = len(ACTION_DICT)):
        
        self.env = env
        self.max_episode_steps = env._max_episode_steps
        
        """
            beta: The discounted factor of Q-value function
            (epsilon): The explore or exploit policy epsilon. 
            initial_epsilon: When the 'steps' is 0, the epsilon is initial_epsilon, 1
            final_epsilon: After the number of 'steps' reach 'epsilon_decay_steps', 
                The epsilon set to the 'final_epsilon' determinately.
            epsilon_decay_steps: The epsilon will decrease linearly along with the steps from 0 to 'epsilon_decay_steps'.
        """
        self.beta = hyper_params['beta']
        self.initial_epsilon = 1
        self.final_epsilon = hyper_params['final_epsilon']
        self.epsilon_decay_steps = hyper_params['epsilon_decay_steps']

        """
            episode: Record training episode
            steps: Add 1 when predicting an action
            learning: The trigger of agent learning. It is on while training agent. It is off while testing agent.
            action_space: The action space of the current environment, e.g 2.
        """
        self.episode = 0
        self.steps = 0
        self.best_reward = 0
        self.learning = True
        self.action_space = action_space

        """
            input_len The input length of the neural network. It equals to the length of the state vector.
            output_len: The output length of the neural network. It is equal to the action space.
            eval_model: The model for predicting action for the agent.
            target_model: The model for calculating Q-value of next_state to update 'eval_model'.
            use_target_model: Trigger for turn 'target_model' on/off
        """
        state = env.reset()
        input_len = len(state)
        output_len = action_space
        self.eval_model = DQNModel(input_len, output_len, learning_rate = hyper_params['learning_rate'])
        self.use_target_model = hyper_params['use_target_model']
        if self.use_target_model:
            self.target_model = DQNModel(input_len, output_len)
#         memory: Store and sample experience replay.
        self.memory = memory
        
        """
            batch_size: Mini batch size for training model.
            update_steps: The frequence of traning model
            model_replace_freq: The frequence of replacing 'target_model' by 'eval_model'
        """
        self.batch_size = hyper_params['batch_size']
        self.update_steps = hyper_params['update_steps']
        self.model_replace_freq = hyper_params['model_replace_freq']

        self.privous_q_net = []
        self.reuslt_count = 0
        self.evaluator_done = False
        self.training_episodes = training_episodes
        self.test_interval = test_interval
        self.results = [0] * (training_episodes // test_interval + 1)
            
    # Linear decrease function for epsilon
    def linear_decrease(self, initial_value, final_value, curr_steps, final_decay_steps):
        decay_rate = curr_steps / final_decay_steps
        if decay_rate > 1:
            decay_rate = 1
        return initial_value - (initial_value - final_value) * decay_rate

    def explore_or_exploit_policy(self, state):
        p = uniform(0, 1)

        # Get decreased epsilon
        epsilon = self.linear_decrease(self.initial_epsilon, 
                               self.final_epsilon,
                               self.steps,
                               self.epsilon_decay_steps)
        
        if p < epsilon:
            #return action
            return randint(0, self.action_space - 1)
        else:
            #return action
            return self.greedy_policy(state)
        
    def greedy_policy(self, state):
        return self.eval_model.predict(state)
    
    def update_batch(self):
        #memory_length = ray.get(self.memory.__len__.remote())
        #if memory_length < self.batch_size:
        #    return
        batch = ray.get(self.memory.sample.remote(self.batch_size))

        (states, actions, reward, next_states,
         is_terminal) = batch
        
        states = states
        next_states = next_states
        terminal = FloatTensor([1 if t else 0 for t in is_terminal])
        reward = FloatTensor(reward)
        batch_index = torch.arange(self.batch_size,
                                   dtype=torch.long)
        
        # Current Q Values
        _, q_values = self.eval_model.predict_batch(states)
        q_values = q_values[batch_index, actions]
        
        # Calculate target
        if self.use_target_model:
            actions, q_next = self.target_model.predict_batch(next_states)
        else:
            actions, q_next = self.eval_model.predict_batch(next_states)
            
        #INSERT YOUR CODE HERE --- neet to compute 'q_targets' used below
        q_target = []
        for i in range(len(terminal)):
            if terminal[i] == True:
                q_target.append(reward[i])
            else:
                q_target.append(reward[i] + self.beta*torch.max(q_next, dim = 1)[0][i])
        q_target = FloatTensor(q_target)
        
        # update model
        self.eval_model.fit(q_values, q_target)

    def learn(self, steps):
        self.episode += 1
        if self.episode > training_episodes:
            return True

        for i in range(steps):
            if self.steps % self.update_steps == 0:
                self.update_batch()
            if self.steps % self.model_replace_freq == 0:
                self.target_model.replace(self.eval_model)
            self.steps += 1

        if self.episode // self.test_interval + 1 > len(self.privous_q_net):
                self.privous_q_net.append(self.eval_model)
        return False

    def ask_evaluation(self):
        if len(self.privous_q_net) > self.reuslt_count:
            num = self.reuslt_count
            evluation_q_net = self.privous_q_net[num]
            self.reuslt_count += 1
            return False, num
        else:
            if self.episode >= self.training_episodes:
                self.evaluator_done = True
            return self.evaluator_done, None

    def add_result(self, result, num):
        self.results[num] = result

    def get_results(self):
        return self.results
    
    # save model
    def save_model(self, avg_reward):
        if avg_reward >= self.best_reward:
            self.best_reward = avg_reward
            self.eval_model.save(result_floder + '/best_model.pt')
        
    # load model
    def load_model(self):
        self.eval_model.load(result_floder + '/best_model.pt')

########################################################################################################################################
########################################################################################################################################
# Collector
@ray.remote
def collecting_worker(server, memory, env):
    max_episode_steps = env._max_episode_steps
    update_steps = 10
    model_replace_freq = 2000

    training_termination = False
    while True:
        if training_termination:
            break
        
        state = env.reset()
        done = False
        steps = 0
            
        while steps < max_episode_steps and not done:
            action = ray.get(server.explore_or_exploit_policy.remote(state))
            next_state, reward, done, _ = env.step(action)
            memory.add.remote(state, action, reward, next_state, done)
            state = next_state
            steps += 1
            
        training_termination = ray.get(server.learn.remote(steps))

# Evaluator            
@ray.remote
def evaluation_worker(server, env, test_number, trials):
    max_episode_steps = env._max_episode_steps

    while True:
        done, num = ray.get(server.ask_evaluation.remote())
        if done:
            break
        if not num:
            continue
        
        total_reward = 0
        for _ in range(trials):        
            state = env.reset()
            done = False
            steps = 0
            avg_reward = 0
            
            while steps < max_episode_steps and not done:
                steps += 1
                action = ray.get(server.greedy_policy.remote(state))
                state, reward, done, _ = env.step(action)
                total_reward += reward

        avg_reward = total_reward / trials
        f = open(result_file, "a+")
        f.write(str(avg_reward) + "\n")
        f.close()
        server.save_model.remote(avg_reward)
        print(avg_reward)
        
        server.add_result.remote(avg_reward, num)

# Distributed DQN
class distributed_DQN_agent():
    def __init__(self, env, hyper_params, cw_num = 4, ew_num = 4, test_interval = 50, training_episodes = 10000):
        self.memory = ReplayBuffer_remote.remote(2000)
        self.env = env
        self.training_episodes = training_episodes
        self.server = model_server.remote(self.env, self.memory, hyperparams_CartPole, training_episodes, test_interval)
        self.cw_num = cw_num
        self.ew_num = ew_num
        self.agent_name = "Distributed DQN"

    def learn_and_evaluate(self, training_episodes, test_interval, trials):
        test_number = int(training_episodes // test_interval)
        
        # collecting experience from different agents
        c_workers_id = []
        for j in range(int(self.cw_num)):
            c_workers_id.append(collecting_worker.remote(self.server, self.memory, self.env))
                
        # evaluation
        e_workers_id = []
        for j in range(int(self.ew_num)):
            e_workers_id.append(evaluation_worker.remote(self.server, self.env, test_number, trials))

        ray.wait(e_workers_id, len(e_workers_id))
        return ray.get(self.server.get_results.remote())

########################################################################################################################################
########################################################################################################################################
def main():
    training_episodes, test_interval = 10000, 50
    start_time = time.time()
    agent = distributed_DQN_agent(env_CartPole, hyperparams_CartPole)
    result = agent.learn_and_evaluate(training_episodes, test_interval, trials = 30)
    plot_result(result, test_interval, ["batch_update with target_model"])
    print()
    print("Time: ",time.time() - start_time)
    #run_time['Distributed DQN'] = time.time() - start_time
    #print("Learning time:\n")
    #print(run_time['Distributed DQN'])

    f = open(result_file, "a+")
    f.write(str(time.time() - start_time) + "\n")
    f.close()

if __name__ == "__main__":
    main()