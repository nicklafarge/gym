import numpy as np
import gym

env_name = 'CartPole-v0'


# env_name = 'MountainCar-v0'
# env_name = 'MsPacman-v0'
# env_name = 'SpaceInvaders-v0'
# env_name = 'Hopper-v1'
# env_name = 'InvertedDoublePendulum-v1'

# n_episodes = 35
# for i_episode in range(n_episodes):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         # print(observation)
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         print reward
#         if done:
#             print("Episode finished after {} timesteps".format(t + 1))
#             break


def run_episode(env, parameters, render=False):
    observation = env.reset()
    totalreward = 0
    for _ in xrange(200):
        if render:
            env.render()
        action = 0 if np.matmul(parameters, observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        totalreward += reward
        if done:
            break
    return totalreward


def random_params(n_trials=50):
    """
    For n_trials=1500, average number of trials was 13.18
    """
    trials_list = []
    for i in range(n_trials):
        bestparams = None
        bestreward = 0
        for n in xrange(10000):
            env.reset()
            parameters = np.random.rand(4) * 2 - 1
            reward = run_episode(env, parameters)
            if reward > bestreward:
                bestreward = reward
                bestparams = parameters
                # considered solved if the agent lasts 200 timesteps
                if reward == 200:
                    trials_list.append(n)
                    print n
                    break

    return trials_list


def hill_climbing(n_trials=50):
    trials_list = []
    noise_scaling = 0.1
    for i in range(n_trials):
        parameters = np.random.rand(4) * 2 - 1
        bestreward = 0
        for n in xrange(10000):
            newparams = parameters + (np.random.rand(4) * 2 - 1) * noise_scaling
            reward = run_episode(env, newparams)
            if reward > bestreward:
                bestreward = reward
                parameters = newparams
                if reward == 200:
                    trials_list.append(parameters)
                    print n
                    break

    return np.average(trials_list)


if __name__ == '__main__':
    env = gym.make(env_name)

    # run_episode(env)


    n_trials = 200
    trials_list = random_params(n_trials)
    # n = hill_climbing()

    t = trials_list[-1]
    run_episode(env, t, render=True)
    # test = np.random.rand(4) * 2 - 1
    # run_episode(env, [0, 0, 0, 0], render=True)
    print 'Average Number of Trials: %0.2f' % np.average(trials_list)

    pass
