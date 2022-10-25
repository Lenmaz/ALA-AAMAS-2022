import numpy as np
import time
import threading
from Environment import Environment


RIGHT = 0
UP = 1
LEFT = 2
that_initial_state = [10, 11, 9]

for_jar = list()


def translate_action(action):
    """
    Specific to the public civility game environmment, translates what each action number means

    :param action: int number identifying the action
    :return: string with the name of the action
    """

    part_1 = ""
    part_2 = ""

    if action < 3:
        part_1 = "MOVE "
    else:
        part_1 = "PUSH GARBAGE "

    if action % 3 == 0:
        part_2 = "RIGHT"
    elif action % 3 == 1:
        part_2 = "FORWARD"
    else:
        part_2 = "LEFT"

    action_name = part_1 + part_2
    return action_name


def scalarisation_function(values, w):
    """
    Scalarises the value of a state using a linear scalarisation function

    :param values: the different components V_0(s), ..., V_n(s) of the value of the state
    :param w:  the weight vector of the scalarisation function
    :return:  V(s), the scalarised value of the state
    """

    f = 0
    for objective in range(len(values)):
        f += w[objective]*values[objective]

    return f


def scalarised_Qs(env, Q_state, w):
    """
    Scalarises the value of each Q(s,a) for a given state using a linear scalarisation function

    :param Q_state: the different Q(s,a) for the state s, each with several components
    :param w: the weight vector of the scalarisation function
    :return: the scalarised value of each Q(s,a)
    """

    scalarised_Q = np.zeros(len(env.all_actions))
    for action in range(len(Q_state)):
        scalarised_Q[action] = scalarisation_function(Q_state[action], w)

    return scalarised_Q


def Q_function_calculator(env, state, V, discount_factor):
    """

    Calculates the value of applying each action to a given state. Heavily adapted to the public civility game

    :param env: the environment of the Markov Decision Process
    :param state: the current state
    :param V: value function to see the value of the next state V(s')
    :param discount_factor: discount factor considered, a real number
    :return: the value obtained for each action
    """

    Q_state = np.zeros([len(env.all_actions), len(V[0,0,0])])
    for action in env.all_actions:
        state_translated = env.translate_state(state[0], state[1], state[2])
        env.hard_reset(state_translated[0], state_translated[1], state_translated[2])
        next_state, rewards, _ = env.step([action])
        for objective in range(len(rewards)):
            Q_state[action, objective] = rewards[objective] + discount_factor * V[next_state[0], next_state[1], next_state[2], objective]
    return Q_state


def deterministic_optimal_policy_calculator(Q, env, weights):
    """
    Create a deterministic policy using the optimal Q-value function

    :param Q: optimal Q-function that has the optimal Q-value for each state-action pair (s,a)
    :param env: the environment, needed to know the number of states (adapted to the public civility game)
    :param weights: weight vector, to know how to scalarise the Q-values in order to select the optimals
    :return: a policy that for each state returns an optimal action
    """
    #
    policy = np.zeros([env.nb_cells, env.nb_cells, env.nb_cells])
    for cell_L in range(env.nb_cells):
        for cell_R in range(env.nb_cells):
            for cell_G in range(env.nb_cells):
                if cell_L != cell_R:
                    # One step lookahead to find the best action for this state
                    policy[cell_L, cell_R, cell_G] = np.argmax(scalarised_Qs(env, Q[cell_L, cell_R, cell_G], weights))

    return policy


def choose_action(st, eps, q_table, env, weights, infoQ, episode):
    """

    :param st: the current state in the environment
    :param eps: the epsilon value
    :param q_table:  q_table or q_function the algorithm is following
    :return:  the most optimal action for the current state or a random action
    """

    num_visited = infoQ[st[0],st[1],st[2]]

    if episode > 3000:
        proto_epsilon = min(eps**num_visited, eps**(episode-3000))
    else:
        proto_epsilon = eps**num_visited

    eps = max(0.001, proto_epsilon)

    NB_ACTIONS = 6

    if np.random.random() <= eps:
        return np.random.randint(NB_ACTIONS)
    else:
        maxi = np.max(scalarised_Qs(env, q_table[st[0], st[1], st[2]], weights))

        possible_actions = list()
        for act in range(NB_ACTIONS):
            q_A = scalarisation_function(q_table[st[0],st[1],st[2],act], weights)
            if q_A == maxi:
                possible_actions.append(act)

        return possible_actions[np.random.randint(len(possible_actions))]


def update_q_table(q_table, env, weights, alpha, gamma, action, state, new_state, reward):
    best_action = np.argmax(scalarised_Qs(env, q_table[new_state[0], new_state[1], new_state[2]], weights))

    for objective in range(len(reward)):

        q_table[state[0], state[1], state[2], action, objective] += alpha * (
            reward[objective] + gamma * q_table[new_state[0], new_state[1], new_state[2], best_action, objective] - q_table[state[0], state[1], state[2], action, objective])

def q_learning(env, weights, alpha=0.2, gamma=0.7):
    """
    Q-Learning Algorithm as defined in Sutton and Barto's 'Reinforcement Learning: An Introduction' Section 6.5,
    (1998).

    It has been adapted to the particularities of the public civility game, a deterministic environment, and also
     adapted to a MOMDP environment, having a reward function with several components (but assuming the linear scalarisation
    function is known).

    :param env: the environment encoding the (MO)MDP
    :param weights: the weight vector of the known linear scalarisation function
    :param alpha: the learning rate of the algorithm, can be set at discretion
    :param gamma: discount factor of the (MO)MPD, can be set at discretion (notice that this will change the Q-values)
    :return: the learnt policy and its associated state-value (V) and state-action-value (Q) functions
    """

    n_objectives = 2
    n_actions = len(env.all_actions)
    n_cells = env.nb_cells
    desired_initial_state = False
    Vs = [np.zeros([n_cells, n_cells, n_cells, n_objectives]), np.zeros([n_cells, n_cells, n_cells, n_objectives])]
    Qs = [np.zeros([n_cells, n_cells, n_cells, n_actions, n_objectives]), np.zeros([n_cells, n_cells, n_cells, n_actions, n_objectives])]


    max_episodes = 10000
    max_steps = 20

    epsilon = 0.999
    infoQ = np.zeros([n_cells, n_cells, n_cells])



    for_graphics = list()

    previous_R_big = [-14, 0]

    for episode in range(1, max_episodes + 1):
        done = False
        dones = [False, False]
        done_but_only_first_time = [False, False]

        env.hard_reset()

        state = env.get_state()


        if episode % 100 == 0:
            print("Episode : ", episode)

        step_count = 0

        R_big = [0, 0]

        while not done and step_count < max_steps:

            step_count += 1
            actions = list()

            infoQ[state[0],state[1],state[2]] += 1.0

            for i in range(2):
                if dones[i]:
                    actions.append(-1)
                else:
                    actions.append(choose_action(state, epsilon, Qs[i], env, weights, infoQ, episode))
            new_state, rewards, dones = env.step(actions)

            for i in range(2):
                if not done_but_only_first_time[i]:
                    update_q_table(Qs[i], env, weights, alpha, gamma, actions[i], state, new_state, rewards[i])
                    if i == 1:
                            R_big[0] += rewards[1][0]
                            R_big[1] += rewards[1][1]

            state = new_state

            for i in range(2):
                done_but_only_first_time[i] = dones[i]

            done = dones[0] and dones[1]

        for_graphics.append(R_big)
        #    previous_R_big = R_big
        #else:
        #    for_graphics.append(previous_R_big)





    # Now that we have Q, it is straightforward to obtain V
    for cell_L in range(env.nb_cells):
        for cell_R in range(env.nb_cells):
            for cell_G in range(env.nb_cells):
                if cell_L != cell_R:
                    for i in range(2):
                        best_action = np.argmax(scalarised_Qs(env, Qs[i][cell_L, cell_R, cell_G], weights))
                        Vs[i][cell_L, cell_R, cell_G] = Qs[i][cell_L, cell_R, cell_G, best_action]


    # Output a deterministic optimal policy
    policies = list()

    for i in range(2):
        policies.append(deterministic_optimal_policy_calculator(Qs[i], env, weights))


    np_graphics = np.array(for_graphics)
    np.save('example.npy', np_graphics)

    return policies #, V, Q


def example_execution(env, policies, render=False):
    """

    Simulation of the environment without learning. The L Agent moves according to the parameter policy provided.

    :param env: the environment encoding the (MO)MDP
    :param policy: a function S -> A assigning to each state the corresponding recommended action
    :return:
    """

    max_timesteps = 20

    for i in range(20):
        timesteps = 0
        env.hard_reset()

        state = env.get_state()
        gamma = 0.7
        done = False

        env.set_stats(i % 2, 99, 99, 99, 99)
        if render:
            if not env.drawing_paused():
                time.sleep(0.5)
                env.update_window()

        while (timesteps < max_timesteps) and (not done):
            timesteps += 1

            actions = list()
            for i in range(2):
                actions.append(policies[i][state[0], state[1], state[2]])

            state, rewards, dones = env.step(actions)

            done = dones[0] and dones[1]  # R Agent does not interfere

            if render:
                if not env.drawing_paused():
                    time.sleep(0.5)
                    env.update_window()

        if env.initial_garbage_position[1] == 1:
            env.initial_garbage_position = [4, 2]
        else:
            env.initial_garbage_position = [4,1]

class QLearner:
    """
    A Wrapper for the Q-learning method, which uses multithreading
    in order to handle the game rendering.
    """

    def __init__(self, environment, policies, drawing=False):

        threading.Thread(target=example_execution, args=(environment, policies, drawing,)).start()
        if drawing:
            env.render('Evaluating')


if __name__ == "__main__":

    env = Environment()
    w_E = 0.71

    print("Learning Process started. Will finish when Episode = 5000.")
    weights = [1.0, w_E]

    policies = q_learning(env, weights)

    #policies = np.load("policies.npy")

    env = Environment(garbage_pos=[4, 1])
    QLearner(env, policies, drawing=True)

    print("-------------------")



