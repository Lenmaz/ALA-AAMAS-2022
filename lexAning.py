import numpy as np
import threading
import time
from Environment import Environment

RIGHT = 0
UP = 1
LEFT = 2

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

def lexicographic_Qs(env, Q_state):
    action_to_choose = -1

    best_ethical_Q = np.max(scalarised_Qs(env, Q_state, [0.0, 1.0]))
    best_individual_Q = -9999999
    for action in range(len(env.all_actions)):
        q_Individual = scalarisation_function(Q_state[action], [1.0, 0.0])
        q_Ethical = scalarisation_function(Q_state[action], [0.0, 1.0])
        if q_Ethical == best_ethical_Q:
            if q_Individual > best_individual_Q:
                best_individual_Q = q_Individual
                action_to_choose = action

    return action_to_choose

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


def deterministic_optimal_policy_calculator(Q, env):
    """
    Create a deterministic policy using the optimal Q-value function

    :param Q: optimal Q-function that has the optimal Q-value for each state-action pair (s,a)
    :param env: the environment, needed to know the number of states (adapted to the public civility game)
    :param weights: weight vector, to know how to scalarise the Q-values in order to select the optimals
    :return: a policy that for each state returns an optimal action
    """

    policy = np.zeros([env.nb_cells, env.nb_cells, env.nb_cells])
    V = np.zeros([env.nb_cells, env.nb_cells, env.nb_cells, 2])

    for cell_L in env.states_agent_left:
        for cell_R in env.states_agent_right:
            for cell_G in env.states_garbage:
                if cell_L != cell_R:
                    # One step lookahead to find the best action for this state
                    best_action = lexicographic_Qs(env, Q[cell_L, cell_R, cell_G])
                    policy[cell_L, cell_R, cell_G] = best_action
                    V[cell_L, cell_R, cell_G] = Q[cell_L, cell_R, cell_G, best_action]
    return policy, V


def choose_action(st, eps, q_table, env, infoQ):
    """

    :param st: the current state in the environment
    :param eps: the epsilon value
    :param q_table:  q_table or q_function the algorithm is following
    :return:  the most optimal action for the current state or a random action
    """

    eps = max(0.01, eps**infoQ[st[0],st[1],st[2]])
    NB_ACTIONS = 6

    if np.random.random() <= eps:
        return np.random.randint(NB_ACTIONS)
    else:

        # First we look for the argmax actions among the ethical objective
        best_Ethical = np.max(scalarised_Qs(env, q_table[st[0], st[1], st[2]], [0.0, 1.0]))

        possible_actions = list()
        best_Individual = -999999

        # Then we compute the best individual when the ethical objective is maximised
        for act in range(NB_ACTIONS):
            q_Ethical = scalarisation_function(q_table[st[0],st[1],st[2],act], [0.0, 1.0])
            q_Individual = scalarisation_function(q_table[st[0], st[1], st[2], act], [1.0, 0.0])
            if q_Ethical == best_Ethical:
                if q_Individual > best_Individual:
                    best_Individual = q_Individual

        # And finally we compute such actions that correspond to this ethical-optimal
        for act in range(NB_ACTIONS):
            q_Ethical = scalarisation_function(q_table[st[0], st[1], st[2], act], [0.0, 1.0])
            q_Individual = scalarisation_function(q_table[st[0], st[1], st[2], act], [1.0, 0.0])
            if q_Ethical == best_Ethical:
                if q_Individual == best_Individual:
                    possible_actions.append(act)

        return possible_actions[np.random.randint(len(possible_actions))]

def update_q_table(q_table, env, alpha, gamma, action, state, new_state, reward):

    best_action = lexicographic_Qs(env, q_table[new_state[0], new_state[1], new_state[2]])

    for objective in range(len(reward)):
        q_table[state[0], state[1], state[2], action, objective] += alpha * (
            reward[objective] + gamma * q_table[new_state[0], new_state[1], new_state[2], best_action, objective] - q_table[state[0], state[1], state[2], action, objective])

def q_LEXning(env, alpha=1.0, discount_factor=0.7, who_is_the_learning_agent=0,):
    """
    Lex Algorithm.

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
    n_actions = 6
    n_cells = env.nb_cells
    print("Number of cells : ", n_cells)

    Q = np.zeros([n_cells, n_cells, n_cells, n_actions, n_objectives])

    max_episodes = 3000
    max_steps = 20

    gamma = discount_factor

    epsilon = 0.999
    infoQ = np.zeros([n_cells, n_cells, n_cells])

    for_graphics = list()

    for episode in range(1, max_episodes + 1):
        done = False

        env.hard_reset()

        state = env.get_state()


        step_count = 0

        R_big = [0, 0]

        while not done and step_count < max_steps:

            step_count += 1
            actions = list()



            actions = list()
            if who_is_the_learning_agent == 0:
                actions.append(choose_action(state, epsilon, Q, env, infoQ))  # L agent uses the learnt policy
                actions.append(1)
            elif who_is_the_learning_agent == 1:
                actions.append(1)
                actions.append(choose_action(state, epsilon, Q, env, infoQ))


            infoQ[state[0],state[1],state[2]] += 1.0
            new_state, reward, dones = env.step(actions)

            reward = reward[who_is_the_learning_agent]

            update_q_table(Q, env, alpha, gamma, actions[who_is_the_learning_agent], state, new_state, reward)

            state = new_state
            done = dones[who_is_the_learning_agent]


    # Output a deterministic optimal policy
    policy, V = deterministic_optimal_policy_calculator(Q, env)


    return policy, V, Q


def example_execution(env, policy, who_is_the_learning_agent=0, render=False):
    """

    Simulation of the environment without learning. The L Agent moves according to the parameter policy provided.

    :param env: the environment encoding the (MO)MDP
    :param policy: a function S -> A assigning to each state the corresponding recommended action
    :return:
    """

    state = env.get_state()
    returns = list()
    gamma = 0.7
    original_gamma = gamma
    done = False
    dones = [False, False]

    individual_objective_fulfilled = False
    env.set_stats(1, 2, 3, 4, 5)
    if render:
        if not env.drawing_paused():
            time.sleep(0.5)
            env.update_window()



    while not done:

        actions = list()
        if who_is_the_learning_agent == 0:
            actions.append(policy[state[0], state[1], state[2]])  # L agent uses the learnt policy
            actions.append(1)
            #actions.append(learnt_policy_1[state[0], state[1], state[2]])
        elif who_is_the_learning_agent == 1:
            actions.append(1)
            actions.append(policy[state[0], state[1], state[2]])
        action_recommended = translate_action(actions[who_is_the_learning_agent])
        print("L Agent position: " + str(state[0]) + ". R Agent position: " + str(state[1]) + ". Garbage position: " + str(
            state[2]) + ". Learner Action: " + action_recommended)

        state, rewards, dones = env.step(actions)
        done = dones[who_is_the_learning_agent]  # R Agent does not interfere

        if render:
            if not env.drawing_paused():
                time.sleep(0.5)
                env.update_window()

        rewards = rewards[who_is_the_learning_agent]

        print("---now returns---")

        if len(returns) == 0:
            returns = rewards

        else:
            for i in range(len(rewards)):
                returns[i] += gamma*rewards[i]


            gamma *= original_gamma

        if done:
            if not individual_objective_fulfilled:
                individual_objective_fulfilled = True

                print("Learning Agent position: " + str(state[0]) + ". Garbage position: " + str(
                    state[2]) + ".")
                print("====Individual objective fulfilled! Agent in goal position====")

    print("Policy Value: ", returns)

class QLearner:
    """
    A Wrapper for the Q-learning method, which uses multithreading
    in order to handle the game rendering.
    """

    def __init__(self, environment, policy, who_is_the_learning_agent=0, drawing=False):

        threading.Thread(target=example_execution, args=(environment, policy, who_is_the_learning_agent, drawing,)).start()
        if drawing:
            env.render('Evaluating')

if __name__ == "__main__":
    who_is_the_learning_agent = 0

    if who_is_the_learning_agent == 0:
        initial_state = [4, 1]
    else:
        initial_state = [4, 2]
    env = Environment(who_is_the_learning_agent=who_is_the_learning_agent)

    policy, v, q = q_LEXning(env, who_is_the_learning_agent=who_is_the_learning_agent)

    env = Environment(garbage_pos=initial_state)
    QLearner(env, policy, who_is_the_learning_agent, drawing=True)




