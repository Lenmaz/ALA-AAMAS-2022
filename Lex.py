import numpy as np
import threading
import time
from Environment import Environment

only_ethical_matters = [0.0, 1.0]
only_individual_matters = [1.0, 0.0]


#learnt_policy_0 = np.load("policy_0.npy")
#learnt_policy_1 = np.load("policy_1.npy")


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

    best_ethical_Q = np.max(scalarised_Qs(env, Q_state, only_ethical_matters))
    best_individual_Q = -9999999
    for action in range(len(env.all_actions)):
        q_Individual = scalarisation_function(Q_state[action], only_individual_matters)
        q_Ethical = scalarisation_function(Q_state[action], only_ethical_matters)
        if q_Ethical == best_ethical_Q:
            if q_Individual > best_individual_Q:
                best_individual_Q = q_Individual
                action_to_choose = action

    return action_to_choose


def lexicographic_best_action(env, Q_state):
    possible_actions = list()
    best_Ethical = np.max(scalarised_Qs(env, Q_state, only_ethical_matters))

    best_Individual = -999999


    # Then we compute the best individual when the ethical objective is maximised
    for act in env.all_actions:
        q_Ethical = scalarisation_function(Q_state[act], only_ethical_matters)
        q_Individual = scalarisation_function(Q_state[act], only_individual_matters)
        if q_Ethical == best_Ethical:
            if q_Individual > best_Individual:
                best_Individual = q_Individual

    # And finally we compute such actions that correspond to this ethical-optimal
    for act in env.all_actions:
        q_Ethical = scalarisation_function(Q_state[act], only_ethical_matters)
        q_Individual = scalarisation_function(Q_state[act], only_individual_matters)
        if q_Ethical == best_Ethical:
            if q_Individual == best_Individual:
                possible_actions.append(act)

    return possible_actions[np.random.randint(len(possible_actions))]



def Q_function_calculator(env, state, V, who_is_the_learning_agent, discount_factor):
    """

    Calculates the value of applying each action to a given state. Heavily adapted to the public civility game

    :param env: the environment of the Markov Decision Process
    :param state: the current state
    :param V: value function to see the value of the next state V(s')
    :param discount_factor: discount factor considered, a real number
    :return: the value obtained for each action
    """

    Q_state = np.zeros([len(env.all_actions), len(V[0,0,0])])
    state_translated = env.translate_state(state[0], state[1], state[2])

    for action in env.all_actions:
        env.hard_reset(state_translated[0], state_translated[1], state_translated[2])

        if who_is_the_learning_agent == 0:
            #the_other_agent_action = learnt_policy_1[state[0], state[1], state[2]]
            the_other_agent_action = 1
            actions = [action, the_other_agent_action]
        elif who_is_the_learning_agent == 1:
            #the_other_agent_action = learnt_policy_0[state[0], state[1], state[2]]
            the_other_agent_action = 1
            actions = [the_other_agent_action, action]
        next_state, rewards, _ = env.step(actions)

        rewards = rewards[who_is_the_learning_agent]

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

    for cell_L in range(env.nb_cells):
        for cell_R in range(env.nb_cells):
            for cell_G in range(env.nb_cells):
                if cell_L != cell_R:
                    # One step lookahead to find the best action for this state
                    best_action = lexicographic_Qs(env, Q[cell_L, cell_R, cell_G])
                    policy[cell_L, cell_R, cell_G] = best_action
                    V[cell_L, cell_R, cell_G] = Q[cell_L, cell_R, cell_G, best_action]
    return policy


def lex(env, who_is_the_learning_agent=0, discount_factor=0.7):
    """
    Lex Algorithm.

    It has been adapted to the particularities of the public civility game.

    :param env: the environment encoding the (MO)MDP
    :param weights: the weight vector of the known linear scalarisation function
    :param theta: convergence parameter, the smaller it is the more precise the algorithm
    :param discount_factor: discount factor of the (MO)MPD, can be set at discretion
    :return:
    """

    n_objectives = 2
    n_actions = len(env.all_actions)
    n_cells = env.nb_cells
    V = np.zeros([n_cells, n_cells, n_cells, n_objectives])
    Q = np.zeros([n_cells, n_cells, n_cells, n_actions, n_objectives])

    for i in range(5):




        # Sweep for every state
        for cell_L in env.states_agent_left:
            for cell_R in env.states_agent_right:
                for cell_G in env.states_garbage:
                        if cell_L != cell_R:
                            # calculate the value of each action for the state
                            Q[cell_L, cell_R, cell_G] = Q_function_calculator(env, [cell_L, cell_R, cell_G], V, who_is_the_learning_agent, discount_factor)
                            # compute the best action for the state
                            best_action = lexicographic_best_action(env, Q[cell_L, cell_R, cell_G])
                            # Update the state value function
                            V[cell_L, cell_R, cell_G] = Q[cell_L, cell_R, cell_G, best_action]


    # Output a deterministic optimal policy
    policy = deterministic_optimal_policy_calculator(Q, env)

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
            #actions.append(learnt_policy_0[state[0], state[1], state[2]])
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

    who_is_the_learning_agent = 1
    env = Environment(who_is_the_learning_agent=who_is_the_learning_agent)


    print("-------------------")
    print("L(earning) Agent will learn now using Value Iteration in the Public Civility Game.")

    print("-------------------")
    print("Learning Process started. Will finish when Delta < Theta.")

    policy, v, q = lex(env, who_is_the_learning_agent, discount_factor=0.7)

    np.save("policy_"+str(who_is_the_learning_agent)+".npy", policy)

    print("-------------------")
    print("The Learnt Policy has the following Value:")
    policy_value = v[10,11,8]
    print(q[10, 11, 9])
    print("Individual Value V_0 = " + str(round(policy_value[0],2)))
    print("Ethical Value (V_N + V_E) = " + str(round(policy_value[1],2)))
    if v[10,11,9][1] >= -10:
        print("We Proceed to show the learnt policy. Please use the image PCG_positions.png provided to identify the agent and garbage positions:")
        print()

        env = Environment(garbage_pos=[4, 2], is_deterministic=True)
        QLearner(env, policy, who_is_the_learning_agent, drawing=True)

        print("-------------------")



