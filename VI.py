import numpy as np
import threading
import time
from Environment import Environment


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
    state_translated = env.translate_state(state[0], state[1], state[2])

    for action in env.all_actions:

        env.hard_reset(state_translated[0], state_translated[1], state_translated[2])
        next_state, rewards, _ = env.step([action, 1])
        rewards = rewards[0]
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
    policy = np.zeros([14, 14, 14])
    for cell_L in env.states_agent_left:
        for cell_R in env.states_agent_right:
            for cell_G in env.states_garbage:
                if cell_L != cell_R:
                    # One step lookahead to find the best action for this state
                    policy[cell_L, cell_R, cell_G] = np.argmax(scalarised_Qs(env, Q[cell_L, cell_R, cell_G], weights))

    return policy


def value_iteration(env, weights, theta=0.1, discount_factor=0.7):
    """
    Value Iteration Algorithm as defined in Sutton and Barto's 'Reinforcement Learning: An Introduction' Section 4.4,
    (1998).

    It has been adapted to the particularities of the public civility game, a deterministic envirnoment, and also
     adapted to a MOMDP environment, having a reward function with several components (but assuming the linear scalarisation
    function is known).

    :param env: the environment encoding the (MO)MDP
    :param weights: the weight vector of the known linear scalarisation function
    :param theta: convergence parameter, the smaller it is the more precise the algorithm
    :param discount_factor: discount factor of the (MO)MPD, can be set at discretion
    :return:
    """

    n_objectives = 2
    n_actions = 6
    n_cells = 14
    V = np.zeros([n_cells, n_cells, n_cells, n_objectives])
    Q = np.zeros([n_cells, n_cells, n_cells, n_actions, n_objectives])

    for _ in range(5):
        # Threshold delta
        delta = 0
        # Sweep for every state
        for cell_L in env.states_agent_left:
            for cell_R in env.states_agent_right:
                for cell_G in env.states_garbage:
                        if cell_L != cell_R:
                            # calculate the value of each action for the state
                            Q[cell_L, cell_R, cell_G] = Q_function_calculator(env, [cell_L, cell_R, cell_G], V, discount_factor)
                            # compute the best action for the state
                            best_action = np.argmax(scalarised_Qs(env, Q[cell_L, cell_R, cell_G], weights))
                            best_action_value = scalarisation_function(Q[cell_L, cell_R, cell_G, best_action], weights)
                            # Recalculate delta
                            delta = max(delta, np.abs(best_action_value - scalarisation_function(V[cell_L, cell_R, cell_G], weights)))
                            # Update the state value function
                            V[cell_L, cell_R, cell_G] = Q[cell_L, cell_R, cell_G, best_action]
        # Check if we can finish the algorithm


        if delta < theta:
            print('Delta = ' + str(round(delta, 3)) + " < Theta = " + str(theta))
            print("Learning Process finished!")
            break
        else:
            print('Delta = ' + str(round(delta, 3)) + " > Theta = " + str(theta))


    # Output a deterministic optimal policy
    policy = deterministic_optimal_policy_calculator(Q, env, weights)

    return policy, V, Q

def example_execution(env, policy, who_is_the_learning_agent=0, render=True):
    """

    Simulation of the environment without learning. The L Agent moves according to the parameter policy provided.

    :param policy: a function S -> A assigning to each state the corresponding recommended action
    :return:
    """
    env.set_stats(1, 2, 3, 4, 5)
    if render:
        if not env.drawing_paused():
            time.sleep(0.5)
            env.update_window()
    state = env.get_state()

    done = False

    ethical_objective_fulfilled = False
    individual_objective_fulfilled = False

    while not done:


        actions = list()
        actions.append(policy[state[0], state[1], state[2]])  # L agent uses the learnt policy
        actions.append(1)  # R Agent does not interfere

        action_recommended = translate_action(policy[state[0], state[1], state[2]])
        print("L Agent position: " + str(state[0]) + ". Garbage position: " + str(
            state[2]) + ". Action: " + action_recommended)

        if state[2] == 2:
            if not ethical_objective_fulfilled:
                ethical_objective_fulfilled = True
                print("====Ethical objective fulfilled! Garbage in wastebasket====")

        state, rewards, dones = env.step(actions)
        done = dones[0]  # R Agent does not interfere

        if render:
            if not env.drawing_paused():
                time.sleep(0.5)
                env.update_window()

        if done:
            if not individual_objective_fulfilled:
                individual_objective_fulfilled = True

                print("L Agent position: " + str(state[0]) + ". Garbage position: " + str(
                    state[2]) + ".")
                print("====Individual objective fulfilled! Agent in goal position====")


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
    env = Environment()
    w_E = 2.1
    print("-------------------")
    print("L(earning) Agent will learn now using Value Iteration in the Public Civility Game.")
    print("The Ethical Weight of the Scalarisation Function is set to W_E = " + str(w_E) + ", found by our Algorithm.")
    print("-------------------")
    print("Learning Process started. Will finish when Delta < Theta.")
    weights = [1.0, w_E]

    policy, v, q = value_iteration(env, weights, discount_factor=0.7)

    print("-------------------")
    print("The Learnt Policy has the following Value:")
    policy_value = v[10,11,8]
    print("Individual Value V_0 = " + str(round(policy_value[0],2)))
    print("Ethical Value (V_N + V_E) = " + str(round(policy_value[1],2)))
    if v[10, 11, 8][1] >= 2.4:
        print("Since V_N + V_E = 2.4, the L Agent has learnt the Ethical Policy.")
    print("-------------------")
    if v[10, 11, 8][1] >= -10:
        print("We Proceed to show the learnt policy. Please use the image PCG_positions.png provided to identify the agent and garbage positions:")
        print()

        env = Environment(garbage_pos=[4, 1], is_deterministic=True)
        QLearner(env, policy, who_is_the_learning_agent=0, drawing=True)

        print("-------------------")



