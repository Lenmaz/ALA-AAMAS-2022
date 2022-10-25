from Environment import Environment
from SAEEP import Single_Agent_Ethical_Environment_Designer
from Lex import lex
from lexAning import q_LEXning


def target_joint_policy_computation():
    target_joint_policy = list()
    for i in range(2):
        print("Computing ethical policy for agent ", i)
        env = Environment(who_is_the_learning_agent=i)    # Decompose
        policy, _, _ = q_LEXning(env, who_is_the_learning_agent=i, discount_factor=0.7)   # Solve
        target_joint_policy.append(policy)                # Aggregate

    return target_joint_policy


def solution_weight_computation(target_joint_policy, epsilon, discount_factor, max_iterations):

    weights = list()
    for i in range(2):
        print("Computing ethical weight for agent ", i)
        env = Environment(who_is_the_learning_agent=i)    # Decompose
        ethical_weight = Single_Agent_Ethical_Environment_Designer(env, target_joint_policy, 0.0, i, discount_factor, max_iterations)   # Solve
        weights.append(ethical_weight)                # Aggregate

    return max(weights) + epsilon


def Multi_Agent_Ethical_Environment_Designer(epsilon, discount_factor, max_iterations):

    target_joint_policy = target_joint_policy_computation()
    solution_ethical_weight = solution_weight_computation(target_joint_policy, epsilon, discount_factor, max_iterations)

    return solution_ethical_weight


if __name__ == "__main__":

    epsilon = 0.1
    who_is_the_learning_agent = 0
    discount_factor = 0.7
    max_iterations = 5

    solution_ethical_weight = Multi_Agent_Ethical_Environment_Designer(epsilon, discount_factor, max_iterations)

    print("Solution weight vector for the Public Civility Game : ", solution_ethical_weight)