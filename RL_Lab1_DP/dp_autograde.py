import numpy as np
from collections import defaultdict

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    # YOUR CODE HERE
    raise NotImplementedError
    return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    # YOUR CODE HERE
    theta_counter = 0
    grid_size = env.nS**(1/2)
    old = np.zeros(env.nS)
    for cell in range(env.nS):
        p = env.P[cell]
        value = 0
        for action in range(grid_size):
            '''Since this is deterministic, the outcome (resulting state) of each action is known.
            Therefore, all we need to do to determine the value is sum the reward (with discount) after each action.
            However, for the sake of completeness I have left in the transition probability (p[action][0]).'''
            transition_prob = p[action][0]
            reward = p[action][1]
            resulting_state = p[action][2]
            value += transition_prob * (reward + V[resulting_state]*discount_factor)

        if value - V[cell] < theta:
            theta_counter += 1
        else:
            theta_counter = 0

        V[cell] = value

        
    return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    # YOUR CODE HERE
    theta_counter = 0
    grid_size = np.sqrt(env.nS)
    old = np.zeros(env.nS)
    for cell in range(env.nS):
        p = env.P[cell]
        value = 0
        for action in range(grid_size):
            '''Since this is deterministic, the outcome (resulting state) of each action is known.
            Therefore, all we need to do to determine the value is sum the reward (with discount) after each action.
            However, for the sake of completeness I have left in the transition probability (p[action][0]).'''
            transition_prob = p[action][0]
            reward = p[action][1]
            resulting_state = p[action][2]
            value += transition_prob * (reward + V[resulting_state]*discount_factor)

        if value - V[cell] < theta:
            theta_counter += 1
        else:
            theta_counter = 0

        V[cell] = value

        
    return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    # YOUR CODE HERE
    theta_counter = 0
    grid_size = np.sqrt(env.nS)
    old = np.zeros(env.nS)
    for cell in range(env.nS):
        p = env.P[cell]
        value = 0
        for action in range(grid_size):
            '''Since this is deterministic, the outcome (resulting state) of each action is known.
            Therefore, all we need to do to determine the value is sum the reward (with discount) after each action.
            However, for the sake of completeness I have left in the transition probability (p[action][0]).'''
            transition_prob = p[action][0]
            reward = p[action][1]
            resulting_state = p[action][2]
            value += transition_prob * (reward + V[resulting_state]*discount_factor)

        if value - V[cell] < theta:
            theta_counter += 1
        else:
            theta_counter = 0

        V[cell] = value

        
    return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    # YOUR CODE HERE
    theta_counter = 0
    grid_size = int(np.sqrt(env.nS))
    old = np.zeros(env.nS)
    for cell in range(env.nS):
        p = env.P[cell]
        value = 0
        for action in range(grid_size):
            '''Since this is deterministic, the outcome (resulting state) of each action is known.
            Therefore, all we need to do to determine the value is sum the reward (with discount) after each action.
            However, for the sake of completeness I have left in the transition probability (p[action][0]).'''
            transition_prob = p[action][0]
            reward = p[action][1]
            resulting_state = p[action][2]
            value += transition_prob * (reward + V[resulting_state]*discount_factor)

        if value - V[cell] < theta:
            theta_counter += 1
        else:
            theta_counter = 0

        V[cell] = value

        
    return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    # YOUR CODE HERE
    theta_counter = 0
    grid_size = int(np.sqrt(env.nS))
    old = np.zeros(env.nS)
    for cell in range(env.nS):
        p = env.P[cell]
        value = 0
        for action in range(grid_size):
            '''Since this is deterministic, the outcome (resulting state) of each action is known.
            Therefore, all we need to do to determine the value is sum the reward (with discount) after each action.
            However, for the sake of completeness I have left in the transition probability (p[action][0]).'''
            transition_prob = p[action][0]
            reward = p[action][1]
            resulting_state = p[action][2]
            value += transition_prob * (reward + V[resulting_state]*discount_factor)

        if value - V[cell] < theta:
            theta_counter += 1
        else:
            theta_counter = 0

        V[cell] = value

        
    return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    # YOUR CODE HERE
    theta_counter = 0
    grid_size = int(np.sqrt(env.nS))
    old = np.zeros(env.nS)
    for cell in range(env.nS):
        p = env.P[cell]
        print(p)
        value = 0
        for action in range(grid_size):
            '''Since this is deterministic, the outcome (resulting state) of each action is known.
            Therefore, all we need to do to determine the value is sum the reward (with discount) after each action.
            However, for the sake of completeness I have left in the transition probability (p[action][0]).'''
            transition_prob = p[action][0]
            reward = p[action][1]
            resulting_state = p[action][2]
            value += transition_prob * (reward + V[resulting_state]*discount_factor)

        if value - V[cell] < theta:
            theta_counter += 1
        else:
            theta_counter = 0

        V[cell] = value

        
    return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    # YOUR CODE HERE
    theta_counter = 0
    grid_size = int(np.sqrt(env.nS))
    old = np.zeros(env.nS)
    for cell in range(env.nS):
        p = env.P[cell]
        print(p)
        value = 0
        for action in range(grid_size):
            '''Since this is deterministic, the outcome (resulting state) of each action is known.
            Therefore, all we need to do to determine the value is sum the reward (with discount) after each action.
            However, for the sake of completeness I have left in the transition probability (p[action][0]).'''
            transition_prob = p[action][0]
            reward = p[action][1]
            resulting_state = p[action][2]
            value += transition_prob * (reward + V[resulting_state]*discount_factor)

        if value - V[cell] < theta:
            theta_counter += 1
        else:
            theta_counter = 0

        V[cell] = value

        
    return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    # YOUR CODE HERE
    theta_counter = 0
    grid_size = int(np.sqrt(env.nS))
    old = np.zeros(env.nS)
    for cell in range(env.nS):
        p = env.P[cell]
        value = 0
        for action in range(grid_size):
            print(p)
            '''Since this is deterministic, the outcome (resulting state) of each action is known.
            Therefore, all we need to do to determine the value is sum the reward (with discount) after each action.
            However, for the sake of completeness I have left in the transition probability (p[action][0]).'''
            transition_prob = p[action][0]
            reward = p[action][1]
            resulting_state = p[action][2]
            value += transition_prob * (reward + V[resulting_state]*discount_factor)

        if value - V[cell] < theta:
            theta_counter += 1
        else:
            theta_counter = 0

        V[cell] = value

        
    return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    # YOUR CODE HERE
    theta_counter = 0
    grid_size = int(np.sqrt(env.nS))
    old = np.zeros(env.nS)
    for cell in range(env.nS):
        p = env.P[cell]
        value = 0
        for action in range(grid_size):
            print(p[action])
            '''Since this is deterministic, the outcome (resulting state) of each action is known.
            Therefore, all we need to do to determine the value is sum the reward (with discount) after each action.
            However, for the sake of completeness I have left in the transition probability (p[action][0]).'''
            transition_prob = p[action][0]
            reward = p[action][1]
            resulting_state = p[action][2]
            value += transition_prob * (reward + V[resulting_state]*discount_factor)

        if value - V[cell] < theta:
            theta_counter += 1
        else:
            theta_counter = 0

        V[cell] = value

        
    return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    # YOUR CODE HERE
    theta_counter = 0
    grid_size = int(np.sqrt(env.nS))
    old = np.zeros(env.nS)
    for cell in range(env.nS):
        p = env.P[cell]
        value = 0
        for action in range(grid_size):
            print(p[action])
            '''Since this is deterministic, the outcome (resulting state) of each action is known.
            Therefore, all we need to do to determine the value is sum the reward (with discount) after each action.
            However, for the sake of completeness I have left in the transition probability (p[action][0]).'''
            transition_prob = p[action][0]
            reward = p[action][1]
            resulting_state = p[action][2]
            value += transition_prob * (reward + V[resulting_state]*discount_factor)

        if value - V[cell] < theta:
            theta_counter += 1
        else:
            theta_counter = 0

        V[cell] = value

        
    return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    # YOUR CODE HERE
    theta_counter = 0
    grid_size = int(np.sqrt(env.nS))
    old = np.zeros(env.nS)
    for cell in range(env.nS):
        p = env.P[cell]
        value = 0
        for action in range(grid_size):
            print(p[action][0])
            '''Since this is deterministic, the outcome (resulting state) of each action is known.
            Therefore, all we need to do to determine the value is sum the reward (with discount) after each action.
            However, for the sake of completeness I have left in the transition probability (p[action][0]).'''
            transition_prob = p[action][0]
            reward = p[action][1]
            resulting_state = p[action][2]
            value += transition_prob * (reward + V[resulting_state]*discount_factor)

        if value - V[cell] < theta:
            theta_counter += 1
        else:
            theta_counter = 0

        V[cell] = value

        
    return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    # YOUR CODE HERE
    theta_counter = 0
    grid_size = int(np.sqrt(env.nS))
    old = np.zeros(env.nS)
    for cell in range(env.nS):
        p = env.P[cell]
        value = 0
        for action in range(grid_size):
            print(p[action][0])
            '''Since this is deterministic, the outcome (resulting state) of each action is known.
            Therefore, all we need to do to determine the value is sum the reward (with discount) after each action.
            However, for the sake of completeness I have left in the transition probability (p[action][0]).'''
            transition_prob = p[action][0]
            reward = p[action][1]
            resulting_state = p[action][2]
            value += transition_prob * (reward + V[resulting_state]*discount_factor)

        if value - V[cell] < theta:
            theta_counter += 1
        else:
            theta_counter = 0

        V[cell] = value

        
    return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    # YOUR CODE HERE
    theta_counter = 0
    grid_size = int(np.sqrt(env.nS))
    old = np.zeros(env.nS)
    for cell in range(env.nS):
        p = env.P[cell]
        value = 0
        for action in range(grid_size):
            print(p[action][0][0])
            '''Since this is deterministic, the outcome (resulting state) of each action is known.
            Therefore, all we need to do to determine the value is sum the reward (with discount) after each action.
            However, for the sake of completeness I have left in the transition probability (p[action][0]).'''
            transition_prob = p[action][0]
            reward = p[action][1]
            resulting_state = p[action][2]
            value += transition_prob * (reward + V[resulting_state]*discount_factor)

        if value - V[cell] < theta:
            theta_counter += 1
        else:
            theta_counter = 0

        V[cell] = value

        
    return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    # YOUR CODE HERE
    theta_counter = 0
    grid_size = int(np.sqrt(env.nS))
    old = np.zeros(env.nS)
    for cell in range(env.nS):
        p = env.P[cell]
        value = 0
        for action in range(grid_size):
            print(p)
            '''Since this is deterministic, the outcome (resulting state) of each action is known.
            Therefore, all we need to do to determine the value is sum the reward (with discount) after each action.
            However, for the sake of completeness I have left in the transition probability (p[action][0]).'''
            transition_prob = p[action][0]
            reward = p[action][1]
            resulting_state = p[action][2]
            value += transition_prob * (reward + V[resulting_state]*discount_factor)

        if value - V[cell] < theta:
            theta_counter += 1
        else:
            theta_counter = 0

        V[cell] = value

        
    return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    # YOUR CODE HERE
    theta_counter = 0
    grid_size = int(np.sqrt(env.nS))
    old = np.zeros(env.nS)
    for cell in range(env.nS):
        p = env.P[cell]
        value = 0
        for action in range(grid_size):
            '''Since this is deterministic, the outcome (resulting state) of each action is known.
            Therefore, all we need to do to determine the value is sum the reward (with discount) after each action.
            However, for the sake of completeness I have left in the transition probability (p[action][0]).'''
            transition_prob = p[action][0][0]
            reward = p[action][0][1]
            resulting_state = p[action][0][2]
            value += transition_prob * (reward + V[resulting_state]*discount_factor)

        if value - V[cell] < theta:
            theta_counter += 1
        else:
            theta_counter = 0

        V[cell] = value

        
    return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    # YOUR CODE HERE
    theta_counter = 0
    grid_size = int(np.sqrt(env.nS))
    old = np.zeros(env.nS)
    for cell in range(env.nS):
        p = env.P[cell]
        value = 0
        for action in range(grid_size):
            '''Since this is deterministic, the outcome (resulting state) of each action is known.
            Therefore, all we need to do to determine the value is sum the reward (with discount) after each action.
            However, for the sake of completeness I have left in the transition probability (p[action][0]).'''
            transition_prob = p[action][0][0]
            reward = p[action][0][1]
            resulting_state = p[action][0][2]
            print(resulting_state)
            value += transition_prob * (reward + V[resulting_state]*discount_factor)

        if value - V[cell] < theta:
            theta_counter += 1
        else:
            theta_counter = 0

        V[cell] = value

        
    return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    # YOUR CODE HERE
    theta_counter = 0
    grid_size = int(np.sqrt(env.nS))
    old = np.zeros(env.nS)
    for cell in range(env.nS):
        p = env.P[cell]
        value = 0
        for action in range(grid_size):
            '''Since this is deterministic, the outcome (resulting state) of each action is known.
            Therefore, all we need to do to determine the value is sum the reward (with discount) after each action.
            However, for the sake of completeness I have left in the transition probability (p[action][0]).'''
            transition_prob = p[action][0][0]
            reward = p[action][0][1]
            resulting_state = p[action][0][2]
            print(resulting_state)
            value += transition_prob * (reward + V[resulting_state]*discount_factor)

        if value - V[cell] < theta:
            theta_counter += 1
        else:
            theta_counter = 0

        V[cell] = value

        
    return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    # YOUR CODE HERE
    theta_counter = 0
    grid_size = int(np.sqrt(env.nS))
    old = np.zeros(env.nS)
    for cell in range(env.nS):
        p = env.P[cell]
        value = 0
        for action in range(grid_size):
            '''Since this is deterministic, the outcome (resulting state) of each action is known.
            Therefore, all we need to do to determine the value is sum the reward (with discount) after each action.
            However, for the sake of completeness I have left in the transition probability (p[action][0]).'''
            transition_prob = p[action][0][0]
            reward = p[action][0][1]
            resulting_state = int(p[action][0][2])
            value += transition_prob * (reward + V[resulting_state]*discount_factor)

        if value - V[cell] < theta:
            theta_counter += 1
        else:
            theta_counter = 0

        V[cell] = value

        
    return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    # YOUR CODE HERE
    theta_counter = 0
    grid_size = int(np.sqrt(env.nS))
    old = np.zeros(env.nS)
    for cell in range(env.nS):
        p = env.P[cell]
        value = 0
        for action in range(grid_size):
            '''Since this is deterministic, the outcome (resulting state) of each action is known.
            Therefore, all we need to do to determine the value is sum the reward (with discount) after each action.
            However, for the sake of completeness I have left in the transition probability (p[action][0]).'''
            transition_prob = p[action][0][0]
            reward = p[action][0][1]
            resulting_state = int(p[action][0][2])
            value += transition_prob * (reward + V[resulting_state]*discount_factor)

        if value - V[cell] < theta:
            theta_counter += 1
        else:
            theta_counter = 0

        V[cell] = value

        
    return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    # YOUR CODE HERE
    theta_counter = 0
    grid_size = int(np.sqrt(env.nS))
    old = np.zeros(env.nS)
    for cell in range(env.nS):
        p = env.P[cell]
        value = 0
        for action in range(grid_size):
            '''Since this is deterministic, the outcome (resulting state) of each action is known.
            Therefore, all we need to do to determine the value is sum the reward (with discount) after each action.
            However, for the sake of completeness I have left in the transition probability (p[action][0]).'''
            transition_prob = p[action][0][0]
            resulting_state = int(p[action][0][1])
            reward = p[action][0][2]
            
            value += transition_prob * (reward + V[resulting_state]*discount_factor)

        if value - V[cell] < theta:
            theta_counter += 1
        else:
            theta_counter = 0

        V[cell] = value

        
    return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    # YOUR CODE HERE
    theta_counter = 0
    grid_size = int(np.sqrt(env.nS))
    old = np.zeros(env.nS)
    for cell in range(env.nS):
        p = env.P[cell]
        value = 0
        for action in range(grid_size):
            '''Since this is deterministic, the outcome (resulting state) of each action is known.
            Therefore, all we need to do to determine the value is sum the reward (with discount) after each action.
            However, for the sake of completeness I have left in the transition probability (p[action][0]).'''
            transition_prob = p[action][0][0]
            resulting_state = int(p[action][0][1])
            reward = p[action][0][2]
            print(p[action])
            
            value += transition_prob * (reward + V[resulting_state]*discount_factor)

        if value - V[cell] < theta:
            theta_counter += 1
        else:
            theta_counter = 0

        V[cell] = value

        
    return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    # YOUR CODE HERE
    theta_counter = 0
    grid_size = int(np.sqrt(env.nS))
    old = np.zeros(env.nS)
    for cell in range(env.nS):
        p = env.P[cell]
        value = 0
        for action in range(grid_size):
            '''Since this is deterministic, the outcome (resulting state) of each action is known.
            Therefore, all we need to do to determine the value is sum the reward (with discount) after each action.
            However, for the sake of completeness I have left in the transition probability (p[action][0]).'''
            transition_prob = p[action][0][0]
            resulting_state = int(p[action][0][1])
            reward = p[action][0][2]
            value += transition_prob * (reward + V[resulting_state]*discount_factor)

        if value - V[cell] < theta:
            theta_counter += 1
        else:
            theta_counter = 0

        V[cell] = value

        
    return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    # YOUR CODE HERE
    theta_counter = 0
    grid_size = int(np.sqrt(env.nS))
    old = np.zeros(env.nS)
    for cell in range(env.nS):
        p = env.P[cell]
        value = 0
        for action in range(grid_size):
            '''Since this is deterministic, the outcome (resulting state) of each action is known.
            Therefore, all we need to do to determine the value is sum the reward (with discount) after each action.
            However, for the sake of completeness I have left in the transition probability (p[action][0]).'''
            transition_prob = p[action][0][0]
            resulting_state = int(p[action][0][1])
            reward = p[action][0][2]
            value += transition_prob * (reward + V[resulting_state]*discount_factor)

        if value - V[cell] < theta:
            theta_counter += 1
            if theta_counter > 16:
                break
        else:
            theta_counter = 0

        V[cell] = value

        
    return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    # YOUR CODE HERE
    theta_counter = 0
    grid_size = int(np.sqrt(env.nS))
    old = np.zeros(env.nS)
    for cell in range(env.nS):
        p = env.P[cell]
        value = 0
        for action in range(grid_size):
            '''Since this is deterministic, the outcome (resulting state) of each action is known.
            Therefore, all we need to do to determine the value is sum the reward (with discount) after each action.
            However, for the sake of completeness I have left in the transition probability (p[action][0]).'''
            transition_prob = p[action][0][0]
            resulting_state = int(p[action][0][1])
            reward = p[action][0][2]
            print(resulting_state, reward)
            value += transition_prob * (reward + V[resulting_state]*discount_factor)

        if value - V[cell] < theta:
            theta_counter += 1
            if theta_counter > 16:
                break
        else:
            theta_counter = 0

        V[cell] = value

        
    return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)

    # YOUR CODE HERE
    theta_counter = 0
    grid_size = int(np.sqrt(env.nS))
    for cell in range(env.nS):
        p = env.P[cell]
        value = 0
        for action in range(grid_size):
            '''Since this is deterministic, the outcome (resulting state) of each action is known.
            Therefore, all we need to do to determine the value is sum the reward (with discount) after each action.
            However, for the sake of completeness I have left in the transition probability (p[action][0]).'''
            transition_prob = p[action][0][0]
            resulting_state = int(p[action][0][1])
            reward = p[action][0][2]
            value += transition_prob * (reward + V[resulting_state]*discount_factor)

        if value - V[cell] < theta:
            theta_counter += 1
            if theta_counter > 16:
                break
        else:
            theta_counter = 0

        V[cell] = value

        
    return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)

    # YOUR CODE HERE
    theta_counter = 0
    grid_size = int(np.sqrt(env.nS))
    for cell in range(env.nS):
        p = env.P[cell]

        value = 0
        for action in range(grid_size):
            '''Since this is deterministic, the outcome (resulting state) of each action is known.
            Therefore, all we need to do to determine the value is sum the reward (with discount) after each action.
            However, for the sake of completeness I have left in the transition probability (p[action][0]).'''
            print(p[action][0])
            transition_prob = p[action][0][0]
            resulting_state = int(p[action][0][1])
            reward = p[action][0][2]
            value += transition_prob * (reward + V[resulting_state]*discount_factor)

        if value - V[cell] < theta:
            theta_counter += 1
            if theta_counter > 16:
                break
        else:
            theta_counter = 0

        V[cell] = value

        
    return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)

    # YOUR CODE HERE
    theta_counter = 0
    grid_size = int(np.sqrt(env.nS))
    for cell in range(env.nS):
        value = 0
        for action in range(grid_size):
            '''Since this is deterministic, the outcome (resulting state) of each action is known.
            Therefore, all we need to do to determine the value is sum the reward (with discount) after each action.
            However, for the sake of completeness I have left in the transition probability (p[action][0]).'''
            p = env.P[cell][action][0]
            transition_prob = p[0]
            resulting_state = int(p[1])
            reward = p[2]
            value += transition_prob * (reward + V[resulting_state]*discount_factor)

        if value - V[cell] < theta:
            theta_counter += 1
            if theta_counter > 16:
                break
        else:
            theta_counter = 0

        V[cell] = value

        
    return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)

    # YOUR CODE HERE
    theta_counter = 0
    grid_size = int(np.sqrt(env.nS))
    for cell in range(env.nS):
        value = 0
        for action in range(grid_size):
            '''Since this is deterministic, the outcome (resulting state) of each action is known.
            Therefore, all we need to do to determine the value is sum the reward (with discount) after each action.
            However, for the sake of completeness I have left in the transition probability (p[action][0]).'''
            p = env.P[cell][action][0]
            transition_prob = p[0]
            resulting_state = int(p[1])
            reward = p[2]
            print(p[2])
            value += transition_prob * (reward + V[resulting_state]*discount_factor)

        if value - V[cell] < theta:
            theta_counter += 1
            if theta_counter > 16:
                break
        else:
            theta_counter = 0

        V[cell] = value

        
    return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)

    # YOUR CODE HERE
    theta_counter = 0
    grid_size = int(np.sqrt(env.nS))
    for cell in range(env.nS):
        value = 0
        for action in range(grid_size):
            '''Since this is deterministic, the outcome (resulting state) of each action is known.
            Therefore, all we need to do to determine the value is sum the reward (with discount) after each action.
            However, for the sake of completeness I have left in the transition probability (p[action][0]).'''
            p = env.P[cell][action][0]
            transition_prob = p[0]
            resulting_state = int(p[1])
            reward = p[2]
            print(p[2])
            value += transition_prob * (reward + V[resulting_state]*discount_factor)
            print(value)

        if value - V[cell] < theta:
            theta_counter += 1
            if theta_counter > 16:
                break
        else:
            theta_counter = 0

        V[cell] = value

        
    return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)

    # YOUR CODE HERE
    theta_counter = 0
    grid_size = int(np.sqrt(env.nS))
    for cell in range(env.nS):
        value = 0
        for action in range(grid_size):
            '''Since this is deterministic, the outcome (resulting state) of each action is known.
            Therefore, all we need to do to determine the value is sum the reward (with discount) after each action.
            However, for the sake of completeness I have left in the transition probability (p[action][0]).'''
            p = env.P[cell][action][0]
            transition_prob = p[0]
            resulting_state = int(p[1])
            reward = p[2]
            value += transition_prob * (reward + V[resulting_state]*discount_factor)

        if value - V[cell] < theta:
            theta_counter += 1
            print(theta_counter)
            if theta_counter > 16:
                break
        else:
            theta_counter = 0

        V[cell] = value

        
    return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)

    # YOUR CODE HERE
    theta_counter = 0
    grid_size = int(np.sqrt(env.nS))
    old = np.zeros(env.nS)
    cell = 0
    while theta_counter < 16:
        cell = (cell + 1)%env.nS
        value = 0
        for action in range(grid_size):
            '''Since this is deterministic, the outcome (resulting state) of each action is known.
            Therefore, all we need to do to determine the value is sum the reward (with discount) after each action.
            However, for the sake of completeness I have left in the transition probability (p[action][0]).'''
            p = env.P[cell][action][0]
            transition_prob = p[0]
            resulting_state = int(p[1])
            reward = p[2]
            value += transition_prob * (reward + V[resulting_state]*discount_factor)
        if abs(value - V[cell]) < theta:
            theta_counter += 1
        else:
            theta_counter = 0

        V[cell] = value

        
    return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)

    # YOUR CODE HERE
    theta_counter = 0
    grid_size = int(np.sqrt(env.nS))
    cell = 0
    while theta_counter < 16:
        cell = (cell + 1)%env.nS
        value = 0
        for action in range(grid_size):
            '''Since this is deterministic, the outcome (resulting state) of each action is known.
            Therefore, all we need to do to determine the value is sum the reward (with discount) after each action.
            However, for the sake of completeness I have left in the transition probability (p[action][0]).'''
            p = env.P[cell][action][0]
            transition_prob = p[0]
            resulting_state = int(p[1])
            reward = p[2]
            value += transition_prob * (reward + V[resulting_state]*discount_factor)
            
        if abs(value - V[cell]) < theta:
            theta_counter += 1
        else:
            theta_counter = 0

        V[cell] = value

        
    return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)

    # YOUR CODE HERE
    theta_counter = 0
    grid_size = int(np.sqrt(env.nS))
    cell = 0
    while theta_counter < 16:
        cell = (cell + 1)%env.nS
        value = 0
        for action in range(grid_size):
            '''Since this is deterministic, the outcome (resulting state) of each action is known.
            Therefore, all we need to do to determine the value is sum the reward (with discount) after each action.
            However, for the sake of completeness I have left in the transition probability (p[action][0]).'''
            p = env.P[cell][action][0]
            transition_prob = p[0]
            resulting_state = int(p[1])
            reward = p[2]
            value += transition_prob * (reward + V[resulting_state]*discount_factor)
            
        if abs(value - V[cell]) < theta:
            theta_counter += 1
            print(theta_counter)
        else:
            theta_counter = 0

        V[cell] = value

        
    return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)

    # YOUR CODE HERE
    theta_counter = 0
    grid_size = int(np.sqrt(env.nS))
    cell = 0
    while theta_counter < 16:
        cell = (cell + 1)%env.nS
        value = 0
        for action in range(grid_size):
            '''Since this is deterministic, the outcome (resulting state) of each action is known.
            Therefore, all we need to do to determine the value is sum the reward (with discount) after each action.
            However, for the sake of completeness I have left in the transition probability (p[action][0]).'''
            p = env.P[cell][action][0]
            transition_prob = p[0]
            resulting_state = int(p[1])
            reward = p[2]
            value += transition_prob * (reward + V[resulting_state]*discount_factor)
            
        if abs(value - V[cell]) < theta:
            theta_counter += 1
        else:
            theta_counter = 0

        print(theta_counter)

        V[cell] = value

        
    return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)

    # YOUR CODE HERE
    theta_counter = 0
    grid_size = int(np.sqrt(env.nS))
    cell = 0
    while theta_counter < 16:
        cell = (cell + 1)%env.nS
        value = 0
        for action in range(grid_size):
            '''Since this is deterministic, the outcome (resulting state) of each action is known.
            Therefore, all we need to do to determine the value is sum the reward (with discount) after each action.
            However, for the sake of completeness I have left in the transition probability (p[action][0]).'''
            p = env.P[cell][action][0]
            transition_prob = p[0]/grid_size
            resulting_state = int(p[1])
            reward = p[2]
            value += transition_prob * (reward + V[resulting_state]*discount_factor)
            
        if abs(value - V[cell]) < theta:
            theta_counter += 1
        else:
            theta_counter = 0

        print(theta_counter)

        V[cell] = value

        
    return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)

    # YOUR CODE HERE
    theta_counter = 0
    grid_size = int(np.sqrt(env.nS))
    cell = 0
    while theta_counter < 16:
        cell = (cell + 1)%env.nS
        value = 0
        for action in range(grid_size):
            '''Since this is deterministic, the outcome (resulting state) of each action is known.
            Therefore, all we need to do to determine the value is sum the reward (with discount) after each action.
            However, for the sake of completeness I have left in the transition probability (p[action][0]).'''
            p = env.P[cell][action][0]
            transition_prob = p[0]/grid_size
            resulting_state = int(p[1])
            reward = p[2]
            value += transition_prob * (reward + V[resulting_state]*discount_factor)
            
        if abs(value - V[cell]) < theta:
            theta_counter += 1
        else:
            theta_counter = 0

        V[cell] = value

        
    return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)

    # YOUR CODE HERE
    theta_counter = 0
    grid_size = int(np.sqrt(env.nS))
    cell = 0
    while theta_counter < 16:
        cell = (cell + 1)%env.nS
        value = 0
        for action in range(grid_size):
            '''Since this is deterministic, the outcome (resulting state) of each action is known.
            Therefore, all we need to do to determine the value is sum the reward (with discount) after each action.
            However, for the sake of completeness I have left in the transition probability (p[action][0]).'''
            p = env.P[cell][action][0]
            transition_prob = p[0]/grid_size
            resulting_state = int(p[1])
            reward = p[2]
            value += transition_prob * (reward + V[resulting_state]*discount_factor)
            
        if abs(value - V[cell]) < theta:
            theta_counter += 1
        else:
            theta_counter = 0

        V[cell] = value
    print(policy)
        
    return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)

    # YOUR CODE HERE
    theta_counter = 0
    grid_size = int(np.sqrt(env.nS))
    cell = 0
    while theta_counter < 16:
        cell = (cell + 1)%env.nS
        value = 0
        for action in range(grid_size):
            '''Since this is deterministic, the outcome (resulting state) of each action is known.
            Therefore, all we need to do to determine the value is sum the reward (with discount) after each action.
            However, for the sake of completeness I have left in the transition probability (p[action][0]).'''
            transition_prob = policy[cell][action]
            p = env.P[cell][action][0]
            resulting_state = int(p[1])
            reward = p[2]
            value += transition_prob * (reward + V[resulting_state]*discount_factor)
            
        if abs(value - V[cell]) < theta:
            theta_counter += 1
        else:
            theta_counter = 0

        V[cell] = value
        
    return np.array(V)

def policy_iter_v(env, policy_eval_v=policy_eval_v, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_v: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    # YOUR CODE HERE
    grid_size = int(np.sqrt(env.nS))
    V = policy_eval_v(policy, env, discount_factor=discount_factor)
    for cell in env.P:
        values = [V[int(env.P[cell][action][0][1])] for action in range(grid_size)]
        max_value = max(values)
        optimal_actions = [action for action  in range(grid_size) if value[action] == max_value]

        policy[cell, :] = 0
        for action in optimal_actions:
            policy[cell, action] = 1/len(optimal_actions)
                
    
    return policy, V

def policy_iter_v(env, policy_eval_v=policy_eval_v, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_v: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    # YOUR CODE HERE
    grid_size = int(np.sqrt(env.nS))
    V = policy_eval_v(policy, env, discount_factor=discount_factor)
    for cell in env.P:
        values = [V[int(env.P[cell][action][0][1])] for action in range(grid_size)]
        max_value = max(values)
        optimal_actions = [action for action  in range(grid_size) if values[action] == max_value]

        policy[cell, :] = 0
        for action in optimal_actions:
            policy[cell, action] = 1/len(optimal_actions)
                
    return policy, V
