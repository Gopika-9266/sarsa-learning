# SARSA Learning Algorithm


## AIM
To develop a Python program to find the optimal policy for the given RL environment using SARSA-Learning and compare the state values with the Monte Carlo method.

## PROBLEM STATEMENT
Train agent with SARSA in Gym environment, making sequential decisions for maximizing cumulative rewards.

## SARSA LEARNING ALGORITHM
### Step 1:
Initialize the Q-table with random values for all state-action pairs.

### Step 2:
Initialize the current state S and choose the initial action A using an epsilon-greedy policy based on the Q-values in the Q-table.

### Step 3:
Repeat until the episode ends and then take action A and observe the next state S' and the reward R.

### Step 4:
Update the Q-value for the current state-action pair (S, A) using the SARSA update rule.

### Step 5:
Update State and Action and repeat the step 3 untill the episodes ends.

## SARSA LEARNING FUNCTION
```
Developed by: Gopika R
Register no: 212222240031
```
```
def sarsa(env,
          gamma=1.0,
          init_alpha=0.5,
          min_alpha=0.01,
          alpha_decay_ratio=0.5,
          init_epsilon=1.0,
          min_epsilon=0.1,
          epsilon_decay_ratio=0.9,
          n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)

    # Decay schedules for alpha and epsilon
    alphas = decay_schedule(init_alpha,
                           min_alpha,
                           alpha_decay_ratio,
                           n_episodes)
    epsilons = decay_schedule(init_epsilon,
                              min_epsilon,
                              epsilon_decay_ratio,
                              n_episodes)

    # Function to select action using epsilon-greedy policy
    select_action = lambda state, Q, epsilon: np.argmax(Q[state]) \
        if np.random.random() > epsilon \
        else np.random.randint(len(Q[state]))

    for e in tqdm(range(n_episodes), leave=False):
        state = env.reset() # Reset the environment for each episode
        # In newer gymnasium versions, reset might return a tuple (state, info).
        # We only need the state here.
        if isinstance(state, tuple):
            state = state[0]

        action = select_action(state, Q, epsilons[e]) # Select the first action
        done = False # Initialize done flag for the episode

        while not done:
            # Take a step in the environment
            # Modify: Use try-except to handle both 4-tuple (older gym) and 5-tuple (newer gymnasium)
            try:
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated # Update the done flag for 5-tuple
            except ValueError:
                # If unpacking 5 fails, try unpacking 4 (older gym behavior)
                next_state, reward, done, _ = env.step(action)
                # The 'done' flag is directly available in the 4-tuple

            # Select the next action using the same policy
            next_action = select_action(next_state, Q, epsilons[e])

            # SARSA update rule
            # Use the updated 'done' flag in the TD target calculation
            td_target = reward + gamma * Q[next_state][next_action] * (not done)
            td_error = td_target - Q[state][action]
            Q[state][action] = Q[state][action] + alphas[e] * td_error

            # Update state and action for the next iteration
            state = next_state
            action = next_action

        # Store Q and policy track
        Q_track[e] = Q
        pi_track.append(np.argmax(Q, axis=1))

    # Calculate V and pi from the learned Q
    V = np.max(Q, axis=1)
    pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return Q, V, pi, Q_track, pi_track
```

## OUTPUT:
![image](https://github.com/user-attachments/assets/137a4b71-fe38-483e-b896-4fb1a1c42178)

![image](https://github.com/user-attachments/assets/05ba7be3-1723-4449-a7bd-838659199482)

![image](https://github.com/user-attachments/assets/c1e92f8f-a8ac-4396-8774-b15e1798fd27)

![image](https://github.com/user-attachments/assets/c76b19ff-5ade-46ed-bfed-3fa330f0c816)


## RESULT:

Thus to develop a Python program to find the optimal policy for the given RL environment using SARSA-Learning and compare the state values with the Monte Carlo method has been implemented successfully
