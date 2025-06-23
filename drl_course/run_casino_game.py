import pyRDDLGym
from pyRDDLGym.core.policy import RandomAgent
import numpy as np

domain_path = "./casino_game/domain.rddl"
instance_path = "./casino_game/instance.rddl"

# Number of episodes to run
num_episodes = 1001
all_rewards = []

for episode in range(num_episodes):
    env = pyRDDLGym.make(domain_path, instance_path)

    # Create a random agent
    agent = RandomAgent(
        action_space=env.action_space, num_actions=env.max_allowed_actions
    )

    # Run the episode
    total_reward = 0
    state, _ = env.reset()

    for step in range(env.horizon):
        # Alternate between a2 and a1
        if step % 2 == 0:
            action = {"action": "a2"}
        else:
            action = {"action": "a1"}

        next_state, reward, done, info, _ = env.step(action)
        total_reward += reward

        # Only print details for the first episode
        if episode == 0:
            print(f"Episode 1, step = {step}")
            print(f"state      = s{int(state['state'])}")
            print(f"action     = {str(action['action'])}")
            print(f"next state = s{int(next_state['state'])}")
            print(f"reward     = {reward}\n")

        state = next_state
        if done:
            break

    all_rewards.append(total_reward)
    env.close()

    # Print progress every 100 episodes
    if (episode + 1) % 1000 == 0:
        print(f"Completed {episode + 1} episodes")

# Calculate statistics
average_reward = np.mean(all_rewards)
std_reward = np.std(all_rewards)
min_reward = np.min(all_rewards)
max_reward = np.max(all_rewards)

print("\n===== Results =====")
print(f"Number of episodes: {num_episodes}")
print(f"Average total reward: {average_reward:.4f}")
print(f"Standard deviation: {std_reward:.4f}")
print(f"Min reward: {min_reward:.4f}")
print(f"Max reward: {max_reward:.4f}")
