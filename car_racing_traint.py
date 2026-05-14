import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Create the environment
env = make_vec_env("CarRacing-v3", n_envs=1)

# Create PPO model with tensorboard logging
model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log="./car_racing_tensorboard/",
    device="cpu"
)

# Train for 10000 timesteps
model.learn(total_timesteps=10000)

# Save the model
model.save("car_racing_ppo")

print("Done Training!")
env.close()
