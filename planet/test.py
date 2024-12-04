import os
import json
import torch
import gymnasium as gym
from .agent import CEMAgent
from .wrappers import RepeatAction
from .configs import TrainConfig, TestConfig
from .model import Encoder, RecurrentStateSpaceModel, RewardModel, ObservationModel


def test(env: gym.Env, test_config: TestConfig):

    env = RepeatAction(env, skip=test_config.action_repeat)
    
    # load trained models
    with open(os.path.join(test_config.model_dir, 'args.json'), 'r') as f:
        train_config = TrainConfig(**json.load(f))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    encoder = Encoder(
        observation_dim=env.observation_space.shape[0],
        obs_embedding_dim=train_config.obs_embedding_dim,
    ).to(device)

    rssm = RecurrentStateSpaceModel(
        state_dim=train_config.state_dim,
        action_dim=env.action_space.shape[0],
        rnn_hidden_dim=train_config.rnn_hidden_dim,
        obs_embedding_dim=train_config.obs_embedding_dim,
    ).to(device)
    
    reward_model = RewardModel(
        state_dim=train_config.state_dim,
        rnn_hidden_dim=train_config.rnn_hidden_dim,
        action_dim=env.action_space.shape[0],
    ).to(device)

    obs_model = ObservationModel(
        state_dim=train_config.state_dim,
        rnn_hidden_dim=train_config.rnn_hidden_dim,
        observation_dim=env.observation_space.shape[0],
    ).to(device)

    # load parameters
    encoder.load_state_dict(torch.load(os.path.join(test_config.model_dir, 'encoder.pth'), weights_only=True))
    rssm.load_state_dict(torch.load(os.path.join(test_config.model_dir, 'rssm.pth'), weights_only=True))
    reward_model.load_state_dict(torch.load(os.path.join(test_config.model_dir, 'reward_model.pth'), weights_only=True))
    obs_model.load_state_dict(torch.load(os.path.join(test_config.model_dir, 'obs_model.pth'), weights_only=True))


    # define agent
    cem_agent = CEMAgent(
        encoder=encoder,
        rssm=rssm,
        reward_model=reward_model,
        horizon=test_config.horizon,
        N_candidates=test_config.N_candidates,
        N_iterations=test_config.N_iterations,
        N_top_candidates=test_config.N_top_candidates
    )

    results = []

    # test learnged model in the environment
    for episode in range(test_config.episodes):
        cem_agent.reset()
        obs, info = env.reset(seed=test_config.env_seed, options=test_config.env_options)
        done = False
        rewards = []
        estimated_rewards = []
        states = []
        estimated_states = []
        observations = []
        recon_observations = []
        actions = []
        while not done:
            # Actual state
            state = info["state"]
            states.append(state)
            # Actual observation
            observations.append(obs)
            action, cem_info = cem_agent(obs)
            estimated_state_mean = cem_info["state_posterior_mean"]
            estimated_state_cov = cem_info["state_posterior_cov"]
            estimated_reward = cem_info["estimated_reward"]
            # Action
            actions.append(action)
            # Estimated state
            estimated_states.append((
                estimated_state_mean.cpu().numpy(),
                  estimated_state_cov.cpu().numpy(),
            ))
            # Estimated reward
            estimated_rewards.append(estimated_reward.cpu().numpy().item())
            # Reconstructed observation
            with torch.no_grad():
                recon_obs = obs_model(
                    state=estimated_state_mean,
                    rnn_hidden=cem_agent.rnn_hidden,
                )
                recon_observations.append(recon_obs.cpu().numpy().flatten())
            obs, reward, done, info = env.step(action)
            # Actual reward
            rewards.append(reward)
        
        result = {
            "episode": episode,
            "result": {
                "rewards": rewards,
                "estimated_rewards": estimated_rewards,
                "states": states,
                "estimated_states": estimated_states,
                "observations": observations,
                "reconstructed_observations": recon_observations,
                "actions": actions,
            }
        }

        results.append(result)

    return results