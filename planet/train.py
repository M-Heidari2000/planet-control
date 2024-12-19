import os
import time
import json
import torch
import einops
import numpy as np
import gymnasium as gym
from datetime import datetime
from torch.optim import Adam
from torch.distributions.kl import kl_divergence
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from .agent import CEMAgent
from .utils import ReplayBuffer
from .wrappers import RepeatAction
from .configs import TrainConfig
from .model import (
    Encoder,
    RecurrentStateSpaceModel,
    ObservationModel,
    RewardModel
)


def train(env: gym.Env, config: TrainConfig):

    # Prepare logging
    log_dir = os.path.join(config.log_dir, datetime.now().strftime('%Y%m%d_%H%M'))
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, 'args.json'), 'w') as f:
        json.dump(config.dict(), f)
    writer = SummaryWriter(log_dir=log_dir)

    # set seed (NOTE: some randomness is still remaining (e.g. cuDNN's randomness))
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    env = RepeatAction(env, skip=config.action_repeat)

    # define replay buffer
    replay_buffer = ReplayBuffer(
        capacity=config.buffer_capacity,
        observation_shape=env.observation_space.shape[0],
        action_shape=env.action_space.shape[0]
    )

    # define models and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = Encoder(
        observation_dim=env.observation_space.shape[0],
        obs_embedding_dim=config.obs_embedding_dim,
    ).to(device)

    rssm = RecurrentStateSpaceModel(
        state_dim=config.state_dim,
        action_dim=env.action_space.shape[0],
        rnn_hidden_dim=config.rnn_hidden_dim,
        obs_embedding_dim=config.obs_embedding_dim,
    ).to(device)

    obs_model = ObservationModel(
        state_dim=config.state_dim,
        rnn_hidden_dim=config.rnn_hidden_dim,
        observation_dim=env.observation_space.shape[0],
    ).to(device)
    
    reward_model = RewardModel(
        state_dim=config.state_dim,
        rnn_hidden_dim=config.rnn_hidden_dim,
        action_dim=env.action_space.shape[0],
    ).to(device)

    all_params = (
        list(encoder.parameters()) +
        list(rssm.parameters()) +
        list(obs_model.parameters()) +
        list(reward_model.parameters())
    )
    
    optimizer = Adam(all_params, lr=config.lr, eps=config.eps)

    # collect initial experience with random action
    for episode in range(config.seed_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_obs, reward, done, _ = env.step(action)
            replay_buffer.push(obs, action, reward, done)
            obs = next_obs

    # main training loop
    for episode in range(config.seed_episodes, config.all_episodes):
        # collect experiences
        start = time.time()
        cem_agent = CEMAgent(
            encoder=encoder,
            rssm=rssm,
            reward_model=reward_model,
            horizon=config.horizon,
            N_candidates=config.N_candidates,
            N_iterations=config.N_iterations,
            N_top_candidates=config.N_top_candidates
        )

        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = cem_agent(obs)
            action += np.random.normal(
                0,
                np.sqrt(config.action_noise_var),
                env.action_space.shape[0]
            )
            next_obs, reward, done, _ = env.step(action)
            replay_buffer.push(obs, action, reward, done)
            obs = next_obs
            total_reward += reward

        writer.add_scalar('total reward at train', total_reward, episode)
        print('episode [%4d/%4d] is collected. Total reward is %f' %
              (episode+1, config.all_episodes, total_reward))
        print('elasped time for interaction: %.2fs' % (time.time() - start))

        # update model parameters
        start = time.time()
        for update_step in range(config.collect_interval):
            observations, actions, rewards, _ = replay_buffer.sample(
                batch_size=config.batch_size,
                chunk_length=config.chunk_length,
            )

            observations = torch.as_tensor(observations, device=device)
            observations = einops.rearrange(observations, 'b l o -> l b o')
            actions = torch.as_tensor(actions, device=device)
            actions = einops.rearrange(actions, 'b l a -> l b a')
            rewards = torch.as_tensor(rewards, device=device)
            rewards = einops.rearrange(rewards, 'b l r -> l b r')

            # embed observations
            embedded_observations = encoder(
                einops.rearrange(observations, 'b l o -> (b l) o')
            ).reshape(config.chunk_length, config.batch_size, -1)

            # prepare Tensor to maintain states sequence and rnn hidden states sequence
            states = torch.zeros(
                (config.chunk_length, config.batch_size, config.state_dim),
                device=device
            )
            rnn_hiddens = torch.zeros(
                (config.chunk_length, config.batch_size, config.rnn_hidden_dim),
                device=device
            )

            # initialize state and rnn hidden state with 0 vector
            state = torch.zeros(config.batch_size, config.state_dim, device=device)
            rnn_hidden = torch.zeros(config.batch_size, config.rnn_hidden_dim, device=device)

            # compute state and rnn hidden sequences and kl loss
            kl_loss = 0
            for l in range(config.chunk_length-1):
                next_state_prior, next_state_posterior, rnn_hidden = rssm(
                    state=state,
                    action=actions[l],
                    rnn_hidden=rnn_hidden,
                    embedded_next_obs=embedded_observations[l+1]
                )
                state = next_state_posterior.rsample()
                states[l+1] = state
                rnn_hiddens[l+1] = rnn_hidden
                kl = kl_divergence(next_state_prior, next_state_posterior).sum(dim=1)
                kl_loss += kl.clamp(min=config.free_nats).mean()
            kl_loss /= (config.chunk_length - 1)

            # compute reconstructed observations and predicted rewards
            flatten_states = states.reshape(-1, config.state_dim)
            flatten_actions = actions.reshape(-1, env.action_space.shape[0])
            flatten_rnn_hiddens = rnn_hiddens.reshape(-1, config.rnn_hidden_dim)
            recon_observations = obs_model(
                state=flatten_states,
                rnn_hidden=flatten_rnn_hiddens
            ).reshape(config.chunk_length, config.batch_size, env.observation_space.shape[0])
            predicted_rewards = reward_model(
                state=flatten_states,
                action=flatten_actions,
                rnn_hidden=flatten_rnn_hiddens,
            ).reshape(config.chunk_length, config.batch_size, 1)

            # compute loss for observation and reward
            # Since the variance of these models are I, MLE is equivalent to MSE
            obs_loss = 0.5 * mse_loss(
                recon_observations[1:],
                observations[1:],
                reduction='none'
            ).mean([0, 1]).sum()
            reward_loss = 0.5 * mse_loss(predicted_rewards[1:], rewards[1:])

            # add all losses and update model parameters with gradient descent
            loss = kl_loss + obs_loss + reward_loss
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(all_params, config.clip_grad_norm)
            optimizer.step()

            # print losses and add tensorboard
            print('update_step: %3d loss: %.5f, kl_loss: %.5f, obs_loss: %.5f, reward_loss: % .5f'
                  % (update_step+1,
                     loss.item(), kl_loss.item(), obs_loss.item(), reward_loss.item()))
            total_update_step = episode * config.collect_interval + update_step
            writer.add_scalar('overall loss', loss.item(), total_update_step)
            writer.add_scalar('kl loss', kl_loss.item(), total_update_step)
            writer.add_scalar('obs loss', obs_loss.item(), total_update_step)
            writer.add_scalar('reward loss', reward_loss.item(), total_update_step)

        print('elasped time for update: %.2fs' % (time.time() - start))

        # test to get score without exploration noise
        if (episode + 1) % config.test_interval == 0:
            start = time.time()
            cem_agent = CEMAgent(
                encoder=encoder,
                rssm=rssm,
                reward_model=reward_model,
                horizon=config.horizon,
                N_candidates=config.N_candidates,
                N_iterations=config.N_iterations,
                N_top_candidates=config.N_top_candidates
            )
            obs, _ = env.reset()
            done = False
            total_reward = 0
            while not done:
                action = cem_agent(obs)
                obs, reward, done, _ = env.step(action)
                total_reward += reward

            writer.add_scalar('total reward at test', total_reward, episode)
            print('Total test reward at episode [%4d/%4d] is %f' %
                  (episode+1, config.all_episodes, total_reward))
            print('elasped time for test: %.2fs' % (time.time() - start))

    # save learned model parameters
    torch.save(encoder.state_dict(), os.path.join(log_dir, 'encoder.pth'))
    torch.save(rssm.state_dict(), os.path.join(log_dir, 'rssm.pth'))
    torch.save(obs_model.state_dict(), os.path.join(log_dir, 'obs_model.pth'))
    torch.save(reward_model.state_dict(), os.path.join(log_dir, 'reward_model.pth'))
    writer.close()
    
    return {"model_dir": log_dir}
