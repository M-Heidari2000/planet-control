import torch
import einops
from torch.distributions import Normal, Uniform


class CEMAgent:
    """
        Action planning by Cross Entropy Method (CEM)
    """
    def __init__(
        self,
        rssm,
        reward_model,
        observation_model,
        planning_horizon,
        num_iterations: int,
        num_candidates: int,
        num_elites: int,
    ):
        self.rssm = rssm
        self.reward_model = reward_model
        self.observation_model = observation_model

        self.num_iterations = num_iterations
        self.num_candidates = num_candidates
        self.num_elites = num_elites
        self.planning_horizon = planning_horizon

        self.device = next(self.reward_model.parameters()).device
        # Initialize the rnn_hidden to zero vector
        self.rnn_hidden = torch.zeros(1, rssm.rnn_hidden_dim, device=self.device)

    def __call__(self, obs, exploration_noise_var: float):
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)

        with torch.no_grad():
            # Compute starting state for planning
            # While taking information from the current observation
            state_posterior = self.rssm.posterior(self.rnn_hidden, obs)

            # Initialize action distribution ~ N(0, I)
            action_dist = Normal(
                torch.zeros((self.horizon, self.rssm.action_dim), device=self.device),
                torch.ones((self.horizon, self.rssm.action_dim), device=self.device),
            )
    
            # Iteratively improve action distribution with CEM
            for _ in range(self.N_iterations):
                # Sample action candidates
                # transpose to (horizon, N_candidates, action_dim) for parallel exploration
                action_candidates = action_dist.sample([self.N_candidates])
                action_candidates = einops.rearrange(action_candidates, 'n h a -> h n a')
    
                # Initialize reward, state, and rnn hidden state
                # The size of state is (self.N_acndidates, state_dim)
                # The size of rnn hidden is (self.N_candidates, rnn_hidden_dim)
                # These are for parallel exploration
                state = state_posterior.sample([self.N_candidates]).squeeze(-2)
                rnn_hidden = self.rnn_hidden.repeat([self.N_candidates, 1])
                total_predicted_reward = torch.zeros(self.N_candidates, device=self.device)
                observation_trajectories = torch.zeros(
                    (self.planning_horizon, self.num_candidates, obs.shape[1]),
                    device=self.device,
                )

                # Compute total predicted reward by open-loop prediction using prior
                for t in range(self.horizon):
                    observation_trajectories[t] = self.observation_model(state=state)
                    total_predicted_reward += self.reward_model(
                        state=state,
                        action=torch.tanh(action_candidates[t]),
                        rnn_hidden=rnn_hidden,
                    ).squeeze()
                    next_state_prior, rnn_hidden = self.rssm.prior(
                        state=state,
                        action=torch.tanh(action_candidates[t]),
                        rnn_hidden=rnn_hidden,
                    )
                    # Since we are passing a batch, output dustribution has the right shape
                    # So no need to specify the dimensions for sampling!
                    state = next_state_prior.sample()
                    
                # find elites
                elite_indexes = total_predicted_reward.argsort(descending=True)[:self.num_elites]
                elites = action_candidates[:, elite_indexes, :]

                # fit a new distribution to the elites
                mean = elites.mean(dim=1)
                std = elites.std(dim=1, unbiased=False)
                action_dist.loc = mean
                action_dist.scale = std
    
            # Return only the first action (Model Predictive Control)
            actions = torch.tanh(mean)
            if exploration_noise_var > 0:
                actions += torch.randn_like(actions) * torch.sqrt(exploration_noise_var)
            best_trajectory = observation_trajectories[:, elite_indexes, :].mean(dim=1)

            # Update RNN hidden state for next step planning
            _, self.rnn_hidden = self.rssm.prior(
                state_posterior.sample(),
                torch.tanh(actions.unsqueeze(0)),
                self.rnn_hidden
            )

        return actions.cpu().numpy(), best_trajectory.cpu().numpy()

    def reset(self):
        self.rnn_hidden = torch.zeros(1, self.rssm.rnn_hidden_dim, device=self.device)


class RSAgent:
    """
        Action planning by Random Shooting method
    """
    def __init__(
        self,
        rssm,
        reward_model,
        observation_model,
        planning_horizon,
        num_candidates: int,
    ):
        self.rssm = rssm
        self.reward_model = reward_model
        self.observation_model = observation_model

        self.num_candidates = num_candidates
        self.planning_horizon = planning_horizon

        self.device = next(self.reward_model.parameters()).device
        # Initialize the rnn_hidden to zero vector
        self.rnn_hidden = torch.zeros(1, rssm.rnn_hidden_dim, device=self.device)

    def __call__(self, obs, exploration_noise_var: float):
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)

        with torch.no_grad():
            # Compute starting state for planning
            # While taking information from the current observation
            state_posterior = self.rssm.posterior(self.rnn_hidden, obs)

            action_dist = Uniform(
                low=-torch.ones((self.planning_horizon, self.posterior_model.action_dim), device=self.device),
                high=torch.ones((self.planning_horizon, self.posterior_model.action_dim),device=self.device),
            )
    
            action_candidates = action_dist.sample([self.N_candidates])
            action_candidates = einops.rearrange(action_candidates, 'n h a -> h n a')


            state = state_posterior.sample([self.N_candidates]).squeeze(-2)
            rnn_hidden = self.rnn_hidden.repeat([self.N_candidates, 1])
            total_predicted_reward = torch.zeros(self.N_candidates, device=self.device)
            observation_trajectories = torch.zeros(
                (self.planning_horizon, self.num_candidates, obs.shape[1]),
                device=self.device,
            )

            # Compute total predicted reward by open-loop prediction using prior
            for t in range(self.horizon):
                observation_trajectories[t] = self.observation_model(state=state)
                total_predicted_reward += self.reward_model(
                    state=state,
                    action=action_candidates[t],
                    rnn_hidden=rnn_hidden,
                ).squeeze()
                next_state_prior, rnn_hidden = self.rssm.prior(
                    state=state,
                    action=action_candidates[t],
                    rnn_hidden=rnn_hidden,
                )
                # Since we are passing a batch, output dustribution has the right shape
                # So no need to specify the dimensions for sampling!
                state = next_state_prior.sample()
                    
            # find the best action sequence
            max_index = total_predicted_reward.argmax()
            actions = action_candidates[:, max_index, :]
            best_trajectory = observation_trajectories[:, max_index, :]
    
            if exploration_noise_var > 0:
                actions += torch.randn_like(actions) * torch.sqrt(exploration_noise_var)
            best_trajectory = observation_trajectories[:, max_index, :]

            # Update RNN hidden state for next step planning
            _, self.rnn_hidden = self.rssm.prior(
                state_posterior.sample(),
                actions[0],
                self.rnn_hidden
            )

        return actions.cpu().numpy(), best_trajectory.cpu().numpy()

    def reset(self):
        self.rnn_hidden = torch.zeros(1, self.rssm.rnn_hidden_dim, device=self.device)


