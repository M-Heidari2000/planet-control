import torch
import einops
from torch.distributions import Normal


class CEMAgent:
    """
        Action planning by Cross Entropy Method (CEM) in learned RSSM Model
    """
    def __init__(
        self,
        encoder,
        rssm,
        reward_model,
        horizon,
        N_iterations,
        N_candidates,
        N_top_candidates,
    ):
        self.encoder = encoder
        self.rssm = rssm
        self.reward_model = reward_model

        self.horizon = horizon
        self.N_iterations = N_iterations
        self.N_candidates = N_candidates
        self.N_top_candidates = N_top_candidates

        self.device = next(self.reward_model.parameters()).device
        # Initialize the rnn_hidden to zero vector
        self.rnn_hidden = torch.zeros(1, rssm.rnn_hidden_dim, device=self.device)

    def __call__(self, obs):
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)

        with torch.no_grad():
            # Compute starting state for planning
            # While taking information from the current observation
            embedded_obs = self.encoder(obs)
            state_posterior = self.rssm.posterior(self.rnn_hidden, embedded_obs)

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
                total_predicted_reward = torch.zeros(self.N_candidates, device=self.device)
                state = state_posterior.sample([self.N_candidates]).squeeze(-2)
                rnn_hidden = self.rnn_hidden.repeat([self.N_candidates, 1])
    
                # Compute total predicted reward by open-loop prediction using prior
                for t in range(self.horizon):
                    total_predicted_reward += self.reward_model(
                        state=state,
                        action=action_candidates[t],
                        rnn_hidden=rnn_hidden,
                    ).squeeze()
                    next_state_prior, rnn_hidden = self.rssm.prior(
                        state, action_candidates[t], rnn_hidden
                    )
                    # Since we are passing a batch, output dustribution has the right shape
                    # So no need to specify the dimensions for sampling!
                    state = next_state_prior.sample()
                    
                # Find top-k samples (elites)
                top_indexes = total_predicted_reward.argsort(descending=True)[:self.N_top_candidates]
                elites = action_candidates[:, top_indexes, :]
    
                # Fit a new distribution to the top-k candidates
                mean = elites.mean(dim=1)
                stddev = (
                    elites - mean.unsqueeze(1)
                ).abs().sum(dim=1) / (self.N_top_candidates - 1)
                action_dist = Normal(mean, stddev)
    
            # Return only the first action (Model Predictive Control)
            action = mean[0]

            # Estimate reward
            estimated_reward = self.reward_model(
                state=state_posterior.sample(),
                action=action.unsqueeze(0),
                rnn_hidden=self.rnn_hidden,
            )
    
            # Update RNN hidden state for next step planning
            _, self.rnn_hidden = self.rssm.prior(
                state_posterior.sample(),
                action.unsqueeze(0),
                self.rnn_hidden
            )

        info = {
            "state_posterior_mean": state_posterior.loc,
            "state_posterior_cov": state_posterior.scale,
            "estimated_reward": estimated_reward,
        }

        return action.cpu().numpy(), info

    def reset(self):
        self.rnn_hidden = torch.zeros(
            1, self.rssm.rnn_hidden_dim, device=self.device
        )