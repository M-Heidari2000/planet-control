import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal


class Encoder(nn.Module):
    """
        Encoder embeds observations to vector
    """
    def __init__(
        self,
        observation_dim: int,
        obs_embedding_dim: int,
    ):
        super().__init__()
        self.mlp_layers = nn.Sequential(
            nn.Linear(observation_dim, 2 * obs_embedding_dim),
            nn.ReLU(),
            nn.Linear(2 * obs_embedding_dim, obs_embedding_dim),
            nn.Flatten(start_dim=1),
        )

    def forward(self, obs):
        embedded_obs = self.mlp_layers(obs)
        return embedded_obs
    

class RewardModel(nn.Module):
    """
        p(r_t | s_t, a_t, h_t)
        Reward model to predict reward from state, actions and rnn hidden state
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        rnn_hidden_dim: int,
        hidden_dim: int=64,
    ):
        super().__init__()
        self.mlp_layers = nn.Sequential(
            nn.Linear(state_dim + action_dim + rnn_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        state,
        action,
        rnn_hidden,
    ):
        reward = self.mlp_layers(
            torch.cat([state, action, rnn_hidden], dim=1),
        )
        return reward
    

class ObservationModel(nn.Module):
    """
        p(o_t | s_t, h_t)
        Observation model to reconstruct observation from state and rnn hidden state
    """
    def __init__(
        self,
        state_dim: int,
        rnn_hidden_dim: int,
        observation_dim: int,
    ):
        super().__init__()
        self.mlp_layers = nn.Sequential(
            nn.Linear(state_dim + rnn_hidden_dim, observation_dim),
            nn.ReLU(),
            nn.Linear(observation_dim, observation_dim),
        )

    def forward(
        self,
        state,
        rnn_hidden,
    ):
        obs = self.mlp_layers(
            torch.cat([state, rnn_hidden], dim=1)
        )
        return obs
    

class RecurrentStateSpaceModel(nn.Module):
    """
        This class includes multiple components
        Deterministic state model: h_t+1 = f(h_t, s_t, a_t)
        Stochastic state model (prior): p(s_t+1 | h_t+1)
        State posterior: q(s_t | h_t, o_t)
        NOTE: actually, this class takes embedded observation by Encoder class
        min_stddev is added to stddev same as original implementation
        Activation function for this class is ReLU same as original implementation
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        rnn_hidden_dim: int,
        obs_embedding_dim: int,
        hidden_dim: int=64,
        min_stddev: float=0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self._min_stddev = min_stddev

        # RNN
        self.rnn = nn.GRUCell(
            input_size=hidden_dim,
            hidden_size=rnn_hidden_dim,
        )

        # Fully Connected layers
        self.fc_state_action = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_rnn_hidden = nn.Sequential(
            nn.Linear(rnn_hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_state_mean_prior = nn.Linear(hidden_dim, state_dim)
        self.fc_state_stddev_prior = nn.Sequential(
            nn.Linear(hidden_dim, state_dim),
            nn.Softplus(),
        )
        self.fc_rnn_hidden_embedded_obs = nn.Sequential(
            nn.Linear(rnn_hidden_dim + obs_embedding_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_state_mean_posteriori = nn.Linear(hidden_dim, state_dim)
        self.fc_state_stddev_posteriori = nn.Sequential(
            nn.Linear(hidden_dim, state_dim),
            nn.Softplus(),
        )

    def prior(self, state, action, rnn_hidden):
        """
            h_t+1 = f(h_t, s_t, a_t)
            Compute prior p(s_t+1 | h_t+1)
        """     
        rnn_input = self.fc_state_action(
            torch.cat([state, action], dim=1),
        )
        rnn_hidden = self.rnn(rnn_input, rnn_hidden)
        x = self.fc_rnn_hidden(rnn_hidden)

        mean = self.fc_state_mean_prior(x)
        stddev = self.fc_state_stddev_prior(x) + self._min_stddev
        return Normal(mean, stddev), rnn_hidden

    def posterior(self, rnn_hidden, embedded_obs):
        """
            Compute posterior q(s_t | h_t, o_t)
        """
        x = self.fc_rnn_hidden_embedded_obs(
            torch.cat([rnn_hidden, embedded_obs], dim=1)
        )
        mean = self.fc_state_mean_posteriori(x)
        stddev = self.fc_state_stddev_posteriori(x) + self._min_stddev
        return Normal(mean, stddev)

    def forward(
        self,
        state,
        action,
        rnn_hidden,
        embedded_next_obs,
    ):
        """
            h_t+1 = f(h_t, s_t, a_t)
            Return prior p(s_t+1 | h_t+1) and posterior q(s_t+1 | h_t+1, o_t+1)
            for model training
        """
        next_state_prior, rnn_hidden = self.prior(state, action, rnn_hidden)
        next_state_posterior = self.posterior(rnn_hidden, embedded_next_obs)
        return next_state_prior, next_state_posterior, rnn_hidden