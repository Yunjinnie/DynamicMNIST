import torch
import torch.nn as nn
import numpy as np
import cv2 as cv
import gymnasium as gym
import logging
import tqdm
import os
import matplotlib.pyplot as plt

class InitialWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, no_ops: int = 0, repeat: int = 1):
        super(InitialWrapper, self).__init__(env)
        self.repeat = repeat
        self.no_ops = no_ops
   

        self.op_counter = 0

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        if self.op_counter < self.no_ops:
            obs, reward, done, info = self.env.step(0)
            self.op_counter += 1

        total_reward = 0.0
        done = False
        for _ in range(self.repeat):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break

        return obs, total_reward, done, info


class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, new_shape: Sequence[int] = (128, 128, 3), grayscale: bool = False):
        super(PreprocessFrame, self).__init__(env)
        self.shape = new_shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=self.shape, dtype=np.float32)
        self.grayscale = grayscale

        if self.grayscale:
            self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(*self.shape[:-1], 1), dtype=np.float32)

    def observation(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.astype(np.uint8)
        new_frame = cv.resize(obs, self.shape[:-1], interpolation=cv.INTER_AREA)
        if self.grayscale:
            new_frame = cv.cvtColor(new_frame, cv.COLOR_RGB2GRAY)
            new_frame = np.expand_dims(new_frame, -1)

        torch_frame = torch.from_numpy(new_frame).float()
        torch_frame = torch_frame / 255.0

        return torch_frame

def make_env(env_name: str, new_shape: Sequence[int] = (128, 128, 3), grayscale: bool = True, **kwargs):
    env = gym.make(env_name, **kwargs)
    env = PreprocessFrame(env, new_shape, grayscale=grayscale)
    return env

class EncoderCNN(nn.Module):
    def __init__(self, in_channels: int, embedding_dim: int = 2048, input_shape: Tuple[int, int] = (128, 128)):
        super(EncoderCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.fc1 = nn.Linear(self._compute_conv_output((in_channels, input_shape[0], input_shape[1])), embedding_dim)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

    def _compute_conv_output(self, shape: Tuple[int, int, int]):
        with torch.no_grad():
            x = torch.randn(1, shape[0], shape[1], shape[2])
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)

            return x.shape[1] * x.shape[2] * x.shape[3]


    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.bn1(x)
        x = torch.relu(self.conv2(x))
        x = self.bn2(x)

        x = torch.relu(self.conv3(x))
        x = self.bn3(x)

        x = self.conv4(x)
        x = self.bn4(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x
    
class DecoderCNN(nn.Module):
    def __init__(self, hidden_size: int, state_size: int,  embedding_size: int,
                 use_bn: bool = True, output_shape: Tuple[int, int] = (3, 128, 128)):
        super(DecoderCNN, self).__init__()

        self.output_shape = output_shape

        self.embedding_size = embedding_size
        self.fc1 = nn.Linear(hidden_size + state_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, 256 * (output_shape[1] // 16) * (output_shape[2] // 16))

        self.conv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)  # ×2
        self.conv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)  # ×2
        self.conv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)  # ×2
        self.conv4 = nn.ConvTranspose2d(32, output_shape[0], kernel_size=3, stride=2, padding=1, output_padding=1)

        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(32)

        self.use_bn = use_bn


    def forward(self, h: torch.Tensor, s: torch.Tensor):
        x = torch.cat([h, s], dim=-1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)

        x = x.view(-1, 256, self.output_shape[1] // 16, self.output_shape[2] // 16)

        if self.use_bn:
            x = torch.relu(self.bn1(self.conv1(x)))
            x = torch.relu(self.bn2(self.conv2(x)))
            x = torch.relu(self.bn3(self.conv3(x)))

        else:
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))

        x = self.conv4(x)

        return x
    
class RewardModel(nn.Module):
    def __init__(self, hidden_dim: int, state_dim: int):
        super(RewardModel, self).__init__()

        self.fc1 = nn.Linear(hidden_dim + state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 2)

    def forward(self, h: torch.Tensor, s: torch.Tensor):
        x = torch.cat([h, s], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    

class DynamicsModel(nn.Module):
    def __init__(self, hidden_dim: int, action_dim: int, state_dim: int, embedding_dim: int, rnn_layer: int = 1):
        super(DynamicsModel, self).__init__()

        self.hidden_dim = hidden_dim
        
        # Can be any recurrent network
        self.rnn = nn.ModuleList([nn.GRUCell(hidden_dim, hidden_dim) for _ in range(rnn_layer)])
        
        # Projection layer to make efficient use of concatenated inputs
        self.project_state_action = nn.Linear(action_dim + state_dim, hidden_dim)
        
        # Return mean and log-variance of the normal distribution
        self.prior = nn.Linear(hidden_dim, state_dim * 2)
        self.project_hidden_action = nn.Linear(hidden_dim + action_dim, hidden_dim)
        
        # Return mean and log-variance of the normal distribution
        self.posterior = nn.Linear(hidden_dim, state_dim * 2)
        self.project_hidden_obs = nn.Linear(hidden_dim + embedding_dim, hidden_dim)

        self.state_dim = state_dim

        self.act_fn = nn.ReLU()

    def forward(self, prev_hidden: torch.Tensor, prev_state: torch.Tensor, actions: torch.Tensor,
                obs: torch.Tensor = None, dones: torch.Tensor = None):
        """
        Forward pass of the dynamics model for one time step.
        :param prev_hidden: Previous hidden state of the RNN: (batch_size, hidden_dim)
        :param prev_state: Previous stochastic state: (batch_size, state_dim)
        :param action: One hot encoded actions: (sequence_length, batch_size, action_dim)
        :param obs: This is the encoded observation from the encoder, not the raw observation!: (sequence_length, batch_size, embedding_dim)
        :param dones: Terminal states of the environment
        :return: 
        """
        B, T, _ = actions.size() # They are crucial to to infernece without access to observations

        hiddens_list = []
        posterior_means_list = []
        posterior_logvars_list = []
        prior_means_list = []
        prior_logvars_list = []
        prior_states_list = []
        posterior_states_list = []
        
        # (B, 1, hidden_dim)
        hiddens_list.append(prev_hidden.unsqueeze(1))  
        prior_states_list.append(prev_state.unsqueeze(1))
        posterior_states_list.append(prev_state.unsqueeze(1))

        for t in range(T - 1):
            ### Combine the state and action ###
            action_t = actions[:, t, :]
            obs_t = obs[:, t, :] if obs is not None else torch.zeros(B, self.embedding_dim, device=actions.device)
            state_t = posterior_states_list[-1][:, 0, :] if obs is not None else prior_states_list[-1][:, 0, :]
            state_t = state_t if dones is None else state_t * (1 - dones[:, t, :])
            hidden_t = hiddens_list[-1][:, 0, :]
            
            state_action = torch.cat([state_t, action_t], dim=-1)
            state_action = self.act_fn(self.project_state_action(state_action))

            ### Update the deterministic hidden state ###
            for i in range(len(self.rnn)):
                hidden_t = self.rnn[i](state_action, hidden_t)

            ### Determine the prior distribution ###
            hidden_action = torch.cat([hidden_t, action_t], dim=-1)
            hidden_action = self.act_fn(self.project_hidden_action(hidden_action))
            prior_params = self.prior(hidden_action)
            prior_mean, prior_logvar = torch.chunk(prior_params, 2, dim=-1)

            ### Sample from the prior distribution ###
            prior_dist = torch.distributions.Normal(prior_mean, torch.exp(F.softplus(prior_logvar)))
            prior_state_t = prior_dist.rsample()

            ### Determine the posterior distribution ###
            # If observations are not available, we just use the prior
            if obs is None:
                posterior_mean = prior_mean
                posterior_logvar = prior_logvar
            else:
                hidden_obs = torch.cat([hidden_t, obs_t], dim=-1)
                hidden_obs = self.act_fn(self.project_hidden_obs(hidden_obs))
                posterior_params = self.posterior(hidden_obs)
                posterior_mean, posterior_logvar = torch.chunk(posterior_params, 2, dim=-1)

            ### Sample from the posterior distribution ###
            posterior_dist = torch.distributions.Normal(posterior_mean, torch.exp(F.softplus(posterior_logvar)))
            
            # Make sure to use rsample to enable the gradient flow
            # Otherwise you could also use code the reparameterization trick by hand
            posterior_state_t = posterior_dist.rsample()

            ### Store results in lists (instead of in-place modification) ###
            posterior_means_list.append(posterior_mean.unsqueeze(1))
            posterior_logvars_list.append(posterior_logvar.unsqueeze(1))
            prior_means_list.append(prior_mean.unsqueeze(1))
            prior_logvars_list.append(prior_logvar.unsqueeze(1))
            prior_states_list.append(prior_state_t.unsqueeze(1))
            posterior_states_list.append(posterior_state_t.unsqueeze(1))
            hiddens_list.append(hidden_t.unsqueeze(1))

        # Convert lists to tensors using torch.cat()
        hiddens = torch.cat(hiddens_list, dim=1)
        prior_states = torch.cat(prior_states_list, dim=1)
        posterior_states = torch.cat(posterior_states_list, dim=1)
        prior_means = torch.cat(prior_means_list, dim=1)
        prior_logvars = torch.cat(prior_logvars_list, dim=1)
        posterior_means = torch.cat(posterior_means_list, dim=1)
        posterior_logvars = torch.cat(posterior_logvars_list, dim=1)

        return hiddens, prior_states, posterior_states, prior_means, prior_logvars, posterior_means, posterior_logvars
    
class RSSM:
    def __init__(self,
                 encoder: EncoderCNN,
                 decoder: DecoderCNN,
                 reward_model: RewardModel,
                 dynamics_model: nn.Module,
                 hidden_dim: int,
                 state_dim: int,
                 action_dim: int,
                 embedding_dim: int,
                 device: str = "mps"):
        """
        Recurrent State-Space Model (RSSM) for learning dynamics models.

        Args:
            encoder: Encoder network for deterministic state
            prior_model: Prior network for stochastic state
            decoder: Decoder network for reconstructing observation
            sequence_model: Recurrent model for deterministic state
            hidden_dim: Hidden dimension of the RNN
            latent_dim: Latent dimension of the stochastic state
            action_dim: Dimension of the action space
            obs_dim: Dimension of the encoded observation space


        """
        super(RSSM, self).__init__()

        self.dynamics = dynamics_model
        self.encoder = encoder
        self.decoder = decoder
        self.reward_model = reward_model

        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim

        #shift to device
        self.dynamics.to(device)
        self.encoder.to(device)
        self.decoder.to(device)
        self.reward_model.to(device)


    def generate_rollout(self, actions: torch.Tensor, hiddens: torch.Tensor = None, states: torch.Tensor = None,
                         obs: torch.Tensor = None, dones: torch.Tensor = None):

        if hiddens is None:
            hiddens = torch.zeros(actions.size(0), self.hidden_dim).to(actions.device)

        if states is None:
            states = torch.zeros(actions.size(0), self.state_dim).to(actions.device)

        dynamics_result = self.dynamics(hiddens, states, actions, obs, dones)
        hiddens, prior_states, posterior_states, prior_means, prior_logvars, posterior_means, posterior_logvars = dynamics_result

        return hiddens, prior_states, posterior_states, prior_means, prior_logvars, posterior_means, posterior_logvars

    def train(self):
        self.dynamics.train()
        self.encoder.train()
        self.decoder.train()
        self.reward_model.train()

    def eval(self):
        self.dynamics.eval()
        self.encoder.eval()
        self.decoder.eval()
        self.reward_model.eval()

    def encode(self, obs: torch.Tensor):
        return self.encoder(obs)

    def decode(self, state: torch.Tensor):
        return self.decoder(state)

    def predict_reward(self, h: torch.Tensor, s: torch.Tensor):
        return self.reward_model(h, s)

    def parameters(self):
        return list(self.dynamics.parameters()) + list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.reward_model.parameters())

    def save(self, path: str):
        torch.save({
            "dynamics": self.dynamics.state_dict(),
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "reward_model": self.reward_model.state_dict()
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path)
        self.dynamics.load_state_dict(checkpoint["dynamics"])
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.decoder.load_state_dict(checkpoint["decoder"])
        self.reward_model.load_state_dict(checkpoint["reward_model"])

class Buffer:
    def __init__(self, buffer_size: int, obs_shape: tuple, action_shape: tuple, device: torch.device):
        self.buffer_size = buffer_size
        self.obs_buffer = np.zeros((buffer_size, *obs_shape), dtype=np.float32)
        self.action_buffer = np.zeros((buffer_size, *action_shape), dtype=np.int32)  # Ensure integer type
        self.reward_buffer = np.zeros((buffer_size, 1), dtype=np.float32)
        self.done_buffer = np.zeros((buffer_size, 1), dtype=np.bool_)

        self.device = device

        self.idx = 0

    def add(self, obs: torch.Tensor, action: int, reward: float, done: bool):
        self.obs_buffer[self.idx] = obs
        self.action_buffer[self.idx] = action
        self.reward_buffer[self.idx] = reward
        self.done_buffer[self.idx] = done

        self.idx = (self.idx + 1) % self.buffer_size


    def sample(self, batch_size: int, sequence_length: int):
        starting_idxs = np.random.randint(0, (self.idx % self.buffer_size) - sequence_length, (batch_size,))

        index_tensor = np.stack([np.arange(start, start + sequence_length) for start in starting_idxs])
        obs_sequence = self.obs_buffer[index_tensor]
        action_sequence = self.action_buffer[index_tensor]
        reward_sequence = self.reward_buffer[index_tensor]
        done_sequence = self.done_buffer[index_tensor]

        return obs_sequence, action_sequence, reward_sequence, done_sequence


    def save(self, path: str):
        np.savez(path, obs_buffer=self.obs_buffer, action_buffer=self.action_buffer,
                 reward_buffer=self.reward_buffer, done_buffer=self.done_buffer, idx=self.idx)

    def load(self, path: str):
        data = np.load(path)
        self.obs_buffer = data["obs_buffer"]
        self.action_buffer = data["action_buffer"]
        self.reward_buffer = data["reward_buffer"]
        self.done_buffer = data["done_buffer"]
        self.idx = data["idx"]


class Policy(ABC):
    @abstractmethod
    def __call__(self, obs):
        pass

class RandomPolicy(Policy):
    def __init__(self, env: Env):
        self.env = env

    def __call__(self, obs):
        return self.env.action_space.sample()


class Agent:
    def __init__(self, env: Env, rssm: RSSM, buffer_size: int = 100000, collection_policy: str = "random", device="mps"):
        self.env = env
        match collection_policy:
            case "random":
                self.rollout_policy = RandomPolicy(env)
            case _:
                raise ValueError("Invalid rollout policy")

        self.buffer = Buffer(buffer_size, env.observation_space.shape, env.action_space.shape, device=device)
        self.rssm = rssm

    def data_collection_action(self, obs):
        return self.rollout_policy(obs)

    def collect_data(self, num_steps: int):
        obs = self.env.reset()
        done = False

        iterator = tqdm(range(num_steps), desc="Data Collection")
        for _ in iterator:
            action = self.data_collection_action(obs)
            next_obs, reward, done, _, _ = self.env.step(action)
            self.buffer.add(next_obs, action, reward, done)
            obs = next_obs
            if done:
                obs = self.env.reset()

    def imagine_rollout(self, prev_hidden: torch.Tensor, prev_state: torch.Tensor, actions: torch.Tensor):

        hiddens, prior_states, posterior_states, prior_means, prior_logvars, posterior_means, posterior_logvars = self.rssm.generate_rollout(actions, prev_hidden, prev_state)

        # We are in the imagaination phase, so we need to use the prior as we do not have access to the true posterior
        rewards = self.rssm.predict_reward(hiddens, prior_states)

        return hiddens, prior_states, posterior_states, prior_means, prior_logvars, posterior_means, posterior_logvars, rewards

    def plan(self, num_steps: int, prev_hidden: torch.Tensor, prev_state: torch.Tensor, actions: torch.Tensor):
        hidden_states = []
        prior_states = []

        hiddens = prev_hidden
        states = prev_state

        for _ in range(num_steps):
            hiddens, states, _, _, _, _, _, _ = self.imagine_rollout(hiddens, states, actions)
            hidden_states.append(hiddens)
            prior_states.append(states)

        hidden_states = torch.stack(hidden_states)
        prior_states = torch.stack(prior_states)

        return hidden_states, prior_states
    
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Output logs to console
        logging.FileHandler("training.log", mode="w")
    ]
)

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, rssm: RSSM, agent: Agent, optimizer: torch.optim.Optimizer, device: torch.device):
        self.rssm = rssm
        self.optimizer = optimizer
        self.device = device
        self.agent = agent

        self.writer = SummaryWriter()

    def collect_data(self, num_steps: int):
        self.agent.collect_data(num_steps)

    def save_buffer(self, path: str):
        self.agent.buffer.save(path)

    def train_batch(self, batch_size: int, seq_len: int, iteration: int, save_images: bool = False):
        obs, actions, rewards, dones = self.agent.buffer.sample(batch_size, seq_len)

        actions = torch.tensor(actions).long().to(self.device)
        actions = F.one_hot(actions, self.rssm.action_dim).float()

        obs = torch.tensor(obs, requires_grad=True).float().to(self.device)
        rewards = torch.tensor(rewards, requires_grad=True).float().to(self.device)
        dones = torch.tensor(dones).float().to(self.device)

        encoded_obs = self.rssm.encoder(obs.reshape(-1, *obs.shape[2:]).permute(0, 3, 1, 2))
        encoded_obs = encoded_obs.reshape(batch_size, seq_len, -1)

        rollout = self.rssm.generate_rollout(actions, obs=encoded_obs, dones=dones)

        hiddens, prior_states, posterior_states, prior_means, prior_logvars, posterior_means, posterior_logvars = rollout

        hiddens_reshaped = hiddens.reshape(batch_size * seq_len, -1)
        posterior_states_reshaped = posterior_states.reshape(batch_size * seq_len, -1)

        decoded_obs = self.rssm.decoder(hiddens_reshaped, posterior_states_reshaped)
        decoded_obs = decoded_obs.reshape(batch_size, seq_len, *obs.shape[-3:])

        reward_params = self.rssm.reward_model(hiddens, posterior_states)
        mean, logvar = torch.chunk(reward_params, 2, dim=-1)
        logvar = F.softplus(logvar)
        reward_dist = Normal(mean, torch.exp(logvar))
        predicted_rewards = reward_dist.rsample()

        if save_images:
            batch_idx = np.random.randint(0, batch_size)
            seq_idx = np.random.randint(0, seq_len - 3)
            fig = self._visualize(obs, decoded_obs, rewards, predicted_rewards, batch_idx,
                                  seq_idx, iteration, grayscale=True)
            if not os.path.exists("reconstructions"):
                os.makedirs("reconstructions")
            fig.savefig(f"reconstructions/iteration_{iteration}.png")
            self.writer.add_figure("Reconstructions", fig, iteration)
            plt.close(fig)

        reconstruction_loss = self._reconstruction_loss(decoded_obs, obs)
        kl_loss = self._kl_loss(prior_means, F.softplus(prior_logvars), posterior_means, F.softplus(posterior_logvars))
        reward_loss = self._reward_loss(rewards, predicted_rewards)

        loss = reconstruction_loss + kl_loss + reward_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.rssm.parameters(), 1, norm_type=2)
        self.optimizer.step()

        return loss.item(), reconstruction_loss.item(), kl_loss.item(), reward_loss.item()

    def train(self, iterations: int, batch_size: int, seq_len: int):
        self.rssm.train()
        iterator = tqdm(range(iterations), desc="Training", total=iterations)
        losses = []
        infos = []
        last_loss = float("inf")
        for i in iterator:
            loss, reconstruction_loss, kl_loss, reward_loss = self.train_batch(batch_size, seq_len, i,
                                                                               save_images=i % 100 == 0)

            self.writer.add_scalar("Loss", loss, i)
            self.writer.add_scalar("Reconstruction Loss", reconstruction_loss, i)
            self.writer.add_scalar("KL Loss", kl_loss, i)
            self.writer.add_scalar("Reward Loss", reward_loss, i)

            if loss < last_loss:
                self.rssm.save("rssm.pth")
                last_loss = loss

            info = {
                "Loss": loss,
                "Reconstruction Loss": reconstruction_loss,
                "KL Loss": kl_loss,
                "Reward Loss": reward_loss
            }
            losses.append(loss)
            infos.append(info)

            if i % 10 == 0:
                logger.info("\n----------------------------")
                logger.info(f"Iteration: {i}")
                logger.info(f"Loss: {loss:.4f}")
                logger.info(f"Running average last 20 losses: {sum(losses[-20:]) / 20: .4f}")
                logger.info(f"Reconstruction Loss: {reconstruction_loss:.4f}")
                logger.info(f"KL Loss: {kl_loss:.4f}")
                logger.info(f"Reward Loss: {reward_loss:.4f}")

    def _visualize(self, obs, decoded_obs, rewards, predicted_rewwards, batch_idx, seq_idx, iterations: int, grayscale: bool = True):
        obs = obs[batch_idx, seq_idx: seq_idx + 3]
        decoded_obs = decoded_obs[batch_idx, seq_idx: seq_idx + 3]

        rewards = rewards[batch_idx, seq_idx: seq_idx + 3]
        predicted_rewards = predicted_rewwards[batch_idx, seq_idx: seq_idx + 3]

        obs = obs.cpu().detach().numpy()
        decoded_obs = decoded_obs.cpu().detach().numpy()

        fig, axs = plt.subplots(3, 2)
        axs[0][0].imshow(obs[0, ..., 0], cmap="gray" if grayscale else None)
        axs[0][0].set_title(f"Iteration: {iterations} -- Reward: {rewards[0, 0]:.4f}")
        axs[0][0].axis("off")
        axs[0][1].imshow(decoded_obs[0, ..., 0], cmap="gray" if grayscale else None)
        axs[0][1].set_title(f"Pred. Reward: {predicted_rewards[0, 0]:.4f}")

        axs[0][1].axis("off")

        axs[1][0].imshow(obs[1, ..., 0], cmap="gray" if grayscale else None)
        axs[1][0].axis("off")
        axs[1][0].set_title(f"Reward: {rewards[1, 0]:.4f} ")
        axs[1][1].imshow(decoded_obs[1, ..., 0], cmap="gray" if grayscale else None)
        axs[1][1].set_title(f"Pred. Reward: {predicted_rewards[1, 0]:.4f}")
        axs[1][1].axis("off")

        axs[2][0].imshow(obs[2, ..., 0], cmap="gray" if grayscale else None)
        axs[2][0].axis("off")
        axs[2][0].set_title(f"Reward: {rewards[2, 0]:.4f}")
        axs[2][1].imshow(decoded_obs[2, ..., 0], cmap="gray" if grayscale else None)
        axs[2][1].set_title(f"Pred. Reward: {predicted_rewards[2, 0]:.4f}")
        axs[2][1].axis("off")

        return fig

    def _reconstruction_loss(self, decoded_obs, obs):
        return F.mse_loss(decoded_obs, obs)

    def _kl_loss(self, prior_means, prior_logvars, posterior_means, posterior_logvars):
        prior_dist = Normal(prior_means, torch.exp(prior_logvars))
        posterior_dist = Normal(posterior_means, torch.exp(posterior_logvars))

        return kl_divergence(posterior_dist, prior_dist).mean()

    def _reward_loss(self, rewards, predicted_rewards):
        return F.mse_loss(predicted_rewards, rewards)
    
env = make_env("CarRacing-v2", render_mode="rgb_array", continuous=False, grayscale=True)
hidden_size = 1024
embedding_dim = 1024
state_dim = 512

encoder = EncoderCNN(in_channels=1, embedding_dim=embedding_dim)
decoder = DecoderCNN(hidden_size=hidden_size, state_size=state_dim, embedding_size=embedding_dim,
                     output_shape=(1,128,128))
reward_model = RewardModel(hidden_dim=hidden_size, state_dim=state_dim)
dynamics_model = DynamicsModel(hidden_dim=hidden_size, state_dim=state_dim, action_dim=5, embedding_dim=embedding_dim)

rssm = RSSM(dynamics_model=dynamics_model,
            encoder=encoder,
            decoder=decoder,
            reward_model=reward_model,
            hidden_dim=hidden_size,
            state_dim=state_dim,
            action_dim=5,
            embedding_dim=embedding_dim)

optimizer = torch.optim.Adam(rssm.parameters(), lr=1e-3)
agent = Agent(env, rssm)
trainer = Trainer(rssm, agent, optimizer=optimizer, device="cuda")
trainer.collect_data(20000)
trainer.save_buffer("buffer.npz")
trainer.train(10000, 32, 20)