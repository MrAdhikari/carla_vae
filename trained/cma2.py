import os
import torch
import numpy as np
import time
import cma
from torch import nn
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import io
import carla

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# --- Load pre-trained models ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load VAE
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        # Encoder: (input: [1, 300, 300])
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),    # -> [32, 150, 150]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),   # -> [64, 75, 75]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # -> [128, 38, 38]
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # -> [256, 19, 19]
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), # -> [512, 10, 10]
            nn.ReLU()
        )

        # Calculate the flattened size after convolutions
        self.flatten_size = 41472

        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, self.flatten_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # -> [256, 19, 19]
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # -> [128, 38, 38]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # -> [64, 75, 75]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # -> [32, 150, 150]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=16, stride=2, padding=1),
            nn.Sigmoid()  # ensures output in [0, 1]
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # standard deviation
        eps = torch.randn_like(std)     # sample from N(0,1)
        return mu + std * eps           # z = mu + sigma * epsilon

    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(-1, 512, 9, 9)  # Reshape to match encoder output
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

# Load MDN-RNN
class MDNRNN(nn.Module):
    def __init__(self, latent_dim=64, action_dim=3, hidden_dim=512, n_gaussians=5):
        super(MDNRNN, self).__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.n_gaussians = n_gaussians

        self.input_dim = latent_dim + action_dim
        self.lstm = nn.LSTM(self.input_dim, hidden_dim, batch_first=True)

        # Mixture weights (pi)
        self.fc_pi = nn.Linear(hidden_dim, n_gaussians)

        # Means (mu) for each Gaussian
        self.fc_mu = nn.Linear(hidden_dim, n_gaussians * latent_dim)

        # Standard deviations (sigma) for each Gaussian
        self.fc_sigma = nn.Linear(hidden_dim, n_gaussians * latent_dim)

    def forward(self, x, hidden=None):
        # x shape: [batch_size, seq_len, input_dim]
        # hidden: tuple of (h_0, c_0) each with shape [1, batch_size, hidden_dim]

        # Pass through LSTM
        out, hidden = self.lstm(x, hidden)

        # Calculate mixture weights (pi)
        pi = self.fc_pi(out)
        pi = nn.functional.softmax(pi, dim=-1)  # Ensure they sum to 1

        # Calculate means (mu) for each Gaussian
        mu = self.fc_mu(out)
        mu = mu.view(x.size(0), x.size(1), self.n_gaussians, self.latent_dim)

        # Calculate standard deviations (sigma) for each Gaussian
        sigma = self.fc_sigma(out)
        sigma = sigma.view(x.size(0), x.size(1), self.n_gaussians, self.latent_dim)
        sigma = torch.exp(sigma)  # Ensure positivity

        return pi, mu, sigma, hidden

# Load CarlaEnv from env.py
from env2 import CarlaEnv

# Define Controller Policy Network
class ControllerPolicy(nn.Module):
    def __init__(self, latent_dim=64, hidden_dim=64, action_dim=3):  # Reduced hidden_dim from 256 to 64
        super(ControllerPolicy, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)  # Removed one hidden layer
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        actions = self.fc2(x)
        
        steer = torch.tanh(actions[:, 0])
        throttle = torch.sigmoid(actions[:, 1])
        brake = torch.sigmoid(actions[:, 2])
        
        return torch.stack([steer, throttle, brake], dim=1)
    
    def get_action(self, latent_state):
        """Get action from latent state"""
        with torch.no_grad():
            action = self.forward(latent_state)
        return action.cpu().numpy().squeeze()

# --- Training utilities ---
def preprocess_image(frame):
    """Convert raw image data to format expected by VAE"""
    # Convert image to grayscale and resize to 300x300
    if frame.shape[-1] == 4:  # If image has alpha channel
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
    resized = cv2.resize(frame, (300, 300), interpolation=cv2.INTER_AREA)
   
    if len(resized.shape) == 3 and resized.shape[2] == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    else:
        gray = resized  # Already grayscale
    
    # Apply min-max normalization as in data collection
    min_val = gray.min()
    max_val = gray.max()
    if max_val > min_val:
        normalized = ((gray - min_val) / (max_val - min_val)).astype(np.float32)
    else:
        normalized = (gray / 255.0).astype(np.float32)
    
    # Convert to tensor and add batch dimension
    img_tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).float().to(device)
    return img_tensor



def sample_from_mixture(pi, mu, sigma):
    """Sample from a mixture of Gaussians"""
    # Choose a Gaussian based on mixture weights
    categorical = torch.distributions.Categorical(pi)
    pis = categorical.sample().item()
    
    # Sample from the selected Gaussian
    selected_mu = mu[0, 0, pis]
    selected_sigma = sigma[0, 0, pis]
    
    normal = torch.distributions.Normal(selected_mu, selected_sigma)
    sample = normal.sample()        
    
    return sample

def evaluate_policy(controller, vae, mdn_rnn, env2, n_episodes=5, horizon=1000, render=False):
    """Evaluate the policy in the real environment"""
    total_rewards = []
    
    for episode in range(n_episodes):
        obs, _ = env2.reset()
        done = False
        episode_reward = 0
        
        # Initialize RNN hidden state
        hidden = None
        
        # Current latent state (will be updated with the real observation)
        current_latent = torch.zeros(1, 64, device=device)
        
        step = 0
        
        while not done and step < horizon:
            # Convert observation to tensor and encode with VAE
            img_tensor = preprocess_image(obs)
            with torch.no_grad():
                mu, logvar = vae.encode(img_tensor)
                current_latent = vae.reparameterize(mu, logvar).view(1, -1)
                action = controller.get_action(current_latent)
                obs, reward, done, truncated, info = env2.step(action)
                done = done or truncated
                episode_reward += reward
                step += 1
                
                action_tensor = torch.tensor(action, dtype=torch.float32).view(1, 1, -1).to(device)
                rnn_input = torch.cat([current_latent.view(1, 1, -1), action_tensor], dim=-1)
                
                # Get new hidden state and then detach it
                pi, mu, sigma, hidden = mdn_rnn(rnn_input, hidden)
                hidden = (hidden[0].detach(), hidden[1].detach())
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode+1} reward: {episode_reward}")
    
    avg_reward = sum(total_rewards) / len(total_rewards)
    return avg_reward, total_rewards

def set_params(controller, params):
    """Set the parameters of the controller from a flat vector"""
    idx = 0
    for param in controller.parameters():
        flat_size = param.numel()
        param.data.copy_(torch.tensor(params[idx:idx+flat_size], dtype=torch.float32).view_as(param))
        idx += flat_size
    return controller

def get_params(controller):
    """Get the parameters of the controller as a flat vector"""
    return np.concatenate([p.data.cpu().numpy().flatten() for p in controller.parameters()])

def fitness_function(params, controller, vae, mdn_rnn, env2, n_episodes=3):
    """Fitness function for CMA-ES. Returns the negative average reward."""
    controller = set_params(controller, params)
    avg_reward, _ = evaluate_policy(controller, vae, mdn_rnn, env2, n_episodes=n_episodes)
    return -avg_reward  # Negative because CMA-ES minimizes

# --- Main training loop ---
def main():
    # 1. Load pre-trained VAE and MDN-RNN models
    print("Loading pre-trained models...")
    latent_dim = 64
    
    # Load VAE
    vae = VAE(latent_dim).to(device)
    vae.load_state_dict(torch.load("./vae.pth", map_location=device))
    vae.eval()
    
    # Load MDN-RNN
    mdn_rnn = MDNRNN().to(device)
    mdn_rnn.load_state_dict(torch.load("./mdn.pth", map_location=device))
    mdn_rnn.eval()
    
    # Create output directory
    os.makedirs("./policy", exist_ok=True)
    
    # 2. Initialize environment
    env2 = CarlaEnv(town='Town10HD')
    
    # 3. Initialize controller policy
    controller = ControllerPolicy().to(device)
    
    # 4. Setup CMA-ES
    initial_params = get_params(controller)
    n_params = len(initial_params)
    print(f"Number of policy parameters: {n_params}")
    
    # Initialize CMA-ES optimizer
    sigma0 = 0.1  # Initial step size
    popsize = 16  # Population size
    
    es = cma.CMAEvolutionStrategy(
    initial_params, 
    sigma0, 
    {'popsize': popsize, 'CMA_diagonal': True}
)
    
    # 5. Training loop
    n_generations = 100
    best_fitness = float('inf')
    best_params = None
    generation_rewards = []
    
    for generation in range(1, n_generations + 1):
        print(f"Generation {generation}/{n_generations}")
        
        # Generate candidate solutions
        solutions = es.ask()
        
        # Evaluate fitness in parallel
        fitness_values = []
        for i, params in enumerate(solutions):
            print(f"Evaluating solution {i+1}/{len(solutions)}")
            fitness = fitness_function(params, controller, vae, mdn_rnn, env2)
            fitness_values.append(fitness)
        
        # Update CMA-ES with fitness values
        es.tell(solutions, fitness_values)
        
        # Track best solution
        min_fitness_idx = np.argmin(fitness_values)
        min_fitness = fitness_values[min_fitness_idx]
        
        if min_fitness < best_fitness:
            best_fitness = min_fitness
            best_params = solutions[min_fitness_idx]
            
            # Set best parameters to controller and save
            controller = set_params(controller, best_params)
            torch.save(controller.state_dict(), f"./policy/best_policy.pth")
        
        # Calculate generation statistics
        avg_fitness = np.mean(fitness_values)
        avg_reward = -avg_fitness  # Convert back to reward
        generation_rewards.append(avg_reward)
        
        print(f"Generation {generation} results:")
        print(f"  Average reward: {avg_reward:.2f}")
        print(f"  Best reward: {-min_fitness:.2f}")
        print(f"  All-time best reward: {-best_fitness:.2f}")
        
        # Save every 5 generations
        if generation % 5 == 0:
            # Save controller
            torch.save(controller.state_dict(), f"./policy/policy_gen_{generation}.pth")
            
            # Save rewards plot
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(generation_rewards) + 1), generation_rewards)
            plt.xlabel('Generation')
            plt.ylabel('Average Reward')
            plt.title('Training Progress')
            plt.savefig(f"./policy/rewards_gen_{generation}.png")
            plt.close()
            
            # Save rewards data
            np.save(f"./policy/rewards_gen_{generation}.npy", np.array(generation_rewards))
            
    # Final evaluation with more episodes
    controller = set_params(controller, best_params)
    final_avg_reward, final_rewards = evaluate_policy(
        controller, vae, mdn_rnn, env2, n_episodes=10, render=True
    )
    
    print(f"Final evaluation - Average reward: {final_avg_reward:.2f}")
    print(f"Individual episode rewards: {final_rewards}")
    
    # Save final controller
    torch.save(controller.state_dict(), "./policy/final_policy.pth")
    
    # Save final rewards plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(generation_rewards) + 1), generation_rewards)
    plt.xlabel('Generation')
    plt.ylabel('Average Reward')
    plt.title('Final Training Progress')
    plt.savefig("./policy/final_rewards.png")
    plt.close()
    
    # Clean up
    env2.close()

main()