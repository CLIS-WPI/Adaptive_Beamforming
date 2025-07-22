# -*- coding: utf-8 -*-
"""
This script implements and evaluates an adaptive beamforming strategy for a
Multi-User MIMO-OFDM system using a two-stage machine learning framework,
as described in the paper "Adaptive Beamforming for Interference-Limited
MU-MIMO using Spatio-Temporal Policy Networks".

The framework consists of:
1. A CNN-GRU Encoder: Learns a spatio-temporal representation from a sequence
   of partial Channel State Information (CSI) snapshots.
2. A PPO Reinforcement Learning Agent: Uses the learned representation to select
   an optimal, quantized combining matrix from a codebook to maximize user throughput.

This implementation is optimized for performance on modern GPUs (e.g., NVIDIA H100)
by leveraging mixed-precision training and TensorFlow's graph execution mode. It
gracefully falls back to CPU if no GPU is available.

Author: Hossein Mohammadi (Original), Optimized by Gemini
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# --- Sionna Imports ---
import sionna
import tensorflow as tf
print(f"Sionna Version: {sionna.__version__}")
print(f"TensorFlow Version: {tf.__version__}")

# Sionna Imports (ensure Sionna version >= 0.19)
from sionna.phy.channel.tr38901 import CDL, PanelArray
from sionna.phy.channel import subcarrier_frequencies, cir_to_ofdm_channel, AWGN
from sionna.phy.ofdm import ResourceGrid, OFDMModulator, OFDMDemodulator, ResourceGridMapper, ResourceGridDemapper
from sionna.phy.mapping import Mapper, Demapper
from sklearn.manifold import TSNE

# ──────────────────────────────────────────────────────────────────────────────
# DEVICE AND PRECISION CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
def configure_environment():
    """
    Configures TensorFlow to use GPU with memory growth and mixed precision if available,
    otherwise falls back to CPU.
    """
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.set_visible_devices(gpus, 'GPU')
            # Enable mixed precision for performance boost on compatible GPUs (e.g., H100)
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print(f"Successfully configured to run on {len(gpus)} GPU(s) with mixed precision.")
        else:
            print("No GPU found. The script will run on the CPU.")
            tf.config.set_visible_devices([], 'GPU') # Explicitly disable GPUs
    except RuntimeError as e:
        print(f"RuntimeError during device configuration: {e}")
        print("Forcing CPU execution.")
        tf.config.set_visible_devices([], 'GPU')

configure_environment()

# Set seed for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Disable eager execution for performance; TensorFlow will use graph mode.
tf.config.run_functions_eagerly(False)


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 1: CNN-GRU ENCODER MODEL DEFINITION
# ──────────────────────────────────────────────────────────────────────────────
class CNNGRUEncoder(tf.keras.Model):
    """
    A hybrid CNN-GRU model to encode a sequence of CSI snapshots into a latent vector.
    The CNN extracts spatial features, and the GRU captures temporal dependencies.
    """
    def __init__(self, embedding_dim):
        super(CNNGRUEncoder, self).__init__()
        # Using float32 for layers that might be sensitive to precision loss
        self.conv1 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', dtype='float32')
        self.bn1 = tf.keras.layers.BatchNormalization(dtype='float32')
        self.conv2 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same', dtype='float32')
        self.bn2 = tf.keras.layers.BatchNormalization(dtype='float32')
        # unroll=True can be faster for short sequences on GPU but uses more memory.
        self.gru = tf.keras.layers.GRU(units=embedding_dim, return_sequences=False, unroll=True, dtype='float32')

    @tf.function
    def call(self, inputs):
        """Forward pass of the encoder."""
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        embedding = self.gru(x)
        return embedding

# ──────────────────────────────────────────────────────────────────────────────
# STAGE 2: REINFORCEMENT LEARNING (PPO) COMPONENTS
# ──────────────────────────────────────────────────────────────────────────────
class Actor(tf.keras.Model):
    """
    The PPO Actor network. It takes the state (latent embedding) and outputs
    a probability distribution over the discrete action space (codebook indices).
    """
    def __init__(self, num_actions):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu', dtype='float32')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu', dtype='float32')
        # Output logits are float32 for numerical stability with softmax
        self.logits = tf.keras.layers.Dense(num_actions, activation=None, dtype='float32')

    @tf.function
    def call(self, state):
        """Forward pass of the Actor network."""
        x = self.dense1(state)
        x = self.dense2(x)
        logits = self.logits(x)
        return tf.nn.softmax(logits)

class Critic(tf.keras.Model):
    """
    The PPO Critic network. It takes the state and outputs a single value,
    estimating the expected return from that state.
    """
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu', dtype='float32')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu', dtype='float32')
        self.value = tf.keras.layers.Dense(1, activation=None, dtype='float32')

    @tf.function
    def call(self, state):
        """Forward pass of the Critic network."""
        x = self.dense1(state)
        x = self.dense2(x)
        value = self.value(x)
        return value

class PPOAgent:
    """
    The PPO Agent that orchestrates the training process, including advantage
    calculation and updating the Actor and Critic networks.
    """
    def __init__(self, actor, critic, optimizer, gamma=0.995, lambda_gae=0.95, epsilon_clip=0.1, value_coeff=0.5, entropy_coeff=0.005):
        self.actor = actor
        self.critic = critic
        self.optimizer = optimizer
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.epsilon_clip = epsilon_clip
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff

    def _compute_advantages_and_returns(self, rewards, values, next_values, dones):
        """
        Computes Generalized Advantage Estimation (GAE) and returns for a trajectory.
        This part is often clearer in NumPy and the overhead is minimal for typical batch sizes.
        """
        num_steps = len(rewards)
        returns = np.zeros(num_steps, dtype=np.float32)
        advantages = np.zeros(num_steps, dtype=np.float32)
        last_gae_lam = 0

        for t in reversed(range(num_steps)):
            if dones[t]:
                delta = rewards[t] - values[t]
                last_gae_lam = 0
            else:
                delta = rewards[t] + self.gamma * next_values[t] - values[t]
            
            advantages[t] = last_gae_lam = delta + self.gamma * self.lambda_gae * (1.0 - dones[t]) * last_gae_lam
        
        returns = advantages + values
        return returns, advantages

    @tf.function
    def train(self, states, actions, old_probs, returns, advantages):
        """
        Executes a single training step for the PPO agent.
        This function is decorated with @tf.function to compile it into a high-performance graph.
        """
        with tf.GradientTape() as tape:
            # Get new probabilities and values from the networks
            new_probs_dist = self.actor(states)
            values = self.critic(states)
            
            # Get probabilities for the actions that were actually taken
            actions_one_hot = tf.one_hot(actions, self.actor.logits.units, dtype=tf.float32)
            new_action_probs = tf.reduce_sum(new_probs_dist * actions_one_hot, axis=1)
            
            # Calculate the ratio for the PPO objective
            ratio = new_action_probs / (old_probs + 1e-10)
            
            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = tf.clip_by_value(ratio, 1.0 - self.epsilon_clip, 1.0 + self.epsilon_clip) * advantages
            actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
            
            # Critic loss (mean squared error)
            critic_loss = tf.reduce_mean(tf.square(returns - tf.squeeze(values)))
            
            # Entropy bonus for exploration
            entropy = -tf.reduce_mean(tf.reduce_sum(new_probs_dist * tf.math.log(new_probs_dist + 1e-10), axis=1))
            
            # Total loss
            total_loss = actor_loss + (self.value_coeff * critic_loss) - (self.entropy_coeff * entropy)
        
        # Calculate and apply gradients
        all_vars = self.actor.trainable_variables + self.critic.trainable_variables
        gradients = tape.gradient(total_loss, all_vars)
        self.optimizer.apply_gradients(zip(gradients, all_vars))
        
        return total_loss, actor_loss, critic_loss

def create_combiner_codebook(num_matrices, num_streams, num_rx_antennas, p_ue_max, num_quant_bits):
    """
    Creates a codebook of quantized and power-normalized combiner matrices.
    """
    codebook = []
    quant_levels = 2**num_quant_bits
    quant_step = 2.0 / (quant_levels - 1)

    for _ in range(num_matrices):
        # Generate random complex matrix
        real_part = tf.random.uniform([num_streams, num_rx_antennas], -1, 1)
        imag_part = tf.random.uniform([num_streams, num_rx_antennas], -1, 1)
        
        # Quantize real and imaginary parts
        w_quant_real = tf.round((real_part + 1.0) / quant_step) * quant_step - 1.0
        w_quant_imag = tf.round((imag_part + 1.0) / quant_step) * quant_step - 1.0
        w_quant = tf.complex(w_quant_real, w_quant_imag)
        
        # Normalize to satisfy power constraint
        norm = tf.norm(w_quant, ord='fro', axis=(-2, -1), keepdims=True)
        scale_factor = tf.cast(tf.sqrt(p_ue_max), dtype=tf.complex64)
        w_normalized = (w_quant / (norm + 1e-10)) * scale_factor
        
        codebook.append(w_normalized)
        
    return tf.stack(codebook)

# ──────────────────────────────────────────────────────────────────────────────
# MIMO RL ENVIRONMENT DEFINITION
# ──────────────────────────────────────────────────────────────────────────────
class MIMOEnvironment:
    """
    Simulates the MU-MIMO environment for the RL agent.
    It generates channel realizations, calculates rewards (SINR), and provides states.
    """
    def __init__(self, channels, encoder, V_k_list, s_k_list, snr_db, params):
        self.channels = channels
        self.encoder = encoder
        self.V_k_list = V_k_list
        self.s_k_list = s_k_list
        self.snr_db = snr_db
        self.params = params
        self.k_idx = 0  # We focus on the first user for reward calculation and state
        
        snr_linear = 10**(self.snr_db / 10.0)
        self.noise_power = tf.cast(1.0 / snr_linear, dtype=tf.float32)
        self.H_history = {k: [] for k in range(self.params['K'])}

    def reset(self):
        """Resets the environment and returns the initial state."""
        self.H_history = {k: [] for k in range(self.params['K'])}
        state, H_k = self.get_state()
        # Ensure the history buffer is full before starting
        while state is None:
            state, H_k = self.get_state()
        return state, H_k

    def get_state(self):
        """
        Generates a new channel realization, updates the history, and returns the
        encoded state for the RL agent.
        """
        # Generate a new channel realization for the user of interest (k=0)
        h, path_delays = self.channels[self.k_idx](batch_size=1, num_time_steps=1, sampling_frequency=self.params['sampling_frequency'])
        frequencies = subcarrier_frequencies(self.params['fft_size'], self.params['subcarrier_spacing'])
        H_freq = cir_to_ofdm_channel(frequencies, h, path_delays)
        
        # Use a single subcarrier for the state representation (as in the paper)
        H_k = tf.squeeze(H_freq[..., 0, :, :, 10]) # Squeeze batch and time dims
        
        # Prepare the channel matrix for the encoder input
        H_real_imag = tf.concat([tf.math.real(H_k), tf.math.imag(H_k)], axis=1)
        H_flat = tf.reshape(H_real_imag, [-1])
        
        # Update the history buffer
        self.H_history[self.k_idx].append(H_flat.numpy())
        if len(self.H_history[self.k_idx]) > self.params['tau']:
            self.H_history[self.k_idx].pop(0)
        
        # If buffer is not full, return None
        if len(self.H_history[self.k_idx]) < self.params['tau']:
            return None, None
        
        # Create the input tensor for the encoder
        X_k = np.stack(self.H_history[self.k_idx], axis=0)
        X_k_tensor = tf.convert_to_tensor(X_k, dtype=tf.float32)
        X_k_batch = tf.expand_dims(X_k_tensor, axis=0)
        
        # Get the latent state from the encoder
        phi_k = self.encoder(X_k_batch)
        return phi_k, H_k

    def step(self, action_index, H_k, codebook):
        """
        Executes one time step in the environment.
        Calculates the SINR and reward based on the chosen action.
        """
        W_k = codebook[action_index]
        
        # Construct the transmitted signal from all users
        x_t = tf.zeros((self.params['Nt'], 1), dtype=tf.complex64)
        for i in range(self.params['K']):
            x_t += tf.matmul(self.V_k_list[i], self.s_k_list[i])
            
        # Generate noise
        noise_stddev = tf.sqrt(self.noise_power / 2.0)
        noise = tf.complex(
            tf.random.normal([self.params['Nr'], 1], stddev=noise_stddev),
            tf.random.normal([self.params['Nr'], 1], stddev=noise_stddev)
        )
        
        # Received signal at user k
        y_k = tf.matmul(H_k, x_t) + noise
        
        # Calculate SINR for user k (assuming first stream for reward)
        w_i = W_k[0:1, :] # First stream's combining vector
        w_i_hermitian = tf.transpose(w_i, conjugate=True)
        
        # Desired signal power
        signal_term = tf.matmul(tf.matmul(w_i, H_k), self.V_k_list[self.k_idx][:, 0:1])
        signal_power = tf.square(tf.abs(signal_term))
        
        # Inter-user interference power
        inter_user_interference = 0.0
        for l in range(self.params['K']):
            if l != self.k_idx:
                interference_term = tf.matmul(tf.matmul(w_i, H_k), self.V_k_list[l])
                inter_user_interference += tf.reduce_sum(tf.square(tf.abs(interference_term)))
        
        # Intra-user interference power (from other streams of the same user)
        intra_user_interference = 0.0
        if self.params['Ns'] > 1:
            interference_term = tf.matmul(tf.matmul(w_i, H_k), self.V_k_list[self.k_idx][:, 1:])
            intra_user_interference = tf.reduce_sum(tf.square(tf.abs(interference_term)))
            
        # Noise power at the output of the combiner
        w_i_squared_norm = tf.reduce_sum(tf.square(tf.abs(w_i)))
        noise_power_at_output = self.noise_power * w_i_squared_norm
        
        # Calculate SINR and reward (spectral efficiency)
        sinr = signal_power / (inter_user_interference + intra_user_interference + noise_power_at_output + 1e-12)
        reward = tf.math.log(1.0 + tf.cast(sinr, tf.float32)) / tf.math.log(2.0)
        
        # Get the next state
        next_phi_k, H_k_next = self.get_state()
        done = (next_phi_k is None)
        
        return next_phi_k, reward, done, H_k_next

# ──────────────────────────────────────────────────────────────────────────────
# SIMULATION PARAMETERS & INSTANTIATION
# ──────────────────────────────────────────────────────────────────────────────
# System Parameters
Nt, Nr, K, Ns, tau = 8, 2, 4, 2, 8
fft_size, num_ofdm_symbols, subcarrier_spacing, cp_length = 256, 14, 30e3, 16
mod_order, carrier_freq, delay_spread = 16, 3.5e9, 300e-9
bits_per_symbol = int(np.log2(mod_order))
sampling_frequency = fft_size * subcarrier_spacing

# RL and Codebook Parameters
P_UE_MAX, NUM_QUANT_BITS, CODEBOOK_SIZE, embedding_dim = 1.0, 4, 256, 128

# Model and Component Instantiation
encoder = CNNGRUEncoder(embedding_dim=embedding_dim)
actor = Actor(num_actions=CODEBOOK_SIZE)
critic = Critic()
combiner_codebook = create_combiner_codebook(CODEBOOK_SIZE, Ns, Nr, P_UE_MAX, NUM_QUANT_BITS)

bs_array = PanelArray(num_rows_per_panel=1, num_cols_per_panel=Nt, polarization="single", polarization_type="V", antenna_pattern="38.901", carrier_frequency=carrier_freq)
ue_array = PanelArray(num_rows_per_panel=1, num_cols_per_panel=Nr, polarization="single", polarization_type="V", antenna_pattern="omni", carrier_frequency=carrier_freq)

# Instantiate channel models according to the newer Sionna API
channels = [
    CDL(model="C", 
        delay_spread=delay_spread, 
        carrier_frequency=carrier_freq, 
        ut_array=ue_array,
        bs_array=bs_array,
        direction="downlink", 
        min_speed=3.0)
    for _ in range(K)
]

# Mapper and Demapper for BER calculation
mapper = Mapper("qam", mod_order)
demapper = Demapper("app", "qam", bits_per_symbol, hard_out=True)

# OFDM Modulator and Demodulator
modulator = OFDMModulator(cp_length)
demodulator = OFDMDemodulator(fft_size, 0, cp_length)

# ──────────────────────────────────────────────────────────────────────────────
# MAIN PPO TRAINING LOOP
# ──────────────────────────────────────────────────────────────────────────────
print("Starting PPO Agent Training...")
num_episodes, max_steps_per_episode, snr_db_train = 10, 100, 15.0

# Learning rate schedule for optimizer
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=3e-4, decay_steps=1000, decay_rate=0.95, staircase=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)
ppo_agent = PPOAgent(actor, critic, optimizer)

params = {
    'K': K, 'Nt': Nt, 'Nr': Nr, 'Ns': Ns, 'tau': tau, 
    'sampling_frequency': sampling_frequency, 'fft_size': fft_size, 
    'subcarrier_spacing': subcarrier_spacing, 'num_ofdm_symbols': num_ofdm_symbols,
    'bits_per_symbol': bits_per_symbol
}

# --- Fixed Precoding & Symbols for the Environment ---
# Generate fixed symbols for all users
fixed_s_k = [tf.complex(tf.random.uniform([Ns, 1]), tf.random.uniform([Ns, 1])) for _ in range(K)]

# Generate a single channel snapshot to compute the fixed RBD precoder
print("--- Calculating Fixed RBD Precoders ---")
temp_h, temp_delays = zip(*[channels[k](batch_size=1, num_time_steps=1, sampling_frequency=sampling_frequency) for k in range(K)])
temp_freqs = subcarrier_frequencies(fft_size, subcarrier_spacing)
temp_H_freq = [cir_to_ofdm_channel(temp_freqs, h, d) for h, d in zip(temp_h, temp_delays)]
temp_H_k = [tf.squeeze(H[..., 0, :, :, 10]) for H in temp_H_freq]

# Calculate RBD precoder based on this snapshot
fixed_V_k = []
for k in range(K):
    # Aggregate interference channels for user k
    H_interf_list = [temp_H_k[i] for i in range(K) if i != k]
    H_interf = tf.concat(H_interf_list, axis=0)
    
    # SVD of interference channel to find its nullspace.
    s_interf, _, v_interf = tf.linalg.svd(H_interf, full_matrices=True)
    
    # Calculate rank from singular values to avoid dtype error with
    # tf.linalg.matrix_rank and to ensure numerical stability.
    tolerance = tf.reduce_max(s_interf) * 1e-6
    rank_interf = tf.reduce_sum(tf.cast(s_interf > tolerance, tf.int32))
    
    # The nullspace is formed by the right singular vectors corresponding to
    # the smallest singular values (i.e., those beyond the rank).
    T_k = v_interf[:, rank_interf:]
    
    # Project the desired user's channel into the nullspace
    H_eff_k = tf.matmul(temp_H_k[k], T_k)
    
    # SVD of the effective channel to get the final precoder
    _, _, v_eff = tf.linalg.svd(H_eff_k)
    V_eff_k = v_eff[:, :Ns]
    V_k = tf.matmul(T_k, V_eff_k)
    fixed_V_k.append(V_k)
print("--- Finished Calculating Precoders ---")

env = MIMOEnvironment(channels, encoder, fixed_V_k, fixed_s_k, snr_db_train, params)
total_rewards_history = []
entropy_history = []

for episode in range(num_episodes):
    states, actions, rewards, next_states, dones, old_probs = [], [], [], [], [], []
    state, H_k = env.reset()
    
    for t in range(max_steps_per_episode):
        action_probs = actor(state)
        # Use categorical sampling to select an action based on the policy
        action = tf.random.categorical(tf.math.log(action_probs), 1)[0, 0].numpy()
        old_prob = action_probs[0, action].numpy()
        
        next_state, reward, done, H_k_next = env.step(action, H_k, combiner_codebook)
        
        if next_state is not None:
            states.append(tf.squeeze(state).numpy())
            actions.append(action)
            rewards.append(tf.squeeze(reward).numpy())
            next_states.append(tf.squeeze(next_state).numpy())
            dones.append(done)
            old_probs.append(old_prob)
            
        if done:
            break
            
        state, H_k = next_state, H_k_next

    if len(states) > 1:
        # PPO requires a full trajectory to compute advantages and returns
        values = critic(np.array(states)).numpy().flatten()
        next_values = critic(np.array(next_states)).numpy().flatten()
        
        returns, advantages = ppo_agent._compute_advantages_and_returns(np.array(rewards), values, next_values, np.array(dones))
        # Normalize advantages for stable training
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        # Train the agent
        ppo_agent.train(
            np.array(states, dtype=np.float32), 
            np.array(actions, dtype=np.int32), 
            np.array(old_probs, dtype=np.float32), 
            returns, 
            advantages
        )
        
        # Log entropy for monitoring exploration
        current_probs = actor(np.array(states, dtype=np.float32))
        current_entropy = -tf.reduce_mean(tf.reduce_sum(current_probs * tf.math.log(current_probs + 1e-10), axis=1))
        entropy_history.append(current_entropy.numpy())

    episode_reward = sum(rewards)
    total_rewards_history.append(episode_reward)
    
    if (episode + 1) % 10 == 0:
        avg_reward = np.mean(total_rewards_history[-10:])
        print(f"Episode: {episode+1}/{num_episodes}, Total Reward: {episode_reward:.2f}, Avg Reward (last 10): {avg_reward:.2f}")

print("\nTraining finished.")
# Save model weights
actor.save_weights("ppo_actor.weights.h5")
critic.save_weights("ppo_critic.weights.h5")
encoder.save_weights("encoder.weights.h5")
print("Trained model weights saved successfully.")

# ──────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS FOR BASELINE ALGORITHMS AND BER
# ──────────────────────────────────────────────────────────────────────────────
def calculate_sinr(W_k, H_k, V_k_list, k_idx, noise_power, params):
    """Calculates the SINR for a given combiner, channel, and precoders."""
    Ns, K = params['Ns'], params['K']
    w_i = W_k[0:1, :] # First stream
    w_i_hermitian = tf.transpose(w_i, conjugate=True)
    
    signal_power = tf.square(tf.abs(tf.matmul(tf.matmul(w_i, H_k), V_k_list[k_idx][:,0:1])))
    
    inter_user_interference = 0.0
    for l in range(K):
        if l != k_idx:
            interference_term = tf.matmul(tf.matmul(w_i, H_k), V_k_list[l])
            inter_user_interference += tf.reduce_sum(tf.square(tf.abs(interference_term)))
            
    intra_user_interference = 0.0
    if Ns > 1:
        interference_term = tf.matmul(tf.matmul(w_i, H_k), V_k_list[k_idx][:,1:])
        intra_user_interference = tf.reduce_sum(tf.square(tf.abs(interference_term)))
        
    w_i_squared_norm = tf.reduce_sum(tf.square(tf.abs(w_i)))
    noise_power_at_output = noise_power * w_i_squared_norm
    
    sinr = signal_power / (inter_user_interference + intra_user_interference + noise_power_at_output + 1e-12)
    return sinr

def mrc_combiner(H_k, V_k, params):
    """Calculates the Maximal Ratio Combiner (MRC)."""
    H_eff = tf.matmul(H_k, V_k)
    return tf.transpose(H_eff, conjugate=True)

def mmse_combiner(H_k, V_k_list, noise_power, params):
    """Calculates the Minimum Mean Square Error (MMSE) combiner."""
    K, Nt, Nr, k_idx = params['K'], params['Nt'], params['Nr'], 0
    
    # Covariance of transmitted signal
    transmit_covariance = tf.zeros([Nt, Nt], dtype=tf.complex64)
    for i in range(K):
        transmit_covariance += tf.matmul(V_k_list[i], V_k_list[i], transpose_b=True)
        
    # Covariance of received signal y
    R_yy = tf.matmul(H_k, tf.matmul(transmit_covariance, H_k, transpose_b=True))
    noise_cov = tf.cast(noise_power, dtype=tf.complex64) * tf.eye(Nr, dtype=tf.complex64)
    R_yy += noise_cov
    
    # Cross-covariance between desired signal s and received signal y
    R_sy = tf.matmul(V_k_list[k_idx], H_k, transpose_a=True, transpose_b=True)
    R_ys = tf.transpose(R_sy, conjugate=True)
    
    # MMSE solution: W_hermitian = inv(R_yy) * R_ys
    W_mmse_hermitian = tf.linalg.lstsq(R_yy, R_ys)
    return tf.transpose(W_mmse_hermitian, conjugate=True)

def run_siso_ofdm_ber(noise_variance, params):
    """Runs a simple SISO OFDM simulation over an AWGN channel for a baseline BER."""
    global mapper, demapper, modulator, demodulator
    batch_size = 64
    
    rg_siso = ResourceGrid(
        num_ofdm_symbols=params['num_ofdm_symbols'],
        fft_size=params['fft_size'],
        subcarrier_spacing=params['subcarrier_spacing'],
        num_tx=1,
        num_streams_per_tx=1
    )
    siso_mapper = ResourceGridMapper(rg_siso)
    
    num_bits = rg_siso.num_data_symbols * params['bits_per_symbol']
    bits = tf.random.uniform(shape=[batch_size, num_bits], maxval=2, dtype=tf.int32)
    
    symbols = mapper(bits)
    x_rg = siso_mapper(tf.reshape(symbols, [batch_size, 1, 1, -1]))
    
    x_time = modulator(x_rg)
    y_time = AWGN()([x_time, noise_variance])
    # CORRECTED: The demodulator in recent Sionna versions only takes the received signal.
    x_demod = demodulator(y_time)
    
    # CORRECTED: get_symbols is now an instance method of ResourceGrid.
    symbols_hat, _ = rg_siso.get_symbols(x_demod)
    bits_hat = demapper(symbols_hat)
    
    num_errors = tf.reduce_sum(tf.cast(bits != bits_hat, tf.float32))
    return num_errors, tf.cast(tf.size(bits), tf.float32)

def run_mu_mimo_ber(combiner, H_k_freq, V_k_list, k_idx, noise_variance, params):
    """Runs a full MU-MIMO OFDM link simulation to calculate BER."""
    global mapper, demapper
    batch_size = 16
    K, Ns, Nt = params['K'], params['Ns'], params['Nt']
    
    rg = ResourceGrid(
        num_ofdm_symbols=params['num_ofdm_symbols'],
        fft_size=params['fft_size'],
        subcarrier_spacing=params['subcarrier_spacing'],
        num_tx=K,
        num_streams_per_tx=Ns
    )
    rg_mapper = ResourceGridMapper(rg)

    num_bits_per_user = rg.num_data_symbols * params['bits_per_symbol']
    bits = tf.random.uniform([batch_size, K, num_bits_per_user], minval=0, maxval=2, dtype=tf.int32)
    symbols = mapper(bits)
    symbols_reshaped = tf.reshape(symbols, [batch_size, K, Ns, -1])
    x_mapped = rg_mapper(symbols_reshaped)

    V_precoder = tf.concat(V_k_list, axis=1)
    V_reshaped = tf.reshape(V_precoder, [Nt, K, Ns])
    x_tx_freq = tf.einsum('bknsf,tkn->btsf', x_mapped, V_reshaped)

    H_k_batch = tf.expand_dims(H_k_freq, axis=0)
    y_freq = tf.einsum('brtf,btsf->brsf', H_k_batch, x_tx_freq)
    
    # Changed noise generation to use separate calls instead of tuple
    noise_real = tf.random.normal(tf.shape(y_freq), stddev=tf.sqrt(noise_variance/2))
    noise_imag = tf.random.normal(tf.shape(y_freq), stddev=tf.sqrt(noise_variance/2))
    noise = tf.complex(noise_real, noise_imag)
    y_freq_noisy = y_freq + noise

    s_hat_freq = tf.einsum('sn,brsf->bsf', combiner, y_freq_noisy)
    
    rg_rx = ResourceGrid(
        num_ofdm_symbols=params['num_ofdm_symbols'],
        fft_size=params['fft_size'],
        subcarrier_spacing=params['subcarrier_spacing'],
        num_tx=Ns,
        num_streams_per_tx=1
    )
    
    # Use ResourceGridDemapper instead of get_symbols method
    rg_demapper = ResourceGridDemapper(rg_rx)
    s_hat = rg_demapper(s_hat_freq)
    bits_hat = demapper(s_hat)
    
    original_bits_k = tf.reshape(bits[:, k_idx, :], [batch_size, -1])
    num_errors = tf.reduce_sum(tf.cast(original_bits_k != bits_hat, tf.float32))
    
    return num_errors, tf.cast(tf.size(original_bits_k), tf.float32)

# ──────────────────────────────────────────────────────────────────────────────
# EVALUATION LOOP OVER ALL SNRs
# ──────────────────────────────────────────────────────────────────────────────
print("\nStarting final evaluation phase...")
results_throughput = {"PPO": [], "RBD": [], "MRC": [], "MMSE": []}
results_ber = {"PPO": [], "RBD": [], "MRC": [], "MMSE": [], "OFDM_AWGN": []}
num_eval_steps = 100

# Load the trained models
encoder.load_weights("encoder.weights.h5")
actor.load_weights("ppo_actor.weights.h5")
print("Loaded trained model weights for evaluation.")

snr_dBs_eval = np.arange(-20, 22, 2)

for snr_db in snr_dBs_eval:
    print(f"Evaluating SNR = {snr_db} dB...")
    
    noise_power_eval = tf.cast(10**(-snr_db / 10.0), dtype=tf.float32)
    noise_variance_eval = 10**(-snr_db / 10.0)
    
    avg_throughputs = {name: [] for name in results_throughput.keys()}
    total_errors = {name: 0.0 for name in results_ber.keys()}
    total_bits = {name: 0.0 for name in results_ber.keys()}
    
    eval_env = MIMOEnvironment(channels, encoder, fixed_V_k, fixed_s_k, snr_db, params)

    for step in range(num_eval_steps):
        # CORRECTED: The channel call now returns 2 values (h, delays).
        h, path_delays = channels[0](batch_size=1, num_time_steps=1, sampling_frequency=sampling_frequency)
        frequencies = subcarrier_frequencies(fft_size, subcarrier_spacing)
        H_k_freq_domain = cir_to_ofdm_channel(frequencies, h, path_delays)
        H_k_single_sc = tf.squeeze(H_k_freq_domain[..., 0, :, :, 10])

        state, _ = eval_env.get_state()
        while state is None: state, _ = eval_env.get_state()
        
        action_probs = actor(state)
        best_action = tf.argmax(action_probs, axis=1)[0].numpy()
        W_ppo = combiner_codebook[best_action]
        
        # Calculate baseline combiners
        W_rbd = tf.transpose(tf.linalg.lstsq(tf.matmul(H_k_single_sc, fixed_V_k[0]), tf.eye(Ns, dtype=tf.complex64)), conjugate=True)
        W_mrc = mrc_combiner(H_k_single_sc, fixed_V_k[0], params)
        W_mmse = mmse_combiner(H_k_single_sc, fixed_V_k, noise_power_eval, params)

        combiners = {"PPO": W_ppo, "RBD": W_rbd, "MRC": W_mrc, "MMSE": W_mmse}

        for name, W in combiners.items():
            sinr = calculate_sinr(W, H_k_single_sc, fixed_V_k, 0, noise_power_eval, params)
            avg_throughputs[name].append(tf.math.log(1.0 + tf.cast(sinr, tf.float32)) / tf.math.log(2.0))
            
            # Run BER simulation for this combiner
            errors, bits_count = run_mu_mimo_ber(W, tf.squeeze(H_k_freq_domain, axis=1), fixed_V_k, 0, noise_variance_eval, params)
            total_errors[name] += errors
            total_bits[name] += bits_count

    # Run SISO AWGN BER for reference
    errors, bits_count = run_siso_ofdm_ber(noise_variance_eval, params)
    total_errors["OFDM_AWGN"] += errors
    total_bits["OFDM_AWGN"] += bits_count

    # Aggregate results for this SNR point
    for name in results_throughput.keys():
        results_throughput[name].append(np.mean([t.numpy() for t in avg_throughputs[name] if not tf.math.is_nan(t)]))
    
    for name in results_ber.keys():
        if total_bits[name] > 0:
            results_ber[name].append(total_errors[name] / total_bits[name])
        else:
            results_ber[name].append(1.0) # Avoid division by zero

# ──────────────────────────────────────────────────────────────────────────────
# PLOTTING RESULTS
# ──────────────────────────────────────────────────────────────────────────────
output_dir = "results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Directory '{output_dir}' created.")

# --- Plot 1: Throughput vs. SNR ---
fig1, ax1 = plt.subplots(figsize=(10, 7))
ax1.plot(snr_dBs_eval, results_throughput["PPO"], 'o-', label='PPO Agent', linewidth=2)
ax1.plot(snr_dBs_eval, results_throughput["RBD"], 's--', label='RBD', linewidth=2)
ax1.plot(snr_dBs_eval, results_throughput["MMSE"], 'd-.', label='MMSE', linewidth=2)
ax1.plot(snr_dBs_eval, results_throughput["MRC"], '^:', label='MRC', linewidth=2)
ax1.set_xlabel("SNR (dB)", fontsize=14)
ax1.set_ylabel("Average Spectral Efficiency (bits/s/Hz)", fontsize=14)
ax1.set_title("Performance Comparison of Combining Strategies", fontsize=16)
ax1.grid(True, which="both", linestyle='--')
ax1.legend(fontsize=12)
ax1.set_ylim(bottom=0)
fig1.savefig(os.path.join(output_dir, 'throughput_vs_snr.png'))
plt.close(fig1)
print("Saved throughput plot to results/throughput_vs_snr.png")

# --- Plot 2: BER vs. SNR ---
fig2, ax2 = plt.subplots(figsize=(10, 7))
ax2.plot(snr_dBs_eval, results_ber["PPO"], 'o-', label='PPO Agent', linewidth=2)
ax2.plot(snr_dBs_eval, results_ber["RBD"], 's--', label='RBD', linewidth=2)
ax2.plot(snr_dBs_eval, results_ber["MMSE"], 'd-.', label='MMSE', linewidth=2)
ax2.plot(snr_dBs_eval, results_ber["MRC"], '^:', label='MRC', linewidth=2)
ax2.plot(snr_dBs_eval, results_ber["OFDM_AWGN"], 'x-k', label='OFDM (SISO AWGN)', linewidth=2)
ax2.set_yscale('log')
ax2.set_xlabel("SNR (dB)", fontsize=14)
ax2.set_ylabel("Bit Error Rate (BER)", fontsize=14)
ax2.set_title("BER Performance Comparison of Combining Strategies", fontsize=16)
ax2.grid(True, which="both", linestyle='--')
ax2.legend(fontsize=12)
ax2.set_ylim(1e-5, 1.0)
fig2.savefig(os.path.join(output_dir, 'ber_vs_snr.png'))
plt.close(fig2)
print("Saved BER plot to results/ber_vs_snr.png")

# --- Plot 3: RL Agent Training Curve ---
fig3, ax3 = plt.subplots(figsize=(10, 7))
def moving_average(data, window_size=10):
    if len(data) < window_size: return np.array([])
    return np.convolve(data, np.ones(window_size), 'valid') / window_size
smoothed_rewards = moving_average(np.array(total_rewards_history))
if smoothed_rewards.size > 0:
    ax3.plot(smoothed_rewards)
    ax3.set_xlabel("Episode", fontsize=14)
    ax3.set_ylabel("Smoothed Average Reward", fontsize=14)
    ax3.set_title("PPO Agent Learning Curve", fontsize=16)
    ax3.grid(True)
    fig3.savefig(os.path.join(output_dir, 'rl_training_curve.png'))
    print("Saved RL training curve plot to results/rl_training_curve.png")
plt.close(fig3)

# --- Plot 4: Policy Entropy vs. Episode ---
fig4, ax4 = plt.subplots(figsize=(10, 7))
ax4.plot(entropy_history)
ax4.set_xlabel("Episode", fontsize=14)
ax4.set_ylabel("Average Policy Entropy", fontsize=14)
ax4.set_title("Policy Entropy During Training", fontsize=16)
ax4.grid(True)
fig4.savefig(os.path.join(output_dir, 'policy_entropy.png'))
plt.close(fig4)
print("Saved policy entropy plot to results/policy_entropy.png")

# --- Plot 5: t-SNE Visualization of Latent Space ---
print("\nGenerating Latent Space Visualization...")
latent_states = []
chosen_actions = []
num_viz_steps = 500
snr_db_viz = 10.0
viz_env = MIMOEnvironment(channels, encoder, fixed_V_k, fixed_s_k, snr_db_viz, params)
state, H_k = viz_env.reset()
for _ in range(num_viz_steps):
    while state is None:
        state, H_k = viz_env.get_state()
    latent_states.append(tf.squeeze(state).numpy())
    action_probs = actor(state)
    best_action = tf.argmax(action_probs, axis=1)[0].numpy()
    chosen_actions.append(best_action)
    state, _, _, H_k = viz_env.step(best_action, H_k, combiner_codebook)

tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=SEED)
latent_2d = tsne.fit_transform(np.array(latent_states))

fig5, ax5 = plt.subplots(figsize=(12, 10))
scatter = ax5.scatter(latent_2d[:, 0], latent_2d[:, 1], c=chosen_actions, cmap='viridis', alpha=0.7)
ax5.set_xlabel("t-SNE Component 1", fontsize=14)
ax5.set_ylabel("t-SNE Component 2", fontsize=14)
ax5.set_title("t-SNE Visualization of Encoder's Latent Space", fontsize=16)
ax5.grid(True, linestyle='--')
legend1 = ax5.legend(*scatter.legend_elements(num=8), title="Chosen Actions") # Show a subset of actions for clarity
ax5.add_artist(legend1)
fig5.savefig(os.path.join(output_dir, 'tsne_latent_space.png'))
plt.close(fig5)
print("Saved t-SNE visualization to results/tsne_latent_space.png")

print("\nAll tasks completed.")
