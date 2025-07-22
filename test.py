# -*- coding: utf-8 -*-
"""
@author: Hossein
"""
"""
Multi-user MIMO-OFDM setup (Sionna ≥ 0.19)
•   K  = number of UEs
•   Nt = BS TX antennas
•   Nr = UE RX antennas
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sionna
import tensorflow as tf
print(f"Sionna Version: {sionna.__version__}")
print(f"TensorFlow Version: {tf.__version__}")
import copy


# Corrected imports for Sionna v1.1.0
from sionna.phy.channel.tr38901 import CDL, PanelArray
from sionna.phy.channel import subcarrier_frequencies, cir_to_ofdm_channel, AWGN
from sionna.phy.ofdm import ResourceGrid, OFDMModulator, OFDMDemodulator, ResourceGridMapper
from sionna.phy.mapping import Mapper, Demapper

from sklearn.manifold import TSNE
# Add a print statement to confirm success


import os

# ──────────────────────────────────────────────────────────────────────────────
# DEVICE CONFIGURATION (CHOOSE GPU OR CPU)
# ──────────────────────────────────────────────────────────────────────────────
# Set this flag to True to use the GPU, or False to force CPU execution.
USE_GPU = True

if USE_GPU:
    # Use the GPU if available
    print("Attempting to run on GPU...")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Set TensorFlow to use all available GPUs and enable memory growth
            tf.config.set_visible_devices(gpus, 'GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Successfully configured to run on {len(gpus)} GPU(s).")
        except RuntimeError as e:
            # This can happen if the GPUs are already initialized
            print(e)
    else:
        print("No GPU found. The script will run on the CPU.")
else:
    # Force TensorFlow to use the CPU only
    # This is done by hiding all GPUs from TensorFlow's perspective
    try:
        tf.config.set_visible_devices([], 'GPU')
        print("USE_GPU is set to False. Forcing CPU execution.")
    except RuntimeError as e:
         # Visible devices must be set before GPUs are initialized
        print(e)

# You can also use this powerful alternative to force CPU execution:
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
# Eager execution is useful for debugging
tf.config.run_functions_eagerly(True)

# ──────────────────────────────────────────────────────────────────────────────
# STAGE 1: CNN-GRU ENCODER MODEL DEFINITION
# ──────────────────────────────────────────────────────────────────────────────
class CNNGRUEncoder(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(CNNGRUEncoder, self).__init__()
        self.conv1 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        # این خط اصلاح شد
        self.gru = tf.keras.layers.GRU(units=embedding_dim, return_sequences=False, unroll=True)

    def call(self, inputs):
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
    def __init__(self, num_actions):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.logits = tf.keras.layers.Dense(num_actions, activation=None)

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        logits = self.logits(x)
        return tf.nn.softmax(logits)

class Critic(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.value = tf.keras.layers.Dense(1, activation=None)

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        value = self.value(x)
        return value

class PPOAgent:
    def __init__(self, actor, critic, optimizer, gamma=0.995, lambda_gae=0.95, epsilon_clip=0.05, value_coeff=0.5, entropy_coeff=0.005):
        self.actor = actor
        self.critic = critic
        self.optimizer = optimizer
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.epsilon_clip = epsilon_clip
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff

    def _compute_advantages_and_returns(self, rewards, values, next_values, dones):
        returns = np.zeros_like(rewards, dtype=np.float32)
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae_lam = 0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                last_gae_lam = 0
            else:
                delta = rewards[t] + self.gamma * next_values[t] - values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.lambda_gae * (1.0 - dones[t]) * last_gae_lam
            returns[t] = advantages[t] + values[t]
        return returns, advantages

    @tf.function
    def train(self, states, actions, old_probs, returns, advantages):
        with tf.GradientTape() as tape:
            new_probs = self.actor(states)
            values = self.critic(states)
            
            actions_one_hot = tf.one_hot(actions, self.actor.logits.units, dtype=tf.float32)
            
            new_action_probs = tf.reduce_sum(new_probs * actions_one_hot, axis=1)
            
            # --- PPO Surrogate Objective (Equation 14) ---
            ratio = new_action_probs / old_probs
            surr1 = ratio * advantages
            surr2 = tf.clip_by_value(ratio, 1.0 - self.epsilon_clip, 1.0 + self.epsilon_clip) * advantages
            
            # Actor (Policy) Loss
            actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
            
            # Critic (Value) Loss
            critic_loss = tf.reduce_mean(tf.square(returns - tf.squeeze(values)))
            
            # Entropy Loss
            entropy = -tf.reduce_mean(tf.reduce_sum(new_probs * tf.math.log(new_probs + 1e-10), axis=1))
            
            total_loss = actor_loss + (self.value_coeff * critic_loss) - (self.entropy_coeff * entropy)
        all_vars = self.actor.trainable_variables + self.critic.trainable_variables
        gradients = tape.gradient(total_loss, all_vars)
        self.optimizer.apply_gradients(zip(gradients, all_vars))
        return total_loss, actor_loss, critic_loss

def create_combiner_codebook(num_matrices, num_streams, num_rx_antennas, p_ue_max, num_quant_bits):
    codebook = []
    quant_levels = 2**num_quant_bits
    quant_step = 2.0 / (quant_levels - 1)
    for _ in range(num_matrices):
        real_part = tf.random.uniform([num_streams, num_rx_antennas], -1, 1)
        imag_part = tf.random.uniform([num_streams, num_rx_antennas], -1, 1)
        w_quant_real = tf.round((real_part + 1.0) / quant_step) * quant_step - 1.0
        w_quant_imag = tf.round((imag_part + 1.0) / quant_step) * quant_step - 1.0
        w_quant = tf.complex(w_quant_real, w_quant_imag)
        
        # Power Normalization to satisfy Frobenius norm constraint
        norm = tf.norm(w_quant, ord='fro', axis=(-2,-1), keepdims=True)
        
        scale_factor = tf.cast(tf.sqrt(p_ue_max), dtype=tf.complex64)
        w_normalized = (w_quant / norm) * scale_factor
        
        codebook.append(w_normalized)
    return tf.stack(codebook)

# ──────────────────────────────────────────────────────────────────────────────
# MIMO RL ENVIRONMENT DEFINITION
# ──────────────────────────────────────────────────────────────────────────────
class MIMOEnvironment:
    def __init__(self, channels, encoder, V_k_list, s_k_list, snr_db, params):
        self.channels = channels
        self.encoder = encoder
        self.V_k_list = V_k_list
        self.s_k_list = s_k_list
        self.snr_db = snr_db
        self.params = params
        self.k_idx = 0
        snr_linear = 10**(self.snr_db / 10.0)
        self.noise_power = tf.cast(1.0 / snr_linear, dtype=tf.float32)
        self.H_history = {k: [] for k in range(self.params['K'])}

    def reset(self):
        self.H_history = {k: [] for k in range(self.params['K'])}
        state, H_k = self.get_state()
        while state is None:
            state, H_k = self.get_state()
        return state, H_k

    def get_state(self):
        h, path_delays = self.channels[self.k_idx](1, 1, self.params['sampling_frequency'])
        frequencies = subcarrier_frequencies(self.params['fft_size'], self.params['subcarrier_spacing'])
        H_freq = cir_to_ofdm_channel(frequencies, h, path_delays)
        H_k = tf.squeeze(H_freq[..., 0, 10])
        H_real_imag = tf.concat([tf.math.real(H_k), tf.math.imag(H_k)], axis=1)
        H_flat = tf.reshape(H_real_imag, [-1])
        self.H_history[self.k_idx].append(H_flat.numpy())
        if len(self.H_history[self.k_idx]) > self.params['tau']:
            self.H_history[self.k_idx].pop(0)
        if len(self.H_history[self.k_idx]) < self.params['tau']:
            return None, None
        X_k = np.stack(self.H_history[self.k_idx][-self.params['tau']:], axis=0)
        X_k_tensor = tf.convert_to_tensor(X_k, dtype=tf.float32)
        X_k_batch = tf.expand_dims(X_k_tensor, axis=0)
        phi_k = self.encoder(X_k_batch)
        return phi_k, H_k

    def step(self, action_index, H_k, codebook): # <--- FIX 1: Added 'codebook' argument
        """
        This function computes the reward for a given action using the provided codebook.
        """
        # --- FIX 2: Use the 'codebook' argument, not the global variable ---
        W_k = codebook[action_index]
        
        x_t = tf.zeros((self.params['Nt'], 1), dtype=tf.complex64)
        for i in range(self.params['K']):
            x_t += tf.matmul(self.V_k_list[i], self.s_k_list[i])
        noise_stddev = tf.sqrt(self.noise_power / 2.0)
        noise = tf.complex(
            tf.random.normal([self.params['Nr'], 1], stddev=noise_stddev),
            tf.random.normal([self.params['Nr'], 1], stddev=noise_stddev)
        )
        y_k = tf.matmul(H_k, x_t) + noise
        
        # Calculate SINR for user k based on Equation 6
        w_i = W_k[0, :]
        w_i_hermitian = tf.expand_dims(tf.transpose(tf.math.conj(w_i)), axis=0)
        signal_power = tf.square(tf.abs(tf.matmul(tf.matmul(w_i_hermitian, H_k), self.V_k_list[self.k_idx][:,0:1])))
        
        inter_user_interference = 0.0
        for l in range(self.params['K']):
            if l != self.k_idx:
                interference_term = tf.matmul(tf.matmul(w_i_hermitian, H_k), self.V_k_list[l])
                inter_user_interference += tf.reduce_sum(tf.square(tf.abs(interference_term)))
        
        intra_user_interference = 0.0
        if self.params['Ns'] > 1:
            interference_term = tf.matmul(tf.matmul(w_i_hermitian, H_k), self.V_k_list[self.k_idx][:,1:])
            intra_user_interference = tf.reduce_sum(tf.square(tf.abs(interference_term)))
            
        w_i_squared_norm = tf.reduce_sum(tf.square(tf.abs(w_i)))
        noise_power_at_output = self.noise_power * w_i_squared_norm
        
        sinr = signal_power / (inter_user_interference + intra_user_interference + noise_power_at_output + 1e-12)
        reward = tf.math.log(1.0 + sinr) / tf.math.log(2.0)
        
        next_phi_k, H_k_next = self.get_state()
        done = (next_phi_k is None)
        return next_phi_k, reward, done, H_k_next, y_k

# ──────────────────────────────────────────────────────────────────────────────
# SIMULATION PARAMETERS & INSTANTIATION
# ──────────────────────────────────────────────────────────────────────────────
# System Parameters
Nt, Nr, K, Ns, tau = 8, 2, 4, 2, 5
fft_size, num_ofdm_symbols, subcarrier_spacing, cp_length = 256, 14, 30e3, 16
mod_order, carrier_freq, delay_spread = 16, 3.5e9, 300e-9
bits_per_symbol = int(np.log2(mod_order))

sampling_frequency = fft_size * subcarrier_spacing

# RL and Codebook Parameters
P_UE_MAX, NUM_QUANT_BITS, CODEBOOK_SIZE, embedding_dim = 1.0, 4, 64, 128

# Model and Component Instantiation
encoder = CNNGRUEncoder(embedding_dim=embedding_dim)
actor = Actor(num_actions=CODEBOOK_SIZE)
critic = Critic()
combiner_codebook = create_combiner_codebook(CODEBOOK_SIZE, Ns, Nr, P_UE_MAX, NUM_QUANT_BITS)
bs_array = PanelArray(num_rows_per_panel=1, num_cols_per_panel=Nt, polarization="single", polarization_type="V", antenna_pattern="38.901", carrier_frequency=carrier_freq)
ue_array = PanelArray(num_rows_per_panel=1, num_cols_per_panel=Nr, polarization="single", polarization_type="V", antenna_pattern="omni", carrier_frequency=carrier_freq)
channels = [CDL("C", delay_spread, carrier_freq, ue_array, bs_array, "downlink") for _ in range(K)]

# Mapper and Demapper for BER calculation
mapper = Mapper("qam", mod_order)
demapper = Demapper("app", constellation_type="qam", num_bits_per_symbol=bits_per_symbol, hard_out=True)

# ResourceGrid defines the shape of the OFDM frame for a simple SISO link
resource_grid = ResourceGrid(num_ofdm_symbols=num_ofdm_symbols,
                             fft_size=fft_size,
                             subcarrier_spacing=subcarrier_spacing,
                             num_tx=1, # Single-Input
                             num_streams_per_tx=1)

# OFDM Modulator and Demodulator
modulator = OFDMModulator(cp_length=cp_length)
demodulator = OFDMDemodulator(fft_size=fft_size,
                              l_min=0,
                              cp_length=cp_length)
# ──────────────────────────────────────────────────────────────────────────────
# MAIN PPO TRAINING LOOP
# ──────────────────────────────────────────────────────────────────────────────
print("Starting PPO Agent Training...")
num_episodes, max_steps_per_episode, snr_db_train = 10, 100, 15.0

# --- NEW: Adaptive Learning Rate Schedule ---
# Start with a higher learning rate and decay it over time.
initial_learning_rate = 1e-5
decay_steps = 1000  # Decay the learning rate every 10,000 training steps
decay_rate = 0.95    # The rate of decay

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=True)

# --- Instantiation with the new schedule ---
# Pass the schedule object directly to the optimizer.
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)
ppo_agent = PPOAgent(actor, critic, optimizer)
# params = {'K': K, 'Nt': Nt, 'Nr': Nr, 'Ns': Ns, 'tau': tau, 'sampling_frequency': sampling_frequency, 'fft_size': fft_size, 'subcarrier_spacing': subcarrier_spacing}
params = {
    'K': K, 
    'Nt': Nt, 
    'Nr': Nr, 
    'Ns': Ns, 
    'tau': tau, 
    'sampling_frequency': sampling_frequency, 
    'fft_size': fft_size, 
    'subcarrier_spacing': subcarrier_spacing,
    'num_ofdm_symbols': num_ofdm_symbols, # <-- Add this
    'bits_per_symbol': bits_per_symbol   # <-- Add this
}

# --- Fixed Precoding & Symbols for the Environment ---
fixed_s_k = [tf.constant((2*np.random.randint(0,2,Ns)+np.random.randint(0,2,Ns)) + 1j*(2*np.random.randint(0,2,Ns)+np.random.randint(0,2,Ns)), shape=(Ns,1), dtype=tf.complex64) for _ in range(K)]
temp_H_k = [tf.squeeze(cir_to_ofdm_channel(subcarrier_frequencies(fft_size, subcarrier_spacing), *channels[k](1,1,sampling_frequency))[...,0,10]) for k in range(K)]


# --- FINAL CORRECTED RBD PRECODER GENERATION LOOP ---
fixed_V_k = []
print("--- Calculating Fixed RBD Precoders ---")

for k in range(K):
    # Form the interference channel matrix for all other users
    H_interf = tf.concat([temp_H_k[i] for i in range(K) if i != k], axis=0)
    
    # Decompose the interference channel to find its nullspace
    # tf.linalg.svd returns s, u, v
    s_interf, _, v_interf = tf.linalg.svd(H_interf)
    
    rank_interf = tf.shape(s_interf)[0]
    num_v_cols = tf.shape(v_interf)[1]

    if rank_interf < num_v_cols:
        # The basis for the nullspace are the last columns of v_interf
        T_k = v_interf[:, rank_interf:]
        
        # Project the desired user's channel into the nullspace
        H_eff_k = tf.matmul(temp_H_k[k], T_k)
        
        # Decompose the effective channel to find the best stream directions
        _ , _, v_eff = tf.linalg.svd(H_eff_k)
        
        # The precoder for the effective channel are the FIRST Ns right singular vectors
        # which are the first Ns columns of v_eff
        V_eff_k = v_eff[:, :Ns]
        
        # Project the precoder back to the original antenna space
        V_k = tf.matmul(T_k, V_eff_k)
    else:
        # Fallback if there is no nullspace
        V_k = tf.eye(Nt, num_columns=Ns, dtype=tf.complex64)
    
    tf.print(f"User {k}: Generated V_k with shape:", tf.shape(V_k))
    
    fixed_V_k.append(V_k)

print("--- Finished Calculating Precoders ---")



env = MIMOEnvironment(channels, encoder, fixed_V_k, fixed_s_k, snr_db_train, params)

total_rewards_history = []
entropy_history = [] # <<< ADD THIS LINE: Initialize list to store entropy

for episode in range(num_episodes):
    states, actions, rewards, next_states, dones, old_probs = [], [], [], [], [], []
    state, H_k = env.reset()
    
    for t in range(max_steps_per_episode):
        action_probs = actor(state)
        action = tf.random.categorical(tf.math.log(action_probs), 1)[0, 0].numpy()
        old_prob = action_probs[0, action].numpy()

        # --- CORRECTED LINE ---
        # Pass the main 'combiner_codebook' to the step function
        next_state, reward, done, H_k_next, _ = env.step(action, H_k, combiner_codebook)
        
        if next_state is not None:
            states.append(tf.squeeze(state).numpy())
            actions.append(action)
            rewards.append(tf.squeeze(reward).numpy())
            next_states.append(tf.squeeze(next_state).numpy())
            dones.append(done)
            old_probs.append(old_prob)
            
        if done: break
        state, H_k = next_state, H_k_next
    if len(states) > 1:
        values = critic(np.array(states)).numpy().flatten()
        next_values = critic(np.array(next_states)).numpy().flatten()
        returns, advantages = ppo_agent._compute_advantages_and_returns(np.array(rewards), values, next_values, dones)
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        #ppo_agent.train(np.array(states, dtype=np.float32), np.array(actions, dtype=np.int32), np.array(old_probs, dtype=np.float32), np.array(returns, dtype=np.float32), np.array(advantages, dtype=np.float32))
        
        # Train the agent using the collected experience
        ppo_agent.train(np.array(states, dtype=np.float32), np.array(actions, dtype=np.int32), np.array(old_probs, dtype=np.float32), np.array(returns, dtype=np.float32), np.array(advantages, dtype=np.float32))
       
        # Calculate and store the policy entropy for analysis
        current_probs = actor(np.array(states, dtype=np.float32))
        current_entropy = -tf.reduce_mean(tf.reduce_sum(current_probs * tf.math.log(current_probs + 1e-10), axis=1))
        entropy_history.append(current_entropy.numpy())
        
                
    episode_reward = sum(rewards)
    total_rewards_history.append(episode_reward)
    if (episode + 1) % 10 == 0:
        avg_reward = np.mean(total_rewards_history[-100:])
        print(f"Episode: {episode+1}, Total Reward: {episode_reward:.2f}, Avg Reward (last 100): {avg_reward:.2f}")

print("\nTraining finished.")
actor.save_weights("ppo_actor.weights.h5")
critic.save_weights("ppo_critic.weights.h5")
encoder.save_weights("encoder.weights.h5")
print("Trained model weights saved successfully.")



print("\nStarting evaluation phase...")

# ──────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS FOR BASELINE ALGORITHMS
# ──────────────────────────────────────────────────────────────────────────────
def calculate_sinr(W_k, H_k, V_k_list, k_idx, noise_power, params):
    """A generic function to calculate SINR for a given combiner W_k."""
    Ns = params['Ns']
    K = params['K']
    
    # Assuming first stream for simplicity
    w_i = W_k[0, :]
    w_i_hermitian = tf.transpose(tf.math.conj(w_i), conjugate=False)
    w_i_hermitian = tf.expand_dims(w_i_hermitian, axis=0)
    
    signal_power = tf.square(tf.abs(tf.matmul(tf.matmul(w_i_hermitian, H_k), V_k_list[k_idx][:,0:1])))

    inter_user_interference = 0.0
    for l in range(K):
        if l != k_idx:
            interference_term = tf.matmul(tf.matmul(w_i_hermitian, H_k), V_k_list[l])
            inter_user_interference += tf.reduce_sum(tf.square(tf.abs(interference_term)))

    intra_user_interference = 0.0
    if Ns > 1:
        interference_term = tf.matmul(tf.matmul(w_i_hermitian, H_k), V_k_list[k_idx][:,1:])
        intra_user_interference = tf.reduce_sum(tf.square(tf.abs(interference_term)))
        
    w_i_squared_norm = tf.reduce_sum(tf.square(tf.abs(w_i)))
    noise_power_at_output = noise_power * w_i_squared_norm
    sinr = signal_power / (inter_user_interference + intra_user_interference + noise_power_at_output + 1e-12)
    return sinr



def mrc_combiner(H_k, V_k, params):
    """Maximal Ratio Combining for a precoded system: W = (H_k * V_k)^H"""
    # Calculate the effective channel for this user's streams
    H_eff = tf.matmul(H_k, V_k) # Shape: [Nr, Ns]
    
    # The MRC combiner is the conjugate transpose of the effective channel
    W_mrc = tf.transpose(tf.math.conj(H_eff)) # Shape: [Ns, Nr]
    return W_mrc



def mmse_combiner(H_k, V_k_list, noise_power, params):
    """Minimum Mean Square Error Combining (Numerically Stable Version)"""
    K, Nt, Nr, k_idx = params['K'], params['Nt'], params['Nr'], 0

    # Calculate R_yy = H_k * (SUM_j V_j*V_j^H) * H_k^H + noise_power*I
    transmit_covariance = tf.zeros([Nt, Nt], dtype=tf.complex64)
    for i in range(K):
        transmit_covariance += tf.matmul(V_k_list[i], V_k_list[i], transpose_b=True)
    R_yy = tf.matmul(H_k, tf.matmul(transmit_covariance, H_k, transpose_b=True))
    noise_cov = tf.cast(noise_power, dtype=tf.complex64) * tf.eye(Nr, dtype=tf.complex64)
    R_yy += noise_cov

    # Calculate R_sy = V_k^H * H_k^H
    R_sy = tf.matmul(V_k_list[k_idx], H_k, transpose_a=True, transpose_b=True)

    # Use the more stable least-squares solver to find W^H = inv(R_yy) * R_ys
    # So, we solve R_yy * W^H = R_ys, where R_ys = R_sy^H
    R_ys = tf.transpose(R_sy, conjugate=True)
    W_mmse_hermitian = tf.linalg.lstsq(R_yy, R_ys)
    
    # The final combiner is the conjugate transpose
    W_mmse = tf.transpose(W_mmse_hermitian, conjugate=True)
    
    return W_mmse



def run_siso_ofdm_ber(noise_variance, params):
    """
    Runs a simple SISO OFDM simulation over an AWGN channel (Final Corrected Version based on Sionna guidance).
    """
    global mapper, demapper, modulator, demodulator, AWGN, ResourceGrid, ResourceGridMapper, ResourceGridDemapper
    
    # 1. تعریف گرید و مپر برای یک لینک ساده SISO
    rg_siso = ResourceGrid(num_ofdm_symbols=params['num_ofdm_symbols'],
                           fft_size=params['fft_size'],
                           subcarrier_spacing=params['subcarrier_spacing'],
                           num_tx=1,
                           num_streams_per_tx=1)
    siso_mapper = ResourceGridMapper(rg_siso)
    
    batch_size = 64
    
    # 2. تولید بیت‌ها به روش صحیح
    # تعداد نمادهای داده را مستقیماً از گرید می‌گیریم
    num_data_symbols = rg_siso.num_data_symbols
    # تعداد بیت‌ها را بر اساس هر آیتم در بچ محاسبه می‌کنیم
    num_bits_per_batch_item = num_data_symbols * params['bits_per_symbol']
    bits = tf.random.uniform(shape=[batch_size, num_bits_per_batch_item], maxval=2, dtype=tf.int32)
    
    # 3. تبدیل بیت‌ها به نماد
    symbols = mapper(bits)
    
    # 4. تغییر شکل نمادها به فرمت صحیح برای مپر
    # فرمت مورد انتظار: [batch_size, num_tx, num_streams_per_tx, num_data_symbols]
    # با استفاده از -1، خود TensorFlow بعد آخر را به درستی محاسبه می‌کند
    symbols_for_mapper = tf.reshape(symbols, [batch_size, 1, 1, -1])

    # 5. قرار دادن نمادها روی گرید
    rg_filled_tensor = siso_mapper(symbols_for_mapper)
    
    # مراحل بعدی بدون تغییر
    x_time = modulator(rg_filled_tensor)
    y_time = AWGN()([x_time, noise_variance])
    x_demod = demodulator(y_time)
    symbols_hat, _ = ResourceGrid.get_symbols(x_demod, rg_siso)
    bits_hat = demapper(symbols_hat)
    
    num_errors = tf.reduce_sum(tf.cast(bits != bits_hat, tf.float32))
    return num_errors, tf.cast(tf.size(bits), tf.float32)


def run_mu_mimo_ber(combiner, H_k_freq, V_k_list, k_idx, noise_variance, params):
    """
    Runs a full MU-MIMO OFDM link simulation using the correct Sionna workflow (Map-then-Precode).
    FINAL CORRECTED VERSION.
    """
    global mapper, demapper
    
    batch_size = 16
    num_total_streams = params['K'] * params['Ns']

    # --- فرستنده (Transmitter) ---
    
    # 1. تعریف گرید و مپر برای تمام استریم‌های داده
    rg_streams = ResourceGrid(num_ofdm_symbols=params['num_ofdm_symbols'],
                              fft_size=params['fft_size'],
                              subcarrier_spacing=params['subcarrier_spacing'],
                              num_tx=num_total_streams,
                              num_streams_per_tx=1)
    stream_mapper = ResourceGridMapper(rg_streams)

    # 2. تولید بیت‌ها و نمادها
    num_data_symbols = rg_streams.num_data_symbols
    num_bits_per_stream = num_data_symbols * params['bits_per_symbol']
    bits = tf.random.uniform(shape=[batch_size, params['K'], params['Ns'], num_bits_per_stream], minval=0, maxval=2, dtype=tf.int32)
    symbols = mapper(bits)
    
    # 3. شکل‌دهی مجدد نمادها برای مپر
    symbols_reshaped = tf.reshape(symbols, [batch_size, num_total_streams, -1])
    symbols_for_mapper = tf.expand_dims(symbols_reshaped, axis=2)
    
    # 4. مپ کردن استریم‌ها روی گرید (Map-then-Precode)
    x_mapped_streams = stream_mapper(symbols_for_mapper)

    # 5. ساخت پیش‌کُدکننده کامل و اعمال آن
    V_precoder = tf.concat(V_k_list, axis=1)
    x_total_freq = tf.einsum('tn,bnsf->btsf', V_precoder, x_mapped_streams)

    # --- کانال و گیرنده (Channel and Receiver) ---

    # 6. اعمال کانال فرکانسی
    H_k_batch = tf.expand_dims(H_k_freq, axis=0)
    y_freq_faded = tf.einsum('brt...,btf...->brf...', H_k_batch, x_total_freq)
    
    # 7. اضافه کردن نویز
    y_freq_noisy = AWGN()([y_freq_faded, noise_variance])
    
    # 8. اعمال ترکیب‌کننده در گیرنده
    s_hat_freq = tf.einsum('sn,brf...->bsf...', combiner, y_freq_noisy)
    
    # 9. استخراج نمادهای داده از گرید (بخش اصلاح شده نهایی)
    # یک گرید برای گیرنده تعریف می‌کنیم (فقط برای استریم‌های کاربر مورد نظر)
    rg_rx = ResourceGrid(num_ofdm_symbols=params['num_ofdm_symbols'],
                         fft_size=params['fft_size'],
                         subcarrier_spacing=params['subcarrier_spacing'],
                         num_tx=params['Ns'],
                         num_streams_per_tx=1)
    
    # از متد استاتیک و ساده get_symbols برای استخراج نمادها استفاده می‌کنیم
    s_hat, _ = ResourceGrid.get_symbols(s_hat_freq, rg_rx)
    
    # 10. دمدولاسیون نمادها به بیت
    bits_hat = demapper(s_hat)
    
    # 11. محاسبه خطا
    original_bits_k = tf.reshape(bits[:, k_idx, :, :], [batch_size, -1])
    num_errors = tf.reduce_sum(tf.cast(original_bits_k != bits_hat, tf.float32))
    
    return num_errors, tf.size(original_bits_k, out_type=tf.float32)



print("\nStarting final evaluation phase...")

results_throughput = {"PPO": [], "RBD": [], "MRC": [], "MMSE": []}
results_ber = {"PPO": [], "RBD": [], "MRC": [], "MMSE": [], "OFDM_AWGN": []}

num_eval_steps = 100 # Number of channel realizations to average over

# Load the trained models
encoder.load_weights("encoder.weights.h5")
actor.load_weights("ppo_actor.weights.h5")
print("Loaded trained model weights for evaluation.")

snr_dBs_eval = np.arange(-20, 22, 2)

for snr_db in snr_dBs_eval:
    print(f"Evaluating SNR = {snr_db} dB...")
    
    # --- Setup for this SNR point ---
    noise_power_eval = tf.cast(10**(-snr_db / 10.0), dtype=tf.float32)
    noise_variance_eval = 10**(-snr_db / 10.0)
    
    # --- Initialize accumulators ---
    avg_throughputs = {name: [] for name in results_throughput.keys()}
    total_errors = {name: 0.0 for name in results_ber.keys()}
    total_bits = {name: 0.0 for name in results_ber.keys()}
    
    # Create environment to get channel states for the PPO agent
    eval_env = MIMOEnvironment(channels, encoder, fixed_V_k, fixed_s_k, snr_db, params)

    for step in range(num_eval_steps):
        # --- Get a new channel realization for this step ---
        h, path_delays = channels[0](1, 1, sampling_frequency)
        frequencies = subcarrier_frequencies(fft_size, subcarrier_spacing)
        H_k_freq_domain = cir_to_ofdm_channel(frequencies, h, path_delays)
        
        # We use the channel at a single subcarrier for SINR calculation and PPO state
        H_k_single_sc = tf.squeeze(H_k_freq_domain[..., 0, 10])

        # --- Get combiners for all methods ---
        # PPO Agent
        state, _ = eval_env.get_state() # Use the env just to get the state
        while state is None: state, _ = eval_env.get_state()
        action_probs = actor(state)
        best_action = tf.argmax(action_probs, axis=1)[0].numpy()
        W_ppo = combiner_codebook[best_action]
        
        # Baselines
        W_rbd = tf.transpose(tf.linalg.lstsq(tf.matmul(H_k_single_sc, fixed_V_k[0]), tf.eye(Ns, dtype=tf.complex64)), conjugate=True)
        W_mrc = mrc_combiner(H_k_single_sc, fixed_V_k[0], params)
        W_mmse = mmse_combiner(H_k_single_sc, fixed_V_k, noise_power_eval, params)

        combiners = {"PPO": W_ppo, "RBD": W_rbd, "MRC": W_mrc, "MMSE": W_mmse}

        # --- Calculate Throughput and BER for each combiner ---
        for name, W in combiners.items():
            # Throughput (from single-subcarrier SINR)
            sinr = calculate_sinr(W, H_k_single_sc, fixed_V_k, 0, noise_power_eval, params)
            avg_throughputs[name].append(tf.math.log(1.0 + sinr) / tf.math.log(2.0))
            
            # BER (from full OFDM simulation)
            #errors, bits = run_mu_mimo_ber(W, H_k_freq_domain, fixed_V_k, 0, noise_variance_eval, params)
            #total_errors[name] += errors
            #total_bits[name] += bits

    # --- Calculate BER for the simple SISO OFDM baseline ---
    #errors, bits = run_siso_ofdm_ber(noise_variance_eval, params)
    #total_errors["OFDM_AWGN"] += errors
    #total_bits["OFDM_AWGN"] += bits

    # --- Store final results for this SNR point ---
    for name in results_throughput.keys():
        results_throughput[name].append(np.mean([t.numpy() for t in avg_throughputs[name]]))
    
    for name in results_ber.keys():
        if total_bits[name] > 0:
            results_ber[name].append(total_errors[name] / total_bits[name])
        else: # Avoid division by zero if a sim fails
            results_ber[name].append(1.0)


# ──────────────────────────────────────────────────────────────────────────────
# NEW EXPERIMENT: LATENT SPACE VISUALIZATION (CNN-GRU ANALYSIS)
# ──────────────────────────────────────────────────────────────────────────────
print("\nGenerating Latent Space Visualization...")

# We will collect states (phi) and the agent's chosen actions
latent_states = []
chosen_actions = []
num_viz_steps = 1000 # How many data points to collect for the plot
snr_db_viz = 10.0   # A representative SNR

viz_env = MIMOEnvironment(channels, encoder, fixed_V_k, fixed_s_k, snr_db_viz, params)
state, H_k = viz_env.reset()

for _ in range(num_viz_steps):
    while state is None:
        state, H_k = viz_env.get_state()
    
    # Store the state
    latent_states.append(tf.squeeze(state).numpy())
    
    # Get the agent's deterministic best action for this state
    action_probs = actor(state)
    best_action = tf.argmax(action_probs, axis=1)[0].numpy()
    chosen_actions.append(best_action)
    
    # Step the environment to get the next state
    state, _, _, H_k, _ = viz_env.step(best_action, H_k, combiner_codebook)

# Use t-SNE to project the high-dimensional latent space to 2D
tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
latent_2d = tsne.fit_transform(np.array(latent_states))


# ──────────────────────────────────────────────────────────────────────────────
# PLOTTING RESULTS
# ──────────────────────────────────────────────────────────────────────────────
import os # <-- ماژول os برای کار با فایل و پوشه اضافه شد

# نام پوشه برای ذخیره نتایج
output_dir = "results"

# اگر پوشه وجود نداشت، آن را بساز
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Directory '{output_dir}' created.")

# --- Plot 1: Throughput vs. SNR ---
plt.figure(figsize=(10, 7))
#plt.plot(snr_dBs_eval, results_throughput["PPO"], 'o-', label='PPO Agent', linewidth=2)
#plt.plot(snr_dBs_eval, results_throughput["RBD"], 's--', label='RBD', linewidth=2)
#plt.plot(snr_dBs_eval, results_throughput["MMSE"], 'd-.', label='MMSE', linewidth=2)
#plt.plot(snr_dBs_eval, results_throughput["MRC"], '^:', label='MRC', linewidth=2)
plt.xlabel("SNR (dB)", fontsize=14)
plt.ylabel("Average Spectral Efficiency (bits/s/Hz)", fontsize=14)
plt.title("Performance Comparison of Combining Strategies", fontsize=16)
plt.grid(True, which="both", linestyle='--')
plt.legend(fontsize=12)
plt.ylim(bottom=0)
# ذخیره نمودار در فایل
plt.savefig(os.path.join(output_dir, 'throughput_vs_snr.png'))
# نمایش نمودار
plt.show()

# --- Plot 2: BER vs. SNR (Consolidated) ---
plt.figure(figsize=(10, 7))
#plt.plot(snr_dBs_eval, results_ber["PPO"], 'o-', label='PPO Agent', linewidth=2)
#plt.plot(snr_dBs_eval, results_ber["RBD"], 's--', label='RBD', linewidth=2)
#plt.plot(snr_dBs_eval, results_ber["MMSE"], 'd-.', label='MMSE', linewidth=2)
#plt.plot(snr_dBs_eval, results_ber["MRC"], '^:', label='MRC', linewidth=2)
#plt.plot(snr_dBs_eval, results_ber["OFDM_AWGN"], 'x-k', label='OFDM (SISO AWGN)', linewidth=2)
plt.yscale('log')
plt.xlabel("SNR (dB)", fontsize=14)
plt.ylabel("Bit Error Rate (BER)", fontsize=14)
plt.title("BER Performance Comparison of Combining Strategies", fontsize=16)
plt.grid(True, which="both", linestyle='--')
plt.legend(fontsize=12)
plt.ylim(1e-5, 1.0)
# ذخیره نمودار در فایل
plt.savefig(os.path.join(output_dir, 'ber_vs_snr.png'))
# نمایش نمودار
plt.show()

# --- Plot 3: RL Agent Training Curve ---
plt.figure(figsize=(10, 7))
# Smooth the rewards curve for better visualization
def moving_average(data, window_size=20):
    """Computes the moving average of a 1D array."""
    if len(data) < window_size:
        # Not enough data to compute a moving average, return an empty array to plot nothing
        return np.array([])
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

# The rest of your plotting code
smoothed_rewards = moving_average(np.array(total_rewards_history))
# We only plot if there are smoothed rewards to show
if smoothed_rewards.size > 0:
    plt.plot(smoothed_rewards)
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Smoothed Average Reward", fontsize=14)
    plt.title("PPO Agent Learning Curve", fontsize=16)
    plt.grid(True)
    # ذخیره نمودار در فایل
    plt.savefig(os.path.join(output_dir, 'rl_training_curve.png'))
    # نمایش نمودار
    plt.show()

# # --- Plot 4: Quantization Analysis ---
# plt.figure(figsize=(10, 7))
# plt.plot(quantization_bits_to_test, final_quantization_results, 'o-', label=f'PPO Agent @ {snr_db_quant_analysis} dB SNR', linewidth=2)
# plt.xlabel("Number of Quantization Bits for Combiner", fontsize=14)
# plt.ylabel("Average Spectral Efficiency (bits/s/Hz)", fontsize=14)
# plt.title("Impact of Combiner Quantization on PPO Performance", fontsize=16)
# plt.grid(True, which="both", linestyle='--')
# plt.legend(fontsize=12)
# plt.xticks(quantization_bits_to_test)
# plt.ylim(bottom=0)
# plt.savefig(os.path.join(output_dir, 'quantization_analysis.png'))
# plt.show()

# --- Plot 5: Policy Entropy vs. Episode (RL Analysis) ---
plt.figure(figsize=(10, 7))
plt.plot(entropy_history)
plt.xlabel("Episode", fontsize=14)
plt.ylabel("Average Policy Entropy", fontsize=14)
plt.title("Policy Entropy During Training", fontsize=16)
plt.grid(True)
# ذخیره نمودار در فایل
plt.savefig(os.path.join(output_dir, 'policy_entropy.png'))
# نمایش نمودار
plt.show()

# --- Plot 6: t-SNE Visualization of Latent Space ---
# Create figure and axes objects explicitly
fig, ax = plt.subplots(figsize=(12, 10))

scatter = ax.scatter(latent_2d[:, 0], latent_2d[:, 1], c=chosen_actions, cmap='viridis', alpha=0.7)
ax.set_xlabel("t-SNE Component 1", fontsize=14)
ax.set_ylabel("t-SNE Component 2", fontsize=14)
ax.set_title("t-SNE Visualization of Encoder's Latent Space", fontsize=16)
ax.grid(True, linestyle='--')

# Create the legend and add it to the axes object 'ax'
legend1 = ax.legend(*scatter.legend_elements(), title="Chosen Actions")
ax.add_artist(legend1) # This now correctly calls the method on the Axes object
# ذخیره نمودار در فایل
# توجه: چون از fig, ax استفاده شده، متد savefig را روی fig فراخوانی می‌کنیم
fig.savefig(os.path.join(output_dir, 'tsne_latent_space.png'))
# نمایش نمودار
plt.show()
