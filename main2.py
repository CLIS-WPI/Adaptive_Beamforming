# pylint: disable=no-member
"""
simulation_throughput.py

An optimized MU-MIMO simulation for evaluating adaptive combining strategies
using a Proximal Policy Optimization (PPO) agent. This script focuses on
analyzing Spectral and Energy Efficiency.
"""
#import os
#import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sionna.phy.channel.tr38901 import CDL, PanelArray
from sionna.phy.channel import subcarrier_frequencies, cir_to_ofdm_channel


# --- DEVICE CONFIGURATION ---
# Set this to True to use the GPU; False to force CPU.
USE_GPU = False
if USE_GPU:
    print("Attempting to run on GPU...")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_visible_devices(gpus, 'GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Successfully configured to run on {len(gpus)} GPU(s).")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU found. The script will run on the CPU.")
else:
    tf.config.set_visible_devices([], 'GPU')
    print("USE_GPU is set to False. Forcing CPU execution.")

# Set Seed for Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.run_functions_eagerly(False) # Use graph execution for performance

#%%
# ──────────────────────────────────────────────────────────────────────────────
# STAGE 1: CNN-GRU ENCODER MODEL DEFINITION
# ──────────────────────────────────────────────────────────────────────────────
class CnnGruEncoder(tf.keras.Model):
    """A CNN-GRU model to extract spatio-temporal features from CSI."""
    def __init__(self, embedding_dim):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv1D(
            filters=64, kernel_size=3, activation='relu', padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv1D(
            filters=128, kernel_size=3, activation='relu', padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.gru = tf.keras.layers.GRU(units=embedding_dim, return_sequences=False)

    def call(self, inputs):
        """Processes a batch of CSI sequences."""
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        embedding = self.gru(x)
        return embedding

#%%
# ──────────────────────────────────────────────────────────────────────────────
# STAGE 2: REINFORCEMENT LEARNING (PPO) COMPONENTS
# ──────────────────────────────────────────────────────────────────────────────
# pylint: disable=too-few-public-methods
class Actor(tf.keras.Model):
    """The Actor network that outputs a policy (probability distribution)."""
    def __init__(self, num_actions):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(
            128, activation='relu', kernel_initializer='he_uniform')
        self.dense2 = tf.keras.layers.Dense(
            64, activation='relu', kernel_initializer='he_uniform')
        self.logits = tf.keras.layers.Dense(num_actions, activation=None)

    def call(self, state_input):
        """Forward pass to get action probabilities."""
        x = self.dense1(state_input)
        x = self.dense2(x)
        logits = self.logits(x)
        return tf.nn.softmax(logits)

class Critic(tf.keras.Model):
    """The Critic network that estimates the value of a state."""
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(
            128, activation='relu', kernel_initializer='he_uniform')
        self.dense2 = tf.keras.layers.Dense(
            64, activation='relu', kernel_initializer='he_uniform')
        self.value = tf.keras.layers.Dense(1, activation=None)

    def call(self, state_input):
        """Forward pass to get the state value."""
        x = self.dense1(state_input)
        x = self.dense2(x)
        value = self.value(x)
        return value

class PpoAgent:
    """The PPO Agent that handles training logic."""
    def __init__(self, actor_model, critic_model, optimizer, config):
        self.actor = actor_model
        self.critic = critic_model
        self.optimizer = optimizer
        # Hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.lambda_gae = config.get('lambda_gae', 0.95)
        self.epsilon_clip = config.get('epsilon_clip', 0.1)
        self.value_coeff = config.get('value_coeff', 0.5)
        self.entropy_coeff = config.get('entropy_coeff', 0.01)

    def _compute_advantages(self, rewards, values, next_values, dones):
        """Computes Generalized Advantage Estimation (GAE) and returns."""
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae_lam = 0.0

        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                last_gae_lam = 0.0
            else:
                delta = rewards[t] + self.gamma * next_values[t] - values[t]
            advantages[t] = last_gae_lam = (
                delta + self.gamma * self.lambda_gae * (1.0 - dones[t]) * last_gae_lam
            )
        returns = advantages + values
        return returns, advantages

    @tf.function
    def train(self, states, actions, old_probs, returns, advantages):
        """Executes a single training step for the PPO agent."""
        with tf.GradientTape() as tape:
            new_probs_dist = self.actor(states)
            values = self.critic(states)

            # FIX: A Dense layer's output dimension is found via its .units
            # attribute, not .shape.
            actions_one_hot = tf.one_hot(
                actions, self.actor.logits.units, dtype=tf.float32)

            new_action_probs = tf.reduce_sum(
                new_probs_dist * actions_one_hot, axis=1)

            # PPO Surrogate Objective
            ratio = new_action_probs / (old_probs + 1e-8)
            surr1 = ratio * advantages
            surr2 = tf.clip_by_value(
                ratio, 1.0 - self.epsilon_clip, 1.0 + self.epsilon_clip
            ) * advantages
            actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

            # Critic (Value) Loss
            critic_loss = tf.reduce_mean(tf.square(returns - tf.squeeze(values)))

            # Entropy Loss for exploration
            entropy = -tf.reduce_mean(
                tf.reduce_sum(new_probs_dist * tf.math.log(
                    new_probs_dist + 1e-10), axis=1))

            total_loss = (actor_loss +
                          (self.value_coeff * critic_loss) -
                          (self.entropy_coeff * entropy))

        all_vars = self.actor.trainable_variables + self.critic.trainable_variables
        gradients = tape.gradient(total_loss, all_vars)
        self.optimizer.apply_gradients(zip(gradients, all_vars))
        return total_loss, actor_loss, critic_loss, entropy

def create_combiner_codebook(num_matrices, num_streams, num_rx_antennas,
                             p_ue_max, num_quant_bits):
    """Creates a quantized and power-normalized codebook for the PPO agent."""
    codebook = []
    quant_levels = 2**num_quant_bits
    quant_step = 2.0 / (quant_levels - 1)
    for _ in range(num_matrices):
        real = tf.random.uniform([num_streams, num_rx_antennas], -1, 1)
        imag = tf.random.uniform([num_streams, num_rx_antennas], -1, 1)
        w_quant_real = tf.round((real + 1.0) / quant_step) * quant_step - 1.0
        w_quant_imag = tf.round((imag + 1.0) / quant_step) * quant_step - 1.0
        w_quant = tf.complex(w_quant_real, w_quant_imag)

        # Power Normalization
        norm = tf.norm(w_quant, ord='fro', axis=(-2,-1), keepdims=True)
        scale_factor = tf.cast(tf.sqrt(p_ue_max), dtype=tf.complex64)
        w_normalized = (w_quant / (norm + 1e-9)) * scale_factor

        codebook.append(w_normalized)
    return tf.stack(codebook)

#%%
# ──────────────────────────────────────────────────────────────────────────────
# PRECODING AND HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────
def calculate_precoders(sys_params):
    """
    Generates a list of random orthogonal precoding matrices.
    This is a robust placeholder to ensure correct tensor shapes.
    """
    num_tx = sys_params['nt']
    num_streams = sys_params['ns']
    num_users = sys_params['k']
    v_k_list = []
    for _ in range(num_users):
        # Create a random real-valued matrix of the correct size
        random_matrix = tf.random.normal(shape=[num_tx, num_streams])
        # Use QR decomposition to get an orthogonal basis (the Q matrix)
        q, _ = tf.linalg.qr(random_matrix)
        # Convert to complex type for the simulation and add to the list
        v_k_list.append(tf.cast(q, dtype=tf.complex64))
    return v_k_list

@tf.function
def calculate_sinr(w_k, h_k, v_k_list, user_idx, noise_power, sys_params):
    """A generic function to calculate SINR for a given combiner."""
    num_streams = sys_params['ns']
    num_users = sys_params['k']
    total_sinr = 0.0

    for i in range(num_streams):
        w_i = w_k[i, :]
        w_i_hermitian = tf.expand_dims(tf.transpose(tf.math.conj(w_i)), axis=0)
        v_i = v_k_list[user_idx][:, i:i+1]

        signal_power = tf.square(tf.abs(tf.matmul(w_i_hermitian,
                                                  tf.matmul(h_k, v_i))))

        # Intra-user interference
        intra_interference = 0.0
        if num_streams > 1:
            for j in range(num_streams):
                if i != j:
                    v_j = v_k_list[user_idx][:, j:j+1]
                    intra_interference += tf.square(
                        tf.abs(tf.matmul(w_i_hermitian, tf.matmul(h_k, v_j))))

        # Inter-user interference
        inter_interference = 0.0
        for l in range(num_users):
            if l != user_idx:
                interference_term = tf.matmul(w_i_hermitian,
                                              tf.matmul(h_k, v_k_list[l]))
                inter_interference += tf.reduce_sum(
                    tf.square(tf.abs(interference_term)))

        noise_at_output = noise_power * tf.reduce_sum(tf.square(tf.abs(w_i)))
        sinr_stream = signal_power / (
            intra_interference + inter_interference + noise_at_output + 1e-12)
        total_sinr += sinr_stream

    return tf.squeeze(total_sinr) / num_streams # Average SINR over streams

@tf.function
def mrc_combiner(h_k, v_k):
    """Maximal Ratio Combining for a precoded system."""
    h_eff = tf.matmul(h_k, v_k)
    w_mrc = tf.transpose(tf.math.conj(h_eff))
    return w_mrc

@tf.function
def mmse_combiner(h_k, v_k_list, noise_power, user_idx, sys_params):
    """Minimum Mean Square Error Combining."""
    num_tx = sys_params['nt']
    num_rx = sys_params['nr']
    num_users = sys_params['k']

    transmit_covariance = tf.zeros([num_tx, num_tx], dtype=tf.complex64)
    for i in range(num_users):
        transmit_covariance += tf.matmul(v_k_list[i], v_k_list[i],
                                         transpose_b=True)
    r_yy = tf.matmul(h_k, tf.matmul(transmit_covariance, h_k, transpose_b=True))
    noise_cov = tf.cast(noise_power, dtype=tf.complex64) * tf.eye(
        num_rx, dtype=tf.complex64)
    r_yy += noise_cov

    r_ys = tf.matmul(h_k, v_k_list[user_idx])
    w_mmse = tf.transpose(tf.linalg.solve(r_yy, r_ys), conjugate=True)
    return w_mmse


def rbd_combiner(h_k, v_k):
    """
    RBD at the receiver side (Zero-Forcing on the effective channel).
    This version manually computes the pseudo-inverse using SVD to support
    complex64 dtype, avoiding the TypeError with tf.linalg.pinv.
    """
    h_eff = tf.matmul(h_k, v_k)

    # Manually compute the pseudo-inverse of the complex matrix h_eff
    s, u, v = tf.linalg.svd(h_eff)

    # The pseudo-inverse of the singular value diagonal matrix is created by
    # taking the reciprocal of non-zero singular values.
    # A tolerance is used for numerical stability.
    tolerance = tf.cast(1e-6, dtype=s.dtype)
    s_inv = tf.where(s > tolerance, 1.0 / s, 0)
    
    # The pseudo-inverse of S is a diagonal matrix
    s_inv_diag = tf.linalg.diag(tf.cast(s_inv, dtype=tf.complex64))

    # Reconstruct the pseudo-inverse: H_pinv = V * S_pinv * U^H
    h_eff_pinv = tf.matmul(v, tf.matmul(s_inv_diag, u, adjoint_b=True))
    
    # The ZF/RBD combiner is the conjugate transpose of the pseudo-inverse
    w_rbd = tf.transpose(h_eff_pinv, conjugate=True)
    return w_rbd

#%%
# ──────────────────────────────────────────────────────────────────────────────
# MIMO RL ENVIRONMENT DEFINITION
# ──────────────────────────────────────────────────────────────────────────────
class MimoEnv:
    """A simulation environment for a multi-user MIMO system."""
    def __init__(self, channel_model, csi_encoder, snr_db, sys_params):
        self.channel_model = channel_model
        self.encoder = csi_encoder
        self.snr_db = snr_db
        self.params = sys_params
        self.user_idx = 0 # We focus on user 0's perspective
        self.noise_power = tf.cast(10**(-self.snr_db / 10.0), dtype=tf.float32)
        self.h_history = {k: [] for k in range(self.params['k'])}
        self.h_k_list = None
        self.v_k_list = None

    def reset(self):
        """Resets the environment and returns the initial state."""
        self.h_history = {k: [] for k in range(self.params['k'])}
        self._generate_channels_and_precoders()

        state = self._get_state(self.h_k_list[self.user_idx])
        while state is None: # Fill the history buffer
            self._generate_channels_and_precoders()
            state = self._get_state(self.h_k_list[self.user_idx])
        return state

    def _generate_channels_and_precoders(self):
        """Generates new channels and corresponding precoders."""
        sampling_freq = self.params['fft_size'] * self.params['subcarrier_spacing']
        h, path_delays = self.channel_model(
            self.params['k'], 1, sampling_freq)
        frequencies = subcarrier_frequencies(
            self.params['fft_size'], self.params['subcarrier_spacing'])
        h_freq = cir_to_ofdm_channel(frequencies, h, path_delays)

        h_k_list_tensor_slice = h_freq[..., 0, self.params['subcarrier_idx']]
        target_shape = [
            self.params['k'], self.params['nr'], self.params['nt']]
        h_k_list_tensor = tf.reshape(h_k_list_tensor_slice, target_shape)

        self.h_k_list = [h_k_list_tensor[i] for i in range(self.params['k'])]
        self.v_k_list = calculate_precoders(self.params)

    def _get_state(self, h_k):
        """Processes a channel matrix to generate a state for the agent."""
        h_real_imag = tf.concat([tf.math.real(h_k), tf.math.imag(h_k)], axis=-1)
        h_flat = tf.reshape(h_real_imag, [-1])

        self.h_history[self.user_idx].append(h_flat.numpy())
        if len(self.h_history[self.user_idx]) > self.params['tau']:
            self.h_history[self.user_idx].pop(0)

        if len(self.h_history[self.user_idx]) < self.params['tau']:
            return None # Not enough history yet

        x_k = np.stack(self.h_history[self.user_idx], axis=0)
        x_k_tensor = tf.convert_to_tensor(x_k, dtype=tf.float32)
        x_k_batch = tf.expand_dims(x_k_tensor, axis=0)

        phi_k = self.encoder(x_k_batch)
        return phi_k
    
    #  modifing the step function to introduce CSI delay (performance of baselines should degrade)
    
    def step(self, action_index, codebook):
        """
        Executes one step in the environment, simulating CSI delay.
        Returns the next state, the reward, a done flag, AND the outdated CSI.
        """
        # 1. Store the channel state that the agent is currently seeing. This is now "outdated".
        h_k_list_outdated = self.h_k_list
        v_k_list_outdated = self.v_k_list
        
        # 2. The agent makes a decision based on this outdated state
        w_k = codebook[action_index]
        
        # 3. The true channel evolves to a new, current state
        self._generate_channels_and_precoders()
        
        # 4. The reward is calculated by applying the action to the NEW channel state
        sinr = calculate_sinr(w_k, self.h_k_list[self.user_idx], self.v_k_list,
                              self.user_idx, self.noise_power, self.params)

        clipped_sinr = tf.maximum(sinr, 1e-9)
        reward = tf.math.log(1.0 + clipped_sinr) / tf.math.log(2.0)
        reward_val = tf.squeeze(reward)

        # 5. The NEXT state for the agent is generated from the NEWEST channel
        next_phi_k = self._get_state(self.h_k_list[self.user_idx])
        done = (next_phi_k is None)

        # 6. Return the outdated channel info along with the other values
        return next_phi_k, reward_val, done, h_k_list_outdated, v_k_list_outdated
    
    
    ## the original one
    # def step(self, action_index, codebook):
    #     """Executes one step in the environment."""
    #     w_k = codebook[action_index]
    #     sinr = calculate_sinr(w_k, self.h_k_list[self.user_idx], self.v_k_list,
    #                           self.user_idx, self.noise_power, self.params)

    #     # Add robustness to reward calculation
    #     clipped_sinr = tf.maximum(sinr, 1e-9)
    #     reward = tf.math.log(1.0 + clipped_sinr) / tf.math.log(2.0)
    #     reward_val = tf.squeeze(reward)

    #     # Get next state
    #     self._generate_channels_and_precoders()
    #     next_phi_k = self._get_state(self.h_k_list[self.user_idx])
    #     done = (next_phi_k is None)

    #     return next_phi_k, reward_val, done
#%%
# ──────────────────────────────────────────────────────────────────────────────
# SIMULATION PARAMETERS
# ──────────────────────────────────────────────────────────────────────────────
# System Parameters
PARAMS = {
    'nt': 8, 'nr': 2, 'k': 4, 'ns': 2, 'tau': 8,
    'fft_size': 256, 'subcarrier_spacing': 30e3, 'carrier_freq': 3.5e9,
    'delay_spread': 300e-9, 'subcarrier_idx': 50
}
PARAMS['sampling_frequency'] = PARAMS['fft_size'] * PARAMS['subcarrier_spacing']

# RL and Codebook Parameters
P_UE_MAX = 1.0
NUM_QUANT_BITS = 4
CODEBOOK_SIZE = 256
EMBEDDING_DIM = 128
PPO_CONFIG = {
    'gamma': 0.99, 'lambda_gae': 0.95, 'epsilon_clip': 0.1,
    'value_coeff': 0.5, 'entropy_coeff': 0.01
}


#%%
# ──────────────────────────────────────────────────────────────────────────────
# TRAINING PHASE
# ──────────────────────────────────────────────────────────────────────────────
print("--- Starting PPO Agent Training ---")
NUM_EPISODES = 5000
MAX_STEPS_PER_EPISODE = 200
SNR_DB_TRAIN = 15.0

# Model Instantiation
encoder = CnnGruEncoder(embedding_dim=EMBEDDING_DIM)
actor = Actor(num_actions=CODEBOOK_SIZE)
critic = Critic()
combiner_codebook = create_combiner_codebook(
    CODEBOOK_SIZE, PARAMS['ns'], PARAMS['nr'], P_UE_MAX, NUM_QUANT_BITS)

# Learning Rate Schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=3e-4, decay_steps=500, decay_rate=0.9,
    staircase=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)
ppo_agent = PpoAgent(actor, critic, optimizer, PPO_CONFIG)

# Channel Model Setup
bs_array = PanelArray(
    num_rows_per_panel=1, num_cols_per_panel=PARAMS['nt'],
    polarization="single", polarization_type="V",
    antenna_pattern="38.901", carrier_frequency=PARAMS['carrier_freq'])
ue_array = PanelArray(
    num_rows_per_panel=1, num_cols_per_panel=PARAMS['nr'],
    polarization="single", polarization_type="V",
    antenna_pattern="omni", carrier_frequency=PARAMS['carrier_freq'])
channel = CDL(
    "C", PARAMS['delay_spread'], PARAMS['carrier_freq'],
    ue_array, bs_array, "downlink", min_speed=1.0)

env = MimoEnv(channel, encoder, SNR_DB_TRAIN, PARAMS)

total_rewards_history = []
entropy_history = []

for episode_idx in range(NUM_EPISODES):
    # Experience buffer for the episode
    states_buf, actions_buf, rewards_buf = [], [], []
    next_states_buf, dones_buf, old_probs_buf = [], [], []

    current_state = env.reset()
    episode_reward_sum = 0

    for t_step in range(MAX_STEPS_PER_EPISODE):
        action_probs_dist = actor(current_state)
        action = tf.random.categorical(
            tf.math.log(action_probs_dist), 1)[0, 0].numpy()
        old_prob = action_probs_dist[0, action].numpy()
        
        # if changed to consider the perfect CSI remove the last two outputs
        next_state, reward_val, done_val, _, _ = env.step(action, combiner_codebook)

        episode_reward_sum += reward_val.numpy()

        if next_state is not None:
            states_buf.append(tf.squeeze(current_state).numpy())
            actions_buf.append(action)
            rewards_buf.append(reward_val.numpy())
            next_states_buf.append(tf.squeeze(next_state).numpy())
            dones_buf.append(done_val)
            old_probs_buf.append(old_prob)

        if done_val:
            break
        current_state = next_state

    if len(states_buf) > 1:
        # Convert buffers to numpy arrays
        states_arr = np.array(states_buf, dtype=np.float32)
        actions_arr = np.array(actions_buf, dtype=np.int32)
        rewards_arr = np.array(rewards_buf, dtype=np.float32)
        next_states_arr = np.array(next_states_buf, dtype=np.float32)
        dones_arr = np.array(dones_buf)
        old_probs_arr = np.array(old_probs_buf, dtype=np.float32)

        values = critic(states_arr).numpy().flatten()
        next_values = critic(next_states_arr).numpy().flatten()

        returns_arr, advantages_arr = ppo_agent._compute_advantages(
            rewards_arr, values, next_values, dones_arr)

        # Normalize advantages
        advantages_arr = ((advantages_arr - np.mean(advantages_arr)) /
                        (np.std(advantages_arr) + 1e-8))

        # Train the agent
        _, _, _, entropy_val = ppo_agent.train(
            states_arr, actions_arr, old_probs_arr, returns_arr, advantages_arr)
        entropy_history.append(entropy_val.numpy())

    avg_episode_reward = episode_reward_sum / (t_step + 1)
    total_rewards_history.append(avg_episode_reward)
    if (episode_idx + 1) % 10 == 0:
        avg_reward_100 = np.mean(total_rewards_history[-100:])
        print(f"Episode: {episode_idx+1}, Avg Reward: {avg_reward_100:.3f}")

print("\n--- Training finished. Saving model weights. ---\n")
actor.save_weights("ppo_actor.weights.h5")
critic.save_weights("ppo_critic.weights.h5")
encoder.save_weights("encoder.weights.h5")

#%%
# ──────────────────────────────────────────────────────────────────────────────
# EVALUATION PHASE
# ──────────────────────────────────────────────────────────────────────────────
print("--- Starting Evaluation Phase ---")
encoder.load_weights("encoder.weights.h5")
actor.load_weights("ppo_actor.weights.h5")

NUM_EVAL_STEPS = 2000
SNR_DBS_EVAL = np.arange(-20, 22, 2)
RESULTS_SE = {"PPO": [], "RBD": [], "MMSE": [], "MRC": []}

# # This is for the case we use the online and  perfect CSI
# for snr_db_val in SNR_DBS_EVAL:
#     print(f"Evaluating SNR = {snr_db_val} dB...")
    
#     # FIX: Cast noise power to float32 to match the data type of TF tensors.
#     noise_power_val = tf.cast(10**(-snr_db_val / 10.0), dtype=tf.float32)
    
#     avg_se_tracker = {name: [] for name in RESULTS_SE}

#     eval_env = MimoEnv(channel, encoder, snr_db_val, PARAMS)
#     eval_state = eval_env.reset()

#     for _ in range(NUM_EVAL_STEPS):
#         h_list = eval_env.h_k_list
#         v_list = eval_env.v_k_list
#         h_user0 = h_list[0]

#         # PPO Agent
#         action_probs_ppo = actor(eval_state)
#         best_action_ppo = tf.argmax(action_probs_ppo, axis=1)[0].numpy()
#         w_ppo = combiner_codebook[best_action_ppo]
#         sinr_ppo = calculate_sinr(
#             w_ppo, h_user0, v_list, 0, noise_power_val, PARAMS)
#         avg_se_tracker["PPO"].append(np.log2(1 + sinr_ppo))

#         # Baselines
#         w_rbd = rbd_combiner(h_user0, v_list[0])
#         sinr_rbd = calculate_sinr(
#             w_rbd, h_user0, v_list, 0, noise_power_val, PARAMS)
#         avg_se_tracker["RBD"].append(np.log2(1 + sinr_rbd))

#         w_mrc = mrc_combiner(h_user0, v_list[0])
#         sinr_mrc = calculate_sinr(
#             w_mrc, h_user0, v_list, 0, noise_power_val, PARAMS)
#         avg_se_tracker["MRC"].append(np.log2(1 + sinr_mrc))

#         w_mmse = mmse_combiner(
#             h_user0, v_list, noise_power_val, 0, PARAMS)
#         sinr_mmse = calculate_sinr(
#             w_mmse, h_user0, v_list, 0, noise_power_val, PARAMS)
#         avg_se_tracker["MMSE"].append(np.log2(1 + sinr_mmse))

#         # Get next state for PPO agent
#         eval_state, _, _ = eval_env.step(best_action_ppo, combiner_codebook)

#     for name, se_values in avg_se_tracker.items():
#         RESULTS_SE[name].append(np.mean(se_values))

# Now this version is for outdated CSI

for snr_db_val in SNR_DBS_EVAL:
    print(f"Evaluating SNR = {snr_db_val} dB...")
    
    noise_power_val = tf.cast(10**(-snr_db_val / 10.0), dtype=tf.float32)
    avg_se_tracker = {name: [] for name in RESULTS_SE}

    eval_env = MimoEnv(channel, encoder, snr_db_val, PARAMS)
    eval_state = eval_env.reset()

    # Store the initial "outdated" channel from the reset state
    h_list_outdated = eval_env.h_k_list
    v_list_outdated = eval_env.v_k_list

    for _ in range(NUM_EVAL_STEPS):
        h_user0_outdated = h_list_outdated[0]

        # --- 1. All methods make decisions based on OUTDATED CSI ---
        # PPO Agent's decision
        action_probs_ppo = actor(eval_state)
        best_action_ppo = tf.argmax(action_probs_ppo, axis=1)[0].numpy()
        w_ppo = combiner_codebook[best_action_ppo]

        # Baselines' decisions (combiners are calculated from outdated channel)
        w_rbd = rbd_combiner(h_user0_outdated, v_list_outdated[0])
        w_mrc = mrc_combiner(h_user0_outdated, v_list_outdated[0])
        w_mmse = mmse_combiner(
            h_user0_outdated, v_list_outdated, noise_power_val, 0, PARAMS)
        
        # --- 2. The environment steps forward to the CURRENT channel state ---
        # The step function now returns the outdated CSI for the *next* iteration
        eval_state, _, _, h_list_outdated, v_list_outdated = eval_env.step(
            best_action_ppo, combiner_codebook)
        
        # Get the true, current channel for evaluation
        h_list_current = eval_env.h_k_list
        v_list_current = eval_env.v_k_list
        h_user0_current = h_list_current[0]
        
        # --- 3. SINR is calculated using combiners from OLD CSI on the CURRENT channel ---
        sinr_ppo = calculate_sinr(
            w_ppo, h_user0_current, v_list_current, 0, noise_power_val, PARAMS)
        avg_se_tracker["PPO"].append(np.log2(1 + sinr_ppo))

        sinr_rbd = calculate_sinr(
            w_rbd, h_user0_current, v_list_current, 0, noise_power_val, PARAMS)
        avg_se_tracker["RBD"].append(np.log2(1 + sinr_rbd))

        sinr_mrc = calculate_sinr(
            w_mrc, h_user0_current, v_list_current, 0, noise_power_val, PARAMS)
        avg_se_tracker["MRC"].append(np.log2(1 + sinr_mrc))

        sinr_mmse = calculate_sinr(
            w_mmse, h_user0_current, v_list_current, 0, noise_power_val, PARAMS)
        avg_se_tracker["MMSE"].append(np.log2(1 + sinr_mmse))

    for name, se_values in avg_se_tracker.items():
        RESULTS_SE[name].append(np.mean(se_values))

#%%
# ──────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 1: LATENT SPACE VISUALIZATION (t-SNE)
# ──────────────────────────────────────────────────────────────────────────────
print("\n--- Generating Latent Space Visualization ---")
latent_states, chosen_actions = [], []
NUM_VIZ_STEPS = 2000
SNR_DB_VIZ = 15.0

viz_env = MimoEnv(channel, encoder, SNR_DB_VIZ, PARAMS)
viz_state = viz_env.reset()

for _ in range(NUM_VIZ_STEPS):
    latent_states.append(tf.squeeze(viz_state).numpy())
    action_probs_viz = actor(viz_state)
    best_action_viz = tf.argmax(action_probs_viz, axis=1)[0].numpy()
    chosen_actions.append(best_action_viz)
    # if changed to previous perfect CSI remove the last two outputs
    viz_state, _, _, _, _ = viz_env.step(best_action_viz, combiner_codebook)

tsne_model = TSNE(n_components=2, perplexity=40, max_iter=1000, random_state=SEED)
latent_2d = tsne_model.fit_transform(np.array(latent_states))

#%%
# ──────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 2: SPECTRAL EFFICIENCY VS. NUMBER OF USERS
# ──────────────────────────────────────────────────────────────────────────────
print("\n--- Evaluating SE vs. Number of Users ---")
K_VALUES = [2, 4, 6, 8, 10, 12]
SNR_DB_FIXED = 20.0
NOISE_POWER_FIXED = 10**(-SNR_DB_FIXED / 10.0)
RESULTS_SE_VS_K = {"PPO": [], "RBD": [], "MMSE": [], "MRC": []}

for k_val in K_VALUES:
    print(f"Evaluating for K = {k_val} users...")
    temp_params = PARAMS.copy()
    temp_params['k'] = k_val
    avg_se_k = {name: [] for name in RESULTS_SE_VS_K}
    temp_h_history = []

    for _ in range(NUM_EVAL_STEPS):
        # Generate channels and precoders for k_val users
        h_k, p_delays = channel(
            temp_params['k'], 1, temp_params['sampling_frequency'])
        freqs = subcarrier_frequencies(
            temp_params['fft_size'], temp_params['subcarrier_spacing'])
        h_k_freq = cir_to_ofdm_channel(freqs, h_k, p_delays)

        h_k_list_sc_slice = h_k_freq[..., 0, temp_params['subcarrier_idx']]
        target_shape_k = [temp_params['k'], temp_params['nr'], temp_params['nt']]
        h_k_list_sc = tf.reshape(h_k_list_sc_slice, target_shape_k)

        h_k_list_eval = [h_k_list_sc[i] for i in range(temp_params['k'])]
        v_k_list_eval = calculate_precoders(temp_params)
        h_user0_eval = h_k_list_eval[0]

        # PPO State Generation
        h_real_imag_flat = tf.reshape(tf.concat(
            [tf.math.real(h_user0_eval), tf.math.imag(h_user0_eval)],
            axis=-1), [-1])
        temp_h_history.append(h_real_imag_flat.numpy())
        if len(temp_h_history) > temp_params['tau']:
            temp_h_history.pop(0)

        if len(temp_h_history) == temp_params['tau']:
            x_k_ppo = np.stack(temp_h_history, axis=0)
            x_k_tensor_ppo = tf.convert_to_tensor(x_k_ppo, dtype=tf.float32)
            state_ppo = encoder(tf.expand_dims(x_k_tensor_ppo, axis=0))

            action_probs_ppo_k = actor(state_ppo)
            best_action_ppo_k = tf.argmax(action_probs_ppo_k, axis=1)[0].numpy()
            w_ppo_k = combiner_codebook[best_action_ppo_k]
            sinr_ppo_k = calculate_sinr(w_ppo_k, h_user0_eval, v_k_list_eval,
                                        0, NOISE_POWER_FIXED, temp_params)
            avg_se_k["PPO"].append(np.log2(1 + sinr_ppo_k))

        # Baselines
        w_rbd_k = rbd_combiner(h_user0_eval, v_k_list_eval[0])
        sinr_rbd_k = calculate_sinr(w_rbd_k, h_user0_eval, v_k_list_eval,
                                    0, NOISE_POWER_FIXED, temp_params)
        avg_se_k["RBD"].append(np.log2(1 + sinr_rbd_k))

        w_mrc_k = mrc_combiner(h_user0_eval, v_k_list_eval[0])
        sinr_mrc_k = calculate_sinr(w_mrc_k, h_user0_eval, v_k_list_eval,
                                    0, NOISE_POWER_FIXED, temp_params)
        avg_se_k["MRC"].append(np.log2(1 + sinr_mrc_k))

        w_mmse_k = mmse_combiner(h_user0_eval, v_k_list_eval,
                                   NOISE_POWER_FIXED, 0, temp_params)
        sinr_mmse_k = calculate_sinr(w_mmse_k, h_user0_eval, v_k_list_eval,
                                     0, NOISE_POWER_FIXED, temp_params)
        avg_se_k["MMSE"].append(np.log2(1 + sinr_mmse_k))

    for name, se_vals in avg_se_k.items():
        if se_vals: # Ensure list is not empty before taking mean
            RESULTS_SE_VS_K[name].append(np.mean(se_vals))
        else:
            RESULTS_SE_VS_K[name].append(0) # Append 0 if no valid steps

#%%
# ──────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 3: ENERGY EFFICIENCY COMPARISON
# ──────────────────────────────────────────────────────────────────────────────
print("\n--- Evaluating Energy Efficiency ---")
# Power consumption model (values are illustrative, representing relative complexity)
POWER_CONSUMPTION = {
    "P_circuit_per_chain": 0.5, # Watt per RF chain
    "P_comp_MRC": 0.1,          # Watt for MRC computation
    "P_comp_RBD": 0.8,          # Watt for pseudo-inverse
    "P_comp_MMSE": 1.0,         # Watt for matrix inversion
    "P_comp_PPO": 0.4           # Watt for NN inference
}
P_CIRCUIT = PARAMS['nr'] * POWER_CONSUMPTION["P_circuit_per_chain"]
P_TOTAL = {
    "MRC": P_CIRCUIT + POWER_CONSUMPTION["P_comp_MRC"],
    "RBD": P_CIRCUIT + POWER_CONSUMPTION["P_comp_RBD"],
    "MMSE": P_CIRCUIT + POWER_CONSUMPTION["P_comp_MMSE"],
    "PPO": P_CIRCUIT + POWER_CONSUMPTION["P_comp_PPO"],
}

RESULTS_EE = {name: [] for name in RESULTS_SE}
for name, se_list in RESULTS_SE.items():
    # Convert SE (bits/s/Hz) to Throughput (bits/s) by assuming 1 Hz BW
    # This makes Energy Efficiency = Spectral Efficiency / Power
    throughput = np.array(se_list)
    RESULTS_EE[name] = throughput / P_TOTAL[name]

#%%
# ──────────────────────────────────────────────────────────────────────────────
# PLOTTING RESULTS
# ──────────────────────────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
FIG_WIDTH, FIG_HEIGHT = 8, 6

# --- Plot 1: Learning Curve ---
plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
plt.plot(np.convolve(total_rewards_history, np.ones(50)/50, mode='valid'))
plt.xlabel("Episode", fontsize=12)
plt.ylabel("Smoothed Average Reward (SE)", fontsize=12)
plt.title("PPO Agent Learning Curve", fontsize=14, weight='bold')
plt.show()

# --- Plot 2: Policy Entropy ---
plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
plt.plot(entropy_history)
plt.xlabel("Episode", fontsize=12)
plt.ylabel("Average Policy Entropy", fontsize=12)
plt.title("Policy Entropy During Training", fontsize=14, weight='bold')
plt.show()

# --- Plot 3: Spectral Efficiency vs. SNR ---
plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
plt.plot(SNR_DBS_EVAL, RESULTS_SE["PPO"], 'o-', label='PPO Agent', linewidth=2)
plt.plot(SNR_DBS_EVAL, RESULTS_SE["MMSE"], 'd-.', label='MMSE', linewidth=2)
plt.plot(SNR_DBS_EVAL, RESULTS_SE["RBD"], 's--', label='RBD (ZF)', linewidth=2)
plt.plot(SNR_DBS_EVAL, RESULTS_SE["MRC"], '^:', label='MRC', linewidth=2)
plt.xlabel("SNR (dB)", fontsize=12)
plt.ylabel("Average Spectral Efficiency (bits/s/Hz)", fontsize=12)
plt.title("Performance of Combining Strategies", fontsize=14, weight='bold')
plt.legend(fontsize=11)
plt.ylim(bottom=0)
plt.show()

# --- Plot 4: SE Bar Chart at a Specific SNR ---
SNR_TARGET = 16 # dB
try:
    snr_idx = list(SNR_DBS_EVAL).index(SNR_TARGET)
    labels = list(RESULTS_SE.keys())
    se_values_bar = [RESULTS_SE[name][snr_idx] for name in labels]

    plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    bars = plt.bar(
        labels, se_values_bar, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.ylabel("Spectral Efficiency (bits/s/Hz)", fontsize=12)
    plt.title(f"Spectral Efficiency at {SNR_TARGET} dB SNR",
              fontsize=14, weight='bold')
    plt.bar_label(bars, fmt='%.2f', fontsize=11)
    plt.show()
except (ValueError, IndexError):
    print(f"SNR target {SNR_TARGET} dB not found in evaluation range.")

# --- Plot 5: SE vs. Number of Users ---
plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
for name, marker in [("PPO", 'o-'), ("MMSE", 'd-.'),
                     ("RBD", 's--'), ("MRC", '^:')]:
    plt.plot(K_VALUES, RESULTS_SE_VS_K[name], marker,
             label=f'{name} Agent' if name == "PPO" else name, linewidth=2)
plt.xlabel("Number of Users (K)", fontsize=12)
plt.ylabel("Average Spectral Efficiency (bits/s/Hz)", fontsize=12)
plt.title(f"System Scalability at {SNR_DB_FIXED} dB SNR",
          fontsize=14, weight='bold')
plt.xticks(K_VALUES)
plt.legend(fontsize=11)
plt.ylim(bottom=0)
plt.show()

# --- Plot 6: Energy Efficiency vs. SNR ---
plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
for name, marker in [("PPO", 'o-'), ("MMSE", 'd-.'),
                     ("RBD", 's--'), ("MRC", '^:')]:
    plt.plot(SNR_DBS_EVAL, RESULTS_EE[name], marker,
             label=f'{name} Agent' if name == "PPO" else name, linewidth=2)
plt.xlabel("SNR (dB)", fontsize=12)
plt.ylabel("Energy Efficiency (bits/Joule)", fontsize=12)
plt.title("Energy Efficiency of Combining Strategies",
          fontsize=14, weight='bold')
plt.legend(fontsize=11)
plt.ylim(bottom=0)
plt.show()

# --- Plot 7: t-SNE Visualization of Latent Space ---
plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT + 1))
scatter = plt.scatter(
    latent_2d[:, 0], latent_2d[:, 1], c=chosen_actions, cmap='viridis', alpha=0.6)
plt.xlabel("t-SNE Component 1", fontsize=12)
plt.ylabel("t-SNE Component 2", fontsize=12)
plt.title("t-SNE Visualization of Encoder's Latent Space",
          fontsize=14, weight='bold')
plt.colorbar(scatter, label='Chosen Action (Combiner Index)')
plt.show()
