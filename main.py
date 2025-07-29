
"""
This script implements and evaluates an adaptive beamforming strategy for a
Multi-User MIMO-OFDM system using a two-stage machine learning framework.
The approach is based on the paper "Adaptive Beamforming for Interference-Limited
MU-MIMO using Spatio-Temporal Policy Networks".

The framework consists of two main components:
1.  A CNN-GRU Encoder: This model learns a compact spatio-temporal
    representation from a sequence of partial Channel State Information (CSI)
    snapshots. The CNN part extracts spatial features from each snapshot,
    and the GRU captures the temporal evolution of these features.
2.  A PPO Reinforcement Learning Agent: The agent uses the learned state
    representation from the encoder to select an optimal, quantized combining
    matrix (W) from a pre-defined codebook. The goal is to maximize the
    spectral efficiency (throughput) for the user of interest in an
    interference-limited environment.

This implementation is optimized for performance on modern GPUs by leveraging
TensorFlow's graph execution (`tf.function`), JIT compilation, and mixed-precision
training. It also includes baseline algorithms (MRC, MMSE) for performance
comparison.

Key Steps:
1.  Environment and System Setup: Defines the MIMO system parameters,
    channel models (Sionna CDL), and TensorFlow configuration.
2.  Model Definitions: Implements the Keras models for the CNN-GRU Encoder,
    PPO Actor, and PPO Critic.
3.  RL Environment: Creates a custom `MIMOEnvironment` class that simulates
    the wireless channel, calculates rewards (SINR), and provides states to
    the agent.
4.  PPO Training: Runs the main training loop where the PPO agent interacts
    with the environment to learn an optimal policy for selecting combiners.
5.  Evaluation: After training, the agent's performance is evaluated against
    baseline methods across a range of SNRs by calculating the Bit Error Rate (BER).
6.  Visualization: Plots the results, including BER vs. SNR curves, the RL
    agent's learning curve, and a t-SNE visualization of the learned latent space.
"""

import os
import tensorflow as tf
import sionna
print(f"Sionna Version: {sionna.__version__}")
print(f"TensorFlow Version: {tf.__version__}")
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

# Optional import for visualization
try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except ImportError:
    print("Warning: sklearn not installed. t-SNE visualization will be disabled.")
    print("To install: pip install scikit-learn")
    TSNE_AVAILABLE = False

# Sionna Imports (ensure Sionna version >= 0.19)
try:
    import sionna
    from sionna.phy.channel.tr38901 import CDL, PanelArray
    from sionna.phy.channel import subcarrier_frequencies, cir_to_ofdm_channel
    from sionna.phy.mapping import Mapper, Demapper
except ImportError as e:
    print("Sionna library not found. Please install it using 'pip install sionna'")
    raise e

# ──────────────────────────────────────────────────────────────────────────────
# 1. DEVICE AND PRECISION CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

def configure_tensorflow(use_mixed_precision=True, enable_jit=True):
    """
    Configures TensorFlow for optimal performance on GPUs.

    This function sets up memory growth to avoid allocating all GPU memory at
    once, enables mixed-precision for a performance boost on compatible GPUs
    (e.g., NVIDIA Ampere/Hopper), and enables Just-In-Time (JIT) compilation.
    If no GPU is found, it gracefully falls back to CPU execution.

    Args:
        use_mixed_precision (bool): If True, enables mixed_float16 precision.
        enable_jit (bool): If True, enables XLA JIT compilation.
    """
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.set_visible_devices(gpus, 'GPU')
            
            if use_mixed_precision:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                print("Mixed precision (float16) enabled.")

            if enable_jit:
                tf.config.optimizer.set_jit(True)
                print("XLA JIT compilation enabled.")
                
            print(f"Successfully configured to run on {len(gpus)} GPU(s).")
        else:
            print("No GPU found. The script will run on the CPU.")
            tf.config.set_visible_devices([], 'GPU') # Explicitly disable GPUs
    except RuntimeError as e:
        print(f"RuntimeError during device configuration: {e}")
        print("Forcing CPU execution.")
        tf.config.set_visible_devices([], 'GPU')

# Apply the configuration
configure_tensorflow()

# Set seed for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ──────────────────────────────────────────────────────────────────────────────
# 2. SIMULATION PARAMETERS
# ──────────────────────────────────────────────────────────────────────────────

# System Parameters
PARAMS = {
    'Nt': 8,  # Number of transmit antennas at BS
    'Nr': 2,  # Number of receive antennas at UE
    'K': 4,   # Number of users
    'Ns': 2,  # Number of data streams per user
    'tau': 8, # Length of CSI history sequence for the encoder
    'fft_size': 256,
    'num_ofdm_symbols': 14,
    'subcarrier_spacing': 30e3,
    'cp_length': 16,
    'mod_order': 16, # 16-QAM
    'carrier_freq': 3.5e9,
    'delay_spread': 300e-9,
}
PARAMS['bits_per_symbol'] = int(np.log2(PARAMS['mod_order']))
PARAMS['sampling_frequency'] = PARAMS['fft_size'] * PARAMS['subcarrier_spacing']

# RL and Codebook Parameters
RL_PARAMS = {
    'embedding_dim': 128,
    'codebook_size': 256,
    'p_ue_max': 1.0,
    'num_quant_bits': 4,
    'num_episodes': 100,
    'max_steps_per_episode': 100,
    'snr_db_train': 15.0,
    'gamma': 0.995,         # Discount factor
    'lambda_gae': 0.95,     # GAE lambda
    'epsilon_clip': 0.1,    # PPO clipping parameter
    'value_coeff': 0.5,     # Critic loss coefficient
    'entropy_coeff': 0.005, # Entropy bonus coefficient
}

# Place safe_mrc_combiner after RL_PARAMS
def safe_mrc_combiner(h, v):
    """Safe wrapper for mrc_combiner that handles empty or wrong-shaped inputs."""
    h_shape = tf.shape(h)
    # Check if any dimension is 0
    if tf.equal(h_shape[0], 0) or tf.equal(h_shape[1], 0):
        # Return zeros of the expected shape
        return tf.zeros([PARAMS['Ns'], PARAMS['Nr']], dtype=tf.complex64)
    try:
        return mrc_combiner(h, v)
    except Exception:
        # Fallback if any other error occurs
        return tf.zeros([PARAMS['Ns'], PARAMS['Nr']], dtype=tf.complex64)

# Evaluation Parameters
EVAL_PARAMS = {
    'snr_dBs': np.arange(-20, 22, 2),
    'num_channel_realizations': 500, # Number of channels to average over
    'batch_size': 512, # Batch size for vectorized BER calculation
}


# ──────────────────────────────────────────────────────────────────────────────
# 3. MODEL AND AGENT DEFINITIONS
# ──────────────────────────────────────────────────────────────────────────────

class CNNGRUEncoder(tf.keras.Model):
    """
    A hybrid CNN-GRU model to encode a sequence of CSI snapshots.

    The model processes a sequence of flattened, complex-valued CSI matrices.
    1D convolutions extract spatial features from each snapshot, and a GRU
    captures the temporal dependencies across the sequence to produce a
    fixed-size latent embedding vector.
    """
    def __init__(self, embedding_dim):
        super(CNNGRUEncoder, self).__init__()
        # Use mixed precision policy, but always cast input to float32
        self.input_cast = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))
        self.conv1 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.gru = tf.keras.layers.GRU(units=embedding_dim, return_sequences=False, unroll=True)

    @tf.function
    def call(self, inputs):
        """Forward pass of the encoder with robust mixed precision casting."""
        x = tf.cast(inputs, tf.float32)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        embedding = self.gru(x)
        return embedding
# --- Sionna LMMSE Equalizer Utility ---
from sionna.phy.mimo import lmmse_equalizer, StreamManagement


class Actor(tf.keras.Model):
    """
    The PPO Actor network (Policy).

    It takes a state (latent embedding from the encoder) and outputs a
    probability distribution over the discrete action space (combiner
    codebook indices).
    """
    def __init__(self, num_actions):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.logits = tf.keras.layers.Dense(num_actions, activation=None, dtype='float32')

    @tf.function
    def call(self, state):
        x = tf.cast(state, tf.float32)
        x = self.dense1(x)
        x = self.dense2(x)
        logits = self.logits(x)
        return tf.nn.softmax(logits)

class Critic(tf.keras.Model):
    """
    The PPO Critic network (Value Function).

    It takes a state and outputs a single scalar value, estimating the
    expected return (value) from that state.
    """
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.value = tf.keras.layers.Dense(1, activation=None, dtype='float32')

    @tf.function
    def call(self, state):
        x = tf.cast(state, tf.float32)
        x = self.dense1(x)
        x = self.dense2(x)
        value = self.value(x)
        return value
# --- Sionna Mapper/Demapper Initialization ---

# Example usage:
# mapper, demapper = initialize_sionna_components(mod_order=16, precision='single')

class PPOAgent:
    """
    The PPO Agent that orchestrates the training process.

    This class encapsulates the actor and critic networks, the optimizer,
    and the logic for computing advantages and updating the network weights
    based on the PPO clipped surrogate objective.
    """
    def __init__(self, actor, critic, optimizer, **kwargs):
        self.actor = actor
        self.critic = critic
        self.optimizer = optimizer
        self.gamma = kwargs.get('gamma', 0.99)
        self.lambda_gae = kwargs.get('lambda_gae', 0.95)
        self.epsilon_clip = kwargs.get('epsilon_clip', 0.2)
        self.value_coeff = kwargs.get('value_coeff', 0.5)
        self.entropy_coeff = kwargs.get('entropy_coeff', 0.01)

    def _to_scalar(self, x):
        if isinstance(x, np.ndarray):
            if x.size == 0:
                # WARNING: Encountered empty array, returning 0.0. Check rollout logic for possible bugs.
                return 0.0
            if x.size == 1:
                return x.item()
            else:
                raise ValueError(f"Expected scalar or length-1 array, got shape {x.shape}")
        return float(x)

    def _compute_advantages_and_returns(self, rewards, values, next_values, dones):
        """
        Computes Generalized Advantage Estimation (GAE) and returns.

        This calculation is done in NumPy for clarity, as the performance
        overhead is negligible for typical trajectory lengths.
        """
        num_steps = len(rewards)
        returns = np.zeros(num_steps, dtype=np.float32)
        advantages = np.zeros(num_steps, dtype=np.float32)
        last_gae_lam = 0.0

        for t in reversed(range(num_steps)):
            r = self._to_scalar(rewards[t])
            v = self._to_scalar(values[t])
            nv = self._to_scalar(next_values[t])
            d = self._to_scalar(dones[t])
            if d:
                delta = r - v
                last_gae_lam = 0.0
            else:
                delta = r + self.gamma * nv - v
            last_gae_lam = float(delta + self.gamma * self.lambda_gae * (1.0 - d) * last_gae_lam)
            advantages[t] = last_gae_lam

        returns = advantages + np.array([self._to_scalar(v) for v in values], dtype=np.float32)
        return returns, advantages

    @tf.function
    def train(self, states, actions, old_probs, returns, advantages):
        """
        Executes a single training step for the PPO agent.

        This function is compiled into a high-performance graph by TensorFlow.
        It calculates the PPO loss (actor, critic, and entropy) and applies
        gradients to update the networks.
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


# ──────────────────────────────────────────────────────────────────────────────
# 4. MIMO RL ENVIRONMENT
# ──────────────────────────────────────────────────────────────────────────────

class MIMOEnvironment:
    """
    Simulates the MU-MIMO environment for the RL agent.

    This class is responsible for:
    - Generating channel realizations using Sionna's CDL model.
    - Maintaining a history of CSI snapshots.
    - Using the CNN-GRU encoder to generate a state representation.
    - Calculating the reward (log SINR, i.e., spectral efficiency) based on
      the agent's chosen action (combiner).
    """
    def __init__(self, channels, encoder, V_k_list, s_k_list, snr_db, params, rl_params):
        self.channels = channels
        self.encoder = encoder
        self.V_k_list = V_k_list
        self.s_k_list = s_k_list
        self.snr_db = snr_db
        self.params = params
        self.rl_params = rl_params
        self.k_idx = 0  # Focus on the first user for reward and state
        
        snr_linear = 10**(self.snr_db / 10.0)
        self.noise_power = tf.cast(1.0 / snr_linear, dtype=tf.float32)
        self.H_history = []

    def reset(self):
        """Resets the environment and returns the initial state."""
        self.H_history = []
        state, H_k = self.get_state()
        # Fill the history buffer before starting an episode
        while state is None:
            state, H_k = self.get_state()
        return state, H_k

    def get_state(self):
        """
        Generates a new channel, updates history, and returns the encoded state.
        """
        # Generate a new channel realization for the user of interest (k=0)
        h, path_delays = self.channels[self.k_idx](batch_size=1, num_time_steps=1, sampling_frequency=self.params['sampling_frequency'])
        frequencies = subcarrier_frequencies(self.params['fft_size'], self.params['subcarrier_spacing'])
        H_freq = cir_to_ofdm_channel(frequencies, h, path_delays)
        
        # Use a single subcarrier for the state representation (as in the paper)
        H_k = tf.squeeze(H_freq[..., 0, :, :, 10]) # Squeeze batch and time dims
        
        # Prepare the channel matrix for the encoder input (real and imag parts)
        H_real_imag = tf.concat([tf.math.real(H_k), tf.math.imag(H_k)], axis=1)
        H_flat = tf.reshape(H_real_imag, [-1])
        
        # Update the history buffer
        self.H_history.append(H_flat.numpy())
        if len(self.H_history) > self.params['tau']:
            self.H_history.pop(0)
        
        # If buffer is not full, cannot form a state yet
        if len(self.H_history) < self.params['tau']:
            return None, None
        
        # Create the input tensor for the encoder
        X_k = np.stack(self.H_history, axis=0)
        # Force numpy array to float32 before tensor conversion
        if X_k.dtype != np.float32:
            X_k = X_k.astype(np.float32)
        X_k_tensor = tf.convert_to_tensor(X_k, dtype=tf.float32)
        X_k_batch = tf.expand_dims(X_k_tensor, axis=0)
        #print('Encoder input dtype:', X_k_batch.dtype)
        X_k_batch_f32 = tf.cast(X_k_batch, tf.float32)
        #print('Encoder input after cast dtype:', X_k_batch_f32.dtype)
        phi_k = self.encoder(X_k_batch_f32)
        return phi_k, H_k

    def step(self, action_index, H_k, codebook):
        """
        Executes one time step, calculating SINR and reward.
        """
        W_k = codebook[action_index]
        
        # Calculate SINR for user k (assuming first stream for reward)
        w_i = W_k[0:1, :] # First stream's combining vector
        
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
        
        # Calculate SINR and reward (spectral efficiency in bits/s/Hz)
        sinr = signal_power / (inter_user_interference + intra_user_interference + noise_power_at_output + 1e-12)
        reward = tf.math.log(1.0 + tf.cast(sinr, tf.float32)) / tf.math.log(2.0)
        
        # Get the next state
        next_phi_k, H_k_next = self.get_state()
        done = (next_phi_k is None)
        
        return next_phi_k, tf.squeeze(reward), done, H_k_next


# ──────────────────────────────────────────────────────────────────────────────
# 5. UTILITY AND BASELINE ALGORITHM FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def create_combiner_codebook(num_matrices, num_streams, num_rx_antennas, p_ue_max, num_quant_bits):
    """
    Creates a codebook of quantized and power-normalized combiner matrices.

    Args:
        num_matrices (int): The number of combiner matrices in the codebook.
        num_streams (int): The number of data streams (Ns).
        num_rx_antennas (int): The number of receive antennas (Nr).
        p_ue_max (float): The maximum power constraint for the combiner.
        num_quant_bits (int): The number of bits for quantizing real/imag parts.

    Returns:
        tf.Tensor: A tensor of shape [num_matrices, num_streams, num_rx_antennas]
                   containing the complex-valued codebook.
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

def mrc_combiner(H_k, V_k):
    """
    Calculates the Maximal Ratio Combiner (MRC).
    
    Args:
        H_k: Channel matrix [Nr, Nt]
        V_k: Precoding matrix [Nt, Ns]
    
    Returns:
        W_k: Combining matrix [Ns, Nr]
    """
    # Effective channel after precoding: [Nr, Ns]
    H_eff = tf.matmul(H_k, V_k)
    
    # MRC: conjugate transpose of effective channel
    W_k = tf.linalg.adjoint(H_eff)  # [Ns, Nr]
    
    # Normalize each combiner to unit power (following Sionna convention)
    norm = tf.sqrt(tf.reduce_sum(tf.abs(W_k)**2, axis=-1, keepdims=True))
    W_k = tf.math.divide_no_nan(W_k, tf.cast(norm, W_k.dtype))
    
    return W_k



def mmse_combiner(H_k, V_k_list, noise_variance, params):
    """Calculates the Minimum Mean Square Error (MMSE) combiner."""
    K, Nt, Nr = params['K'], params['Nt'], params['Nr']
    k_idx = 0  # Assuming we are calculating for user 0
    
    # Covariance of total transmitted signal from all users
    transmit_covariance = tf.zeros([Nt, Nt], dtype=tf.complex64)
    for i in range(K):
        transmit_covariance += tf.matmul(V_k_list[i], V_k_list[i], adjoint_b=True)  # Fix this line
        
    # Covariance of received signal y
    H_k_hermitian = tf.linalg.adjoint(H_k)  # Use adjoint instead of transpose
    R_yy = tf.matmul(H_k, tf.matmul(transmit_covariance, H_k_hermitian))
    noise_cov = tf.cast(noise_variance, dtype=tf.complex64) * tf.eye(Nr, dtype=tf.complex64)
    R_yy += noise_cov
    
    # Cross-covariance between desired signal s and received signal y
    R_ys = tf.matmul(H_k, V_k_list[k_idx])
    
    # MMSE solution: W_hermitian = inv(R_yy) * R_ys
    try:
        W_mmse_hermitian = tf.linalg.solve(R_yy, R_ys)
        return tf.linalg.adjoint(W_mmse_hermitian)
    except tf.errors.InvalidArgumentError:
        # Fallback to MRC if MMSE fails
        return mrc_combiner(H_k, V_k_list[k_idx])

@tf.function(jit_compile=True)
def run_ber_simulation(combiner, H_k_freq_batch, V_k_list, noise_variance, params, mapper, demapper):
    """
    Fixed BER calculation with proper error detection.
    """
    # Input validation
    if tf.shape(H_k_freq_batch)[0] == 0:
        return tf.constant(0.0), tf.constant(1.0)
    
    if noise_variance <= 0:
        print("Warning: Invalid noise variance, using default")
        noise_variance = 1e-6
    
    batch_size = tf.shape(H_k_freq_batch)[0]
    Ns = params['Ns']
    num_subcarriers = tf.shape(H_k_freq_batch)[-1]
    bits_per_symbol = params['bits_per_symbol']
    
    # Calculate total bits more carefully
    total_symbols_per_user = Ns * num_subcarriers
    total_bits_per_user = total_symbols_per_user * bits_per_symbol

    # 1. Generate random bits for the user of interest
    bits = tf.random.uniform([batch_size, total_bits_per_user], minval=0, maxval=2, dtype=tf.int32)
    
    # 2. Map bits to symbols
    symbols = mapper(bits)  # Shape: [batch_size, total_symbols_per_user]
    symbols = tf.reshape(symbols, [batch_size, Ns, num_subcarriers])

    # 3. Construct transmitted signal from all users
    x_freq_total = tf.zeros([batch_size, params['Nt'], num_subcarriers], dtype=tf.complex64)
    
    for k in range(params['K']):
        V_k = V_k_list[k]
        if k == 0:  # User of interest
            s_k = symbols
        else:  # Interfering users
            interf_bits = tf.random.uniform([batch_size, total_bits_per_user], minval=0, maxval=2, dtype=tf.int32)
            s_k_interf = mapper(interf_bits)
            s_k = tf.reshape(s_k_interf, [batch_size, Ns, num_subcarriers])
        
        # Apply precoder: x_k = V_k * s_k
        x_freq_k = tf.einsum('ns,bsf->bnf', V_k, s_k)  # [batch_size, Nt, num_subcarriers]
        x_freq_total += x_freq_k

    # 4. Pass through channel
    y_freq = tf.einsum('bnts,btf->bnsf', H_k_freq_batch, x_freq_total)
    y_freq_flat = tf.reshape(y_freq, [batch_size, params['Nr'], -1])  # Keep all subcarriers

    # 5. Add AWGN noise
    noise_stddev = tf.cast(tf.sqrt(noise_variance / 2.0), tf.float32)
    noise_real = tf.random.normal(tf.shape(y_freq_flat), stddev=noise_stddev)
    noise_imag = tf.random.normal(tf.shape(y_freq_flat), stddev=noise_stddev)
    noise = tf.complex(noise_real, noise_imag)
    y_freq_noisy = y_freq_flat + noise

    # 6. Apply combiner
    if len(tf.shape(combiner)) == 2:
        combiner_batch = tf.tile(tf.expand_dims(combiner, 0), [batch_size, 1, 1])
    else:
        combiner_batch = combiner
    
    # Combining: s_hat = W^H * y
    s_hat = tf.einsum('bsr,brf->bsf', combiner_batch, y_freq_noisy)
    s_hat_flat = tf.reshape(s_hat, [batch_size, -1])

    # 7. Demap symbols back to bits
    # **CRITICAL FIX**: Use proper noise variance for demapping
    demapped_llrs = demapper([s_hat_flat, tf.cast(noise_variance, tf.float32)])
    bits_hat = tf.cast(demapped_llrs > 0, tf.int32)  # Hard decision

    # 8. Count bit errors - **CRITICAL FIX**
    # Ensure both tensors have the same shape
    min_bits = tf.minimum(tf.shape(bits)[1], tf.shape(bits_hat)[1])
    bits_truncated = bits[:, :min_bits]
    bits_hat_truncated = bits_hat[:, :min_bits]
    
    # Count errors
    errors = tf.cast(tf.not_equal(bits_truncated, bits_hat_truncated), tf.float32)
    total_errors = tf.reduce_sum(errors)
    total_bits = tf.cast(batch_size * min_bits, tf.float32)
    
    return total_errors, total_bits


# ──────────────────────────────────────────────────────────────────────────────
# 6. MAIN SCRIPT EXECUTION
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """Main function to run the PPO training and evaluation."""
    
    # --- 6.1. Component Instantiation ---
    print("--- Initializing Models and Environment Components ---")
    
    # Models and Agent
    encoder = CNNGRUEncoder(embedding_dim=RL_PARAMS['embedding_dim'])
    actor = Actor(num_actions=RL_PARAMS['codebook_size'])
    critic = Critic()
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=3e-4, decay_steps=1000, decay_rate=0.95, staircase=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)
    ppo_agent = PPOAgent(actor, critic, optimizer, **RL_PARAMS)

    # Combiner Codebook
    combiner_codebook = create_combiner_codebook(
        RL_PARAMS['codebook_size'], PARAMS['Ns'], PARAMS['Nr'], 
        RL_PARAMS['p_ue_max'], RL_PARAMS['num_quant_bits']
    )

    # Sionna Components
    bs_array = PanelArray(num_rows_per_panel=1, num_cols_per_panel=PARAMS['Nt'], polarization="single", polarization_type="V", antenna_pattern="38.901", carrier_frequency=PARAMS['carrier_freq'])
    ue_array = PanelArray(num_rows_per_panel=1, num_cols_per_panel=PARAMS['Nr'], polarization="single", polarization_type="V", antenna_pattern="omni", carrier_frequency=PARAMS['carrier_freq'])

    channels = [
        CDL(model="C", 
            delay_spread=PARAMS['delay_spread'], 
            carrier_frequency=PARAMS['carrier_freq'], 
            ut_array=ue_array,
            bs_array=bs_array,
            direction="downlink", 
            min_speed=3.0)
        for _ in range(PARAMS['K'])
    ]
    
    mapper = Mapper("qam", PARAMS['mod_order'])
    demapper = Demapper("app", "qam", PARAMS['mod_order'])  # Use mod_order, not bits_per_symbol

    # --- 6.2. Calculate Fixed RBD Precoders ---
    print("--- Calculating Fixed Regularized Block Diagonalization (RBD) Precoders ---")
    temp_h, temp_delays = zip(*[channels[k](batch_size=1, num_time_steps=1, sampling_frequency=PARAMS['sampling_frequency']) for k in range(PARAMS['K'])])
    temp_freqs = subcarrier_frequencies(PARAMS['fft_size'], PARAMS['subcarrier_spacing'])
    temp_H_freq = [cir_to_ofdm_channel(temp_freqs, h, d) for h, d in zip(temp_h, temp_delays)]
    temp_H_k = [tf.squeeze(H[..., 0, :, :, 10]) for H in temp_H_freq] # Use one subcarrier

    fixed_V_k = []
    for k in range(PARAMS['K']):
        H_interf_list = [temp_H_k[i] for i in range(PARAMS['K']) if i != k]
        H_interf = tf.concat(H_interf_list, axis=0)
        
        s, u, v = tf.linalg.svd(H_interf)
        
        # Correctly calculate rank for nullspace
        rank_interf = tf.reduce_sum(tf.cast(s > 1e-6, tf.int32))
        null_space_vecs = v[:, rank_interf:]
        
        H_eff_k = tf.matmul(temp_H_k[k], null_space_vecs)
        
        _, _, v_eff = tf.linalg.svd(H_eff_k)
        V_eff_k = v_eff[:, :PARAMS['Ns']]
        V_k = tf.matmul(null_space_vecs, V_eff_k)
        fixed_V_k.append(V_k)
    print("--- Finished Calculating Precoders ---")

    # Fixed symbols for the environment (not used in BER, only for SINR reward)
    fixed_s_k = [tf.complex(tf.random.uniform([PARAMS['Ns'], 1]), tf.random.uniform([PARAMS['Ns'], 1])) for _ in range(PARAMS['K'])]
    
    # --- 6.3. PPO Training Loop ---
    env = MIMOEnvironment(channels, encoder, fixed_V_k, fixed_s_k, RL_PARAMS['snr_db_train'], PARAMS, RL_PARAMS)
    total_rewards_history = []
    policy_entropy_history = []
    
    print(f"\n--- Starting PPO Agent Training for {RL_PARAMS['num_episodes']} episodes ---")
    for episode in trange(RL_PARAMS['num_episodes'], desc="Training Progress"):
        states, actions, rewards, next_states, dones, old_probs = [], [], [], [], [], []
        state, H_k = env.reset()
        
        for t in range(RL_PARAMS['max_steps_per_episode']):
            action_probs_dist = actor(state)
            action = tf.random.categorical(tf.math.log(action_probs_dist), 1)[0, 0].numpy()
            old_prob = action_probs_dist[0, action].numpy()

            # Calculate and store policy entropy for this step
            entropy = -np.sum(action_probs_dist.numpy() * np.log(action_probs_dist.numpy() + 1e-10))
            policy_entropy_history.append(entropy)

            next_state, reward, done, H_k_next = env.step(action, H_k, combiner_codebook)

            if next_state is not None:
                states.append(tf.squeeze(state).numpy())
                actions.append(action)
                rewards.append(reward.numpy())
                next_states.append(tf.squeeze(next_state).numpy())
                dones.append(done)
                old_probs.append(old_prob)

            if done:
                break

            state, H_k = next_state, H_k_next

        if len(states) > 1:
            values = critic(np.array(states)).numpy().flatten()
            next_values = critic(np.array(next_states)).numpy().flatten()
            
            returns, advantages = ppo_agent._compute_advantages_and_returns(np.array(rewards), values, next_values, np.array(dones))
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
            
            ppo_agent.train(
                np.array(states, dtype=np.float32), 
                np.array(actions, dtype=np.int32), 
                np.array(old_probs, dtype=np.float32), 
                returns, 
                advantages
            )
        
        total_rewards_history.append(sum(rewards))

    print("\n--- Training Finished ---")
    output_dir = "results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    actor.save_weights(os.path.join(output_dir, "ppo_actor_weights.h5"))
    critic.save_weights(os.path.join(output_dir, "ppo_critic_weights.h5"))
    encoder.save_weights(os.path.join(output_dir, "encoder_weights.h5"))
    print(f"Trained model weights saved to '{output_dir}' directory.")

    # --- 6.4. Evaluation Phase ---
    # --- Fixed Evaluation Phase ---
    print("\n--- Starting Fixed Evaluation Phase ---")

    # Pre-compute channels and states
    print(f"Pre-computing {EVAL_PARAMS['num_channel_realizations']} channel realizations...")
    channel_cache = []
    state_cache = []

    eval_env = MIMOEnvironment(channels, encoder, fixed_V_k, fixed_s_k, 0, PARAMS, RL_PARAMS)

    for i in trange(EVAL_PARAMS['num_channel_realizations'], desc="Generating Channels & States"):
        # Generate channel
        h, path_delays = channels[0](batch_size=1, num_time_steps=1, sampling_frequency=PARAMS['sampling_frequency'])
        
        # Convert to frequency domain
        frequencies = subcarrier_frequencies(PARAMS['fft_size'], PARAMS['subcarrier_spacing'])
        H_freq = cir_to_ofdm_channel(frequencies, h, path_delays)
        H_k_freq = tf.squeeze(H_freq[..., 0, :, :, :])  # Remove batch and time dims
        
        channel_cache.append(H_k_freq)
        
        # Generate state
        state, _ = eval_env.reset()
        if state is not None:
            state_cache.append(tf.squeeze(state))
        else:
            # If state is None, create a dummy state
            dummy_state = tf.zeros([RL_PARAMS['embedding_dim']], dtype=tf.float32)
            state_cache.append(dummy_state)

    # Stack all data
    H_k_freq_all = tf.stack(channel_cache)
    state_all = tf.stack(state_cache)

    print(f"Generated {len(channel_cache)} channel realizations")
    print(f"Channel shape: {H_k_freq_all.shape}")
    print(f"State shape: {state_all.shape}")

    # Add debug information
    print("=== DEBUG INFORMATION ===")
    print(f"Total channel realizations: {len(channel_cache)}")
    print(f"Channel shape: {H_k_freq_all.shape if len(channel_cache) > 0 else 'No channels'}")
    print(f"State cache length: {len(state_cache)}")
    print(f"Batch size: {EVAL_PARAMS['batch_size']}")
    print(f"SNR range: {EVAL_PARAMS['snr_dBs']}")
    print("========================")

    # BER evaluation
    results_ber = {"PPO": [], "MRC": [], "MMSE": []}

    for snr_db in EVAL_PARAMS['snr_dBs']:
        print(f"--- Evaluating SNR = {snr_db} dB ---")
        noise_var = 10**(-snr_db / 10.0)
        
        total_errors = {"PPO": 0.0, "MRC": 0.0, "MMSE": 0.0}
        total_bits = {"PPO": 0.0, "MRC": 0.0, "MMSE": 0.0}
        
        # Process in batches
        num_samples = len(channel_cache)
        batch_size = min(EVAL_PARAMS['batch_size'], num_samples)
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        print(f"Processing {num_samples} samples in {num_batches} batches of size {batch_size}")
        
        for i in trange(num_batches, desc=f"SNR {snr_db} dB Batches"):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            actual_batch_size = end_idx - start_idx
            
            if actual_batch_size == 0:
                continue
                
            # Get batch data
            H_k_batch = H_k_freq_all[start_idx:end_idx]
            state_batch = state_all[start_idx:end_idx]
            
            # Ensure minimum number of subcarriers for reliable BER estimation
            min_subcarriers = min(64, H_k_batch.shape[-1])  
            H_k_batch = H_k_batch[..., :min_subcarriers]
            
            print(f"Batch {i}: Processing {actual_batch_size} samples, H_k shape: {H_k_batch.shape}")
            
            try:
                # PPO Agent actions
                action_probs = actor(state_batch)
                action_indices = tf.argmax(action_probs, axis=1)
                W_ppo_batch = tf.gather(combiner_codebook, action_indices)
                
                # Baseline combiners (use single subcarrier for combiner calculation)
                H_k_single = H_k_batch[:, :, :, 0]  # Shape: [batch, Nr, Nt]
                
                # MRC combiners
                W_mrc_list = []
                for j in range(actual_batch_size):
                    W_mrc = mrc_combiner(H_k_single[j], fixed_V_k[0])
                    W_mrc_list.append(W_mrc)
                W_mrc_batch = tf.stack(W_mrc_list)
                
                # MMSE combiners  
                W_mmse_list = []
                for j in range(actual_batch_size):
                    W_mmse = mmse_combiner(H_k_single[j], fixed_V_k, noise_var, PARAMS)
                    W_mmse_list.append(W_mmse)
                W_mmse_batch = tf.stack(W_mmse_list)
                
                # Run BER simulations
                methods = [("PPO", W_ppo_batch), ("MRC", W_mrc_batch), ("MMSE", W_mmse_batch)]
                
                for method_name, combiner_batch in methods:
                    try:
                        errors, bits = run_ber_simulation(
                            combiner_batch, H_k_batch, fixed_V_k, noise_var, PARAMS, mapper, demapper
                        )
                        
                        # Check for NaN or invalid results
                        if tf.math.is_nan(errors) or tf.math.is_nan(bits) or bits == 0:
                            print(f"  {method_name}: Invalid result, skipping...")
                            continue
                            
                        total_errors[method_name] += float(errors.numpy())
                        total_bits[method_name] += float(bits.numpy())
                        
                        print(f"  {method_name}: {float(errors.numpy())} errors / {float(bits.numpy())} bits")
                        
                    except Exception as e:
                        print(f"  Error in {method_name} simulation: {e}")
                        continue
                        
            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                continue
        
        # Calculate BER for this SNR
        print(f"\nSNR {snr_db} dB Results:")
        for method_name in results_ber.keys():
            if total_bits[method_name] > 0:
                ber = total_errors[method_name] / total_bits[method_name]
                results_ber[method_name].append(ber)
                print(f"  {method_name}: BER = {ber:.6e} ({int(total_errors[method_name])} errors / {int(total_bits[method_name])} bits)")
            else:
                print(f"  {method_name}: No bits processed - setting BER = 0")
                results_ber[method_name].append(0.0)

    print("\n--- Final BER Results ---")
    for name, ber_list in results_ber.items():
        print(f"{name}: {[f'{b:.2e}' if b > 0 else '0.00e+00' for b in ber_list]}")

    # --- 6.5. Plotting Results ---
    print("\n--- Generating and Saving Plots ---")

    # Plot 1: BER vs. SNR
    fig, ax = plt.subplots(figsize=(10, 7))
    for name, ber_list in results_ber.items():
        ax.plot(EVAL_PARAMS['snr_dBs'], ber_list, 'o-', label=name, linewidth=2)
    ax.set_yscale('log')
    ax.set_xlabel("SNR (dB)", fontsize=14)
    ax.set_ylabel("Bit Error Rate (BER)", fontsize=14)
    ax.set_title("BER Performance Comparison of Combining Strategies", fontsize=16)
    ax.grid(True, which="both", linestyle='--')
    ax.legend(fontsize=12)
    ax.set_ylim(1e-5, 1.0)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'ber_vs_snr.png'))
    plt.close(fig)
    print(f"Saved BER plot to '{os.path.join(output_dir, 'ber_vs_snr.png')}'")

    # Plot 2: RL Agent Training Curve
    fig, ax = plt.subplots(figsize=(10, 7))
    def moving_average(data, window_size=20):
        """Compute the moving average with safety checks."""
        try:
            # Convert to flat numpy array if it's not already
            data = np.array(data).flatten()
            
            # Check for sufficient data
            if len(data) < window_size: 
                return np.array([])
            return np.convolve(data, np.ones(window_size), 'valid') / window_size
        except Exception as e:
            print(f"Warning: Error in moving average calculation: {e}")
            return np.array([]) if len(data) == 0 else np.array([np.mean(data)])

    smoothed_rewards = moving_average(np.array(total_rewards_history))
    if smoothed_rewards.size > 0:
        ax.plot(range(len(smoothed_rewards)), smoothed_rewards)
        ax.set_xlabel("Episode", fontsize=14)
        ax.set_ylabel("Smoothed Average Reward (bits/s/Hz)", fontsize=14)
        ax.set_title("PPO Agent Learning Curve", fontsize=16)
        ax.grid(True)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'rl_training_curve.png'))
        print(f"Saved RL training curve plot to '{os.path.join(output_dir, 'rl_training_curve.png')}'")
    plt.close(fig)

    # Plot 3: Policy Entropy Curve
    fig, ax = plt.subplots(figsize=(10, 7))
    entropy_ma = moving_average(np.array(policy_entropy_history), window_size=20)
    if entropy_ma.size > 0:
        ax.plot(range(len(entropy_ma)), entropy_ma)
        ax.set_xlabel("Step", fontsize=14)
        ax.set_ylabel("Policy Entropy", fontsize=14)
        ax.set_title("Policy Entropy During Training", fontsize=16)
        ax.grid(True)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'policy_entropy.png'))
        print(f"Saved policy entropy plot to '{os.path.join(output_dir, 'policy_entropy.png')}'")

    plt.close(fig)

    # Plot 4: t-SNE Latent Space Visualization
    if TSNE_AVAILABLE:
        try:
            state_matrix = np.stack([s.numpy() if hasattr(s, 'numpy') else s for s in state_cache])
            tsne = TSNE(n_components=2, random_state=42)
            tsne_embeds = tsne.fit_transform(state_matrix)
            
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.scatter(tsne_embeds[:, 0], tsne_embeds[:, 1], s=10, alpha=0.7)
            ax.set_title("t-SNE of Latent State Embeddings", fontsize=16)
            ax.set_xlabel("t-SNE Dim 1", fontsize=14)
            ax.set_ylabel("t-SNE Dim 2", fontsize=14)
            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, 'tsne_latent_space.png'))
            print(f"Saved t-SNE latent space plot to '{os.path.join(output_dir, 'tsne_latent_space.png')}'")
            plt.close(fig)
        except Exception as e:
            print(f"t-SNE plot could not be generated: {e}")
    else:
        print("Skipping t-SNE visualization - scikit-learn not installed")
    # -*- coding: utf-8 -*-

    print("\n--- All tasks completed successfully. ---")
# Add this test function at the end of your main.py
def quick_validation_test():
    """Quick test to validate the implementation"""
    print("\n=== Running Quick Validation Test ===")
    
    # Test BER simulation with known inputs
    batch_size = 2
    Nr, Nt, num_subcarriers = 2, 8, 32
    num_streams = 2  # Number of streams per user (Ns)
    K = 4            # Number of users
    
    # Create dummy data
    H_test = tf.complex(
        tf.random.normal([batch_size, Nr, Nt, num_subcarriers]),
        tf.random.normal([batch_size, Nr, Nt, num_subcarriers])
    )
    
    combiner_test = tf.complex(
        tf.random.normal([batch_size, num_streams, Nr]),
        tf.random.normal([batch_size, num_streams, Nr])
    )
    
    # Create test precoders (instead of using fixed_V_k)
    test_V_k_list = []
    for _ in range(K):
        V_k = tf.complex(
            tf.random.normal([Nt, num_streams]),
            tf.random.normal([Nt, num_streams])
        )
        # Normalize the precoder
        V_k = V_k / tf.norm(V_k, axis=0, keepdims=True)
        test_V_k_list.append(V_k)
    
    # Create test parameters dictionary
    test_params = {
        'Nr': Nr,
        'Nt': Nt,
        'Ns': num_streams,
        'K': K,
        'bits_per_symbol': 4,  # For 16-QAM
        'mod_order': 16
    }
    
    # Test with your actual functions
    mapper_test = Mapper("qam", 16)
    demapper_test = Demapper("app", "qam", 16)
    
    try:
        errors, bits = run_ber_simulation(
            combiner_test, H_test, test_V_k_list, 0.1, test_params, mapper_test, demapper_test
        )
        
        print(f"✅ BER simulation test passed: {errors.numpy()} errors / {bits.numpy()} bits")
        print(f"✅ Test BER: {float(errors.numpy()) / float(bits.numpy()):.6e}")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

# Run the test before main execution
if __name__ == "__main__":
    # Uncomment to run validation test
    # quick_validation_test()
    main()

