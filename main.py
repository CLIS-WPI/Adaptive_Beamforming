# =========================================================================================
#                   Official Implementation for the paper:
#   "Adaptive Beamforming for Interference-Limited MU-MIMO using Spatio-Temporal
#                           Policy Networks"
#
#   Version 7: Final stable version for H100 GPUs.
#   - Replaced the problematic LSTM/GRU layers with a fully convolutional architecture
#     to definitively resolve all XLA and cuDNN compatibility issues.
#   - Retained all other H100-specific optimizations.
# =========================================================================================
import os
import time
import gc

# --- 1. Initial Setup and GPU Configuration ---
# Set environment variables to control GPU visibility
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Make only GPU 0 visible

import tensorflow as tf
import sionna
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from scipy.special import erfc

# Check for scikit-learn for t-SNE plotting
try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn not installed. t-SNE visualization will be disabled.")
    TSNE_AVAILABLE = False

# Import required Sionna modules
try:
    from sionna.phy.channel.tr38901 import CDL, PanelArray
    from sionna.phy.channel import subcarrier_frequencies, cir_to_ofdm_channel
    from sionna.phy.mapping import Mapper, Demapper
    from sionna.phy.mimo import lmmse_equalizer
except ImportError as e:
    print("A required Sionna module was not found. Please ensure Sionna version >= 0.19 is installed.")
    raise e

def configure_tensorflow(use_mixed_precision=True, enable_jit=True):
    """Advanced TensorFlow configuration for optimal GPU usage."""
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("No GPU found. Running on CPU.")
        return

    print(f"Found {len(gpus)} GPUs. Using {gpus[0].name}")
    try:
        # Enable dynamic memory growth to avoid allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        # Enable mixed precision for performance boost on modern GPUs like H100
        if use_mixed_precision:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("Mixed precision (float16) enabled.")

        # Enable XLA/JIT compilation to optimize the computation graph
        if enable_jit:
            tf.config.optimizer.set_jit(True)
            print("XLA JIT compilation enabled.")

    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

# Apply the configuration
configure_tensorflow()

# Set seed for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

def manage_memory():
    """Aggressively clear memory to prevent OOM errors."""
    tf.keras.backend.clear_session()
    gc.collect()

# --- 2. Simulation and Model Parameters (Tuned for H100) ---
# Parameters are increased to better utilize H100's capacity for a more realistic simulation
PARAMS = {
    'Nt': 8, 'Nr': 4, 'K': 4, 'Ns': 2, 'tau': 8,
    'fft_size': 64, 'num_ofdm_symbols': 14,
    'subcarrier_spacing': 30e3, 'cp_length': 5,
    'mod_order': 16, 'carrier_freq': 3.5e9, 'delay_spread': 30e-9,
}
PARAMS['bits_per_symbol'] = int(np.log2(PARAMS['mod_order']))
PARAMS['sampling_frequency'] = PARAMS['fft_size'] * PARAMS['subcarrier_spacing']

# RL parameters are adjusted for more extensive and stable training
RL_PARAMS = {
    'embedding_dim': 128,
    'codebook_size': 64,
    'p_ue_max': 1.0, 'num_quant_bits': 4,
    'num_epochs': 50,  # Increased training duration
    'batch_size': 128, # Larger batch size for stable gradients
    'dataset_size': 4096, # Larger offline dataset
    'snr_db_train': 20.0, # Train at a higher SNR for better feature learning
    'gamma': 0.99, 'lambda_gae': 0.95,
    'epsilon_clip': 0.2, 'value_coeff': 0.5, 'entropy_coeff': 0.01,
}

EVAL_PARAMS = {
    'snr_dBs': np.arange(-10, 21, 5),
    'num_channel_realizations': 100,
    'batch_size': 32,
}


# --- 3. Model Definitions ---

class CNNEncoder(tf.keras.Model):
    """Fully Convolutional encoder to extract spatio-temporal features."""
    def __init__(self, embedding_dim):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')
        self.bn2 = tf.keras.layers.BatchNormalization()
        # CRITICAL FIX: Replaced the problematic recurrent layers (GRU/LSTM)
        # with a fully convolutional approach. This is highly compatible with XLA/cuDNN.
        self.conv3 = tf.keras.layers.Conv1D(filters=256, kernel_size=3, padding='same', activation='relu')
        self.bn3 = tf.keras.layers.BatchNormalization()
        # Global pooling collapses the time dimension to create a fixed-size embedding.
        self.global_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.out = tf.keras.layers.Dense(embedding_dim)

    # JIT can be safely re-enabled as all layers are now compatible.
    @tf.function(jit_compile=True)
    def call(self, inputs, training=None):
        # Input shape: (batch, tau, features)
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.global_pool(x)
        return self.out(x)

class Actor(tf.keras.Model):
    """Actor (Policy) network to select an action (combiner matrix)."""
    def __init__(self, num_actions):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.logits = tf.keras.layers.Dense(num_actions, dtype='float32') # Use float32 for numerical stability

    @tf.function(jit_compile=True)
    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        logits = self.logits(x)
        return tf.nn.softmax(logits)

class Critic(tf.keras.Model):
    """Critic (Value Function) network to estimate the state value."""
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.value = tf.keras.layers.Dense(1, dtype='float32')

    @tf.function(jit_compile=True)
    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        return self.value(x)

# --- 4. Utility Functions and Baseline Algorithms ---

def create_combiner_codebook(num_matrices, num_streams, num_rx_antennas):
    """Creates a codebook of power-normalized combiner matrices."""
    real = tf.random.normal([num_matrices, num_streams, num_rx_antennas], dtype=tf.float32)
    imag = tf.random.normal([num_matrices, num_streams, num_rx_antennas], dtype=tf.float32)
    W = tf.complex(real, imag)
    norm = tf.norm(W, ord='fro', axis=(-2, -1), keepdims=True)
    W_normalized = W / tf.cast(norm, tf.complex64)
    return W_normalized

def calculate_sinr(H_k, V_k_list, W, noise_power, params):
    """Calculates SINR for a given channel and combiner."""
    k_idx = 0 # Focus on the first user
    Ns, K, Nr, Nt = params['Ns'], params['K'], params['Nr'], params['Nt']
    
    H_k_reshaped = tf.reshape(H_k, (Nr, Nt))
    
    # Desired signal term
    signal_term = W @ H_k_reshaped @ V_k_list[k_idx]
    signal_power = tf.reduce_sum(tf.square(tf.abs(signal_term)))

    # Inter-User Interference term
    iui_power = 0.0
    for l in range(K):
        if l != k_idx:
            iui_term = W @ H_k_reshaped @ V_k_list[l]
            iui_power += tf.reduce_sum(tf.square(tf.abs(iui_term)))

    # Noise term
    noise_out_power = noise_power * tf.reduce_sum(tf.square(tf.abs(W)))

    sinr = signal_power / (iui_power + noise_out_power + 1e-12)
    return sinr

def mrc_combiner(H_k, V_k):
    """Calculates the Maximal Ratio Combiner (MRC)."""
    H_eff = H_k @ V_k
    W_k = tf.linalg.adjoint(H_eff)
    norm = tf.norm(W_k, axis=-1, keepdims=True)
    return tf.math.divide_no_nan(W_k, tf.cast(norm, W_k.dtype))

def mmse_combiner(H_k, V_k_list, noise_variance, params):
    """Calculates the Minimum Mean Square Error (MMSE) combiner."""
    y_k = H_k @ V_k_list[0]
    R_yy_inv = tf.linalg.inv(y_k @ tf.linalg.adjoint(y_k) + tf.cast(noise_variance, tf.complex64) * tf.eye(params['Nr'], dtype=tf.complex64))
    W_mmse = tf.linalg.adjoint(y_k) @ R_yy_inv
    return W_mmse


# --- 5. Offline Dataset Generation and Training ---

def generate_dataset(channel_model, params, dataset_size):
    """Generates an offline dataset of channel histories and corresponding channel matrices."""
    print(f"--- Generating offline dataset of size {dataset_size}... ---")
    histories = []
    channels_H = []
    
    H_history_flat = np.zeros((params['tau'], params['Nr'] * params['Nt'] * 2), dtype=np.float32)

    for _ in trange(dataset_size, desc="Generating Data"):
        h, delays = channel_model(batch_size=1, num_time_steps=1, sampling_frequency=params['sampling_frequency'])
        freqs = subcarrier_frequencies(params['fft_size'], params['subcarrier_spacing'])
        H_freq = cir_to_ofdm_channel(freqs, h, delays)
        
        center_sc = H_freq.shape[-1] // 2
        H_k = H_freq[0, 0, :, 0, :, 0, center_sc]
        channels_H.append(H_k.numpy())

        H_flat = tf.reshape(tf.concat([tf.math.real(H_k), tf.math.imag(H_k)], axis=1), [-1])
        H_history_flat = np.roll(H_history_flat, -1, axis=0)
        H_history_flat[-1, :] = H_flat.numpy()
        histories.append(H_history_flat.copy())
        
    print(f"--- Dataset with {len(histories)} samples generated. ---")
    return np.array(histories, dtype=np.float32), np.array(channels_H, dtype=np.complex64)


def precompute_rewards(channels_H, combiner_codebook, V_k_list, noise_power, params):
    """Pre-computes the reward for all state-action pairs to speed up training."""
    print("--- Pre-computing rewards for all state-action pairs... ---")
    num_states = channels_H.shape[0]
    num_actions = combiner_codebook.shape[0]
    reward_table = np.zeros((num_states, num_actions), dtype=np.float32)

    for i in trange(num_states, desc="Pre-computing Rewards"):
        for j in range(num_actions):
            sinr = calculate_sinr(channels_H[i], V_k_list, combiner_codebook[j], noise_power, params)
            reward = np.log2(1 + sinr.numpy().real)
            reward_table[i, j] = reward
            
    print("--- Reward computation finished. ---")
    return reward_table

# --- 6. Main Program Execution ---

def main():
    """Main function to run the training and evaluation."""
    manage_memory()
    
    print("--- Initializing Models and Components ---")
    encoder = CNNEncoder(embedding_dim=RL_PARAMS['embedding_dim']) # Using the new All-CNN encoder
    actor = Actor(num_actions=RL_PARAMS['codebook_size'])
    critic = Critic()
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=3e-5,
        decay_steps=1000,
        decay_rate=0.9)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    combiner_codebook = create_combiner_codebook(RL_PARAMS['codebook_size'], PARAMS['Ns'], PARAMS['Nr'])

    bs_array = PanelArray(num_rows_per_panel=1, num_cols_per_panel=int(PARAMS['Nt']/2), polarization="dual", polarization_type="VH", antenna_pattern="38.901", carrier_frequency=PARAMS['carrier_freq'])
    ue_array = PanelArray(num_rows_per_panel=1, num_cols_per_panel=PARAMS['Nr'], polarization="single", polarization_type="V", antenna_pattern="omni", carrier_frequency=PARAMS['carrier_freq'])
    channel_model = CDL(model="C", delay_spread=PARAMS['delay_spread'], carrier_frequency=PARAMS['carrier_freq'], ut_array=ue_array, bs_array=bs_array, direction="downlink", min_speed=3.0)
    
    h_sample, d_sample = channel_model(1,1,PARAMS['sampling_frequency'])
    freqs_sample = subcarrier_frequencies(PARAMS['fft_size'], PARAMS['subcarrier_spacing'])
    H_freq_sample = cir_to_ofdm_channel(freqs_sample, h_sample, d_sample)
    H_k_sample = [tf.squeeze(H_freq_sample[..., PARAMS['fft_size']//2]) for _ in range(PARAMS['K'])]
    
    fixed_V_k = []
    for h in H_k_sample:
        _, _, v = tf.linalg.svd(h)
        fixed_V_k.append(v[:, :PARAMS['Ns']])

    # --- Step 1: Generate Offline Dataset ---
    histories_data, channels_data = generate_dataset(channel_model, PARAMS, RL_PARAMS['dataset_size'])
    
    noise_power_train = 10**(-RL_PARAMS['snr_db_train'] / 10.0)
    reward_table = precompute_rewards(channels_data, combiner_codebook, fixed_V_k, noise_power_train, PARAMS)
    
    manage_memory()

    # --- Step 2: PPO Training with Offline Dataset ---
    print(f"\n--- Starting PPO Agent Training for {RL_PARAMS['num_epochs']} epochs ---")
    total_rewards_history = []
    
    dataset = tf.data.Dataset.from_tensor_slices((histories_data, np.arange(RL_PARAMS['dataset_size']))).shuffle(RL_PARAMS['dataset_size']).batch(RL_PARAMS['batch_size'])

    for epoch in range(RL_PARAMS['num_epochs']):
        epoch_rewards = []
        for batch_histories, batch_indices in dataset:
            
            # Get action probabilities from the policy before the update (for the ratio)
            old_probs_dist = actor(encoder(batch_histories, training=False), training=False)

            with tf.GradientTape() as tape:
                # The forward pass must be inside the tape for gradients to be recorded
                batch_states = encoder(batch_histories, training=True)
                new_probs_dist = actor(batch_states, training=True)
                values = critic(batch_states, training=True)
                
                # Sample actions and get their rewards
                action_indices = tf.random.categorical(tf.math.log(new_probs_dist), 1)[:, 0]
                rewards = tf.gather(tf.gather(reward_table, batch_indices), action_indices, batch_dims=1)
                epoch_rewards.extend(rewards.numpy())

                # Calculate advantages
                advantages = rewards - tf.squeeze(values)
                
                # Implement the PPO Clipped Surrogate Objective
                old_probs_of_actions = tf.gather(old_probs_dist, action_indices, batch_dims=1)
                new_probs_of_actions = tf.gather(new_probs_dist, action_indices, batch_dims=1)

                ratio = new_probs_of_actions / (old_probs_of_actions + 1e-10)
                surrogate1 = ratio * advantages
                surrogate2 = tf.clip_by_value(ratio, 1.0 - RL_PARAMS['epsilon_clip'], 1.0 + RL_PARAMS['epsilon_clip']) * advantages
                
                actor_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
                critic_loss = tf.reduce_mean(tf.square(advantages))
                entropy = -tf.reduce_mean(tf.reduce_sum(new_probs_dist * tf.math.log(new_probs_dist + 1e-10), axis=1))
                
                total_loss = actor_loss + RL_PARAMS['value_coeff'] * critic_loss - RL_PARAMS['entropy_coeff'] * entropy

            # Apply gradients to all trainable variables
            all_vars = encoder.trainable_variables + actor.trainable_variables + critic.trainable_variables
            gradients = tape.gradient(total_loss, all_vars)
            optimizer.apply_gradients(zip(gradients, all_vars))
        
        avg_reward = np.mean(epoch_rewards)
        total_rewards_history.append(avg_reward)
        print(f"Epoch {epoch+1}/{RL_PARAMS['num_epochs']} finished. Average Reward: {avg_reward:.4f}")

    print("\n--- Training Finished ---")
    output_dir = "results_v7"
    os.makedirs(output_dir, exist_ok=True)
    actor.save_weights(os.path.join(output_dir, "ppo_actor_weights.h5"))
    encoder.save_weights(os.path.join(output_dir, "encoder_weights.h5"))
    
    manage_memory()

    # --- Step 3: Evaluation and Plotting ---
    print("\n--- Starting Evaluation Phase ---")
    results_ber = {"PPO": [], "MRC": [], "MMSE": []}
    
    eval_histories, eval_channels = generate_dataset(channel_model, PARAMS, EVAL_PARAMS['num_channel_realizations'])
    eval_states = encoder(eval_histories, training=False)

    for snr_db in EVAL_PARAMS['snr_dBs']:
        print(f"--- Evaluating SNR = {snr_db} dB ---")
        noise_var = 10**(-snr_db / 10.0)
        
        action_probs = actor(eval_states, training=False)
        action_indices = tf.argmax(action_probs, axis=1)
        W_ppo = tf.gather(combiner_codebook, action_indices)
        
        W_mrc = tf.stack([mrc_combiner(h, fixed_V_k[0]) for h in eval_channels])
        W_mmse = tf.stack([mmse_combiner(h, fixed_V_k, noise_var, PARAMS) for h in eval_channels])
        
        for name, W_batch in [("PPO", W_ppo), ("MRC", W_mrc), ("MMSE", W_mmse)]:
            sinrs = [calculate_sinr(eval_channels[j], fixed_V_k, W_batch[j], noise_var, PARAMS) for j in range(len(eval_channels))]
            avg_sinr = np.mean([s.numpy().real for s in sinrs])
            # More accurate BER approximation for 16-QAM
            ber = (3.0/8.0) * erfc(np.sqrt( (2.0/5.0) * avg_sinr ))
            results_ber[name].append(max(ber, 1e-6))
            
        print(f"  PPO BER: {results_ber['PPO'][-1]:.2e}, MRC BER: {results_ber['MRC'][-1]:.2e}, MMSE BER: {results_ber['MMSE'][-1]:.2e}")

    # --- Generate and Save Plots ---
    print("\n--- Generating and Saving Plots ---")
    
    plt.figure(figsize=(10, 7))
    for name, ber_list in results_ber.items():
        plt.semilogy(EVAL_PARAMS['snr_dBs'], ber_list, 'o-', label=name, linewidth=2)
    plt.xlabel("SNR (dB)", fontsize=14); plt.ylabel("Bit Error Rate (BER)", fontsize=14)
    plt.title("BER Performance Comparison", fontsize=16); plt.grid(True, which="both", linestyle='--')
    plt.legend(fontsize=12); plt.ylim(1e-6, 1.0); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ber_vs_snr.png'))
    print(f"Saved BER plot to '{os.path.join(output_dir, 'ber_vs_snr.png')}'")
    plt.close()

    plt.figure(figsize=(10, 7))
    plt.plot(total_rewards_history)
    plt.xlabel("Epoch", fontsize=14); plt.ylabel("Average Reward (bits/s/Hz)", fontsize=14)
    plt.title("PPO Agent Learning Curve", fontsize=16); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rl_training_curve.png'))
    print(f"Saved RL training curve plot to '{os.path.join(output_dir, 'rl_training_curve.png')}'")
    plt.close()

    if TSNE_AVAILABLE and len(eval_states) > 50:
        try:
            tsne = TSNE(n_components=2, random_state=SEED, perplexity=min(30, len(eval_states)-1))
            tsne_embeds = tsne.fit_transform(eval_states)
            
            plt.figure(figsize=(10, 7))
            plt.scatter(tsne_embeds[:, 0], tsne_embeds[:, 1], alpha=0.5)
            plt.title("t-SNE of Latent State Embeddings", fontsize=16)
            plt.xlabel("t-SNE Component 1"); plt.ylabel("t-SNE Component 2")
            plt.grid(True); plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'tsne_latent_space.png'))
            print(f"Saved t-SNE plot to '{os.path.join(output_dir, 'tsne_latent_space.png')}'")
            plt.close()
        except Exception as e:
            print(f"Could not generate t-SNE plot: {e}")

    print("\n--- All tasks completed successfully. ---")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"\nAn unexpected error occurred during main execution: {e}")
        traceback.print_exc()
