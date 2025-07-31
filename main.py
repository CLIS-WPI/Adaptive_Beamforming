# =========================================================================================
#                   Official Implementation for the paper:
#   "Adaptive Beamforming for Interference-Limited MU-MIMO using Spatio-Temporal
#                           Policy Networks"
#
#   Version 12: Final Stable Multi-GPU Version.
#   - Fixed the distributed loss calculation by setting the reduction to NONE, as required
#     by tf.distribute.Strategy for custom training loops.
# =========================================================================================
import os
import time
import gc

# --- 1. Initial Setup and GPU Configuration ---
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import tensorflow as tf
import sionna
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from scipy.special import erfc

try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn not installed. t-SNE visualization will be disabled.")
    TSNE_AVAILABLE = False

try:
    from sionna.phy.channel.tr38901 import CDL, PanelArray
    from sionna.phy.channel import subcarrier_frequencies, cir_to_ofdm_channel
    from sionna.phy.mapping import Mapper, Demapper
except ImportError as e:
    print("A required Sionna module was not found. Please ensure Sionna version >= 0.19 is installed.")
    raise e

def configure_tensorflow(use_mixed_precision=True, enable_jit=True):
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("No GPU found. Running on CPU.")
        return
    print(f"Found {len(gpus)} GPUs.")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if use_mixed_precision:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("Mixed precision (float16) enabled.")
        if enable_jit:
            tf.config.optimizer.set_jit(True)
            print("XLA JIT compilation enabled.")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

configure_tensorflow()

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

def manage_memory():
    tf.keras.backend.clear_session()
    gc.collect()

# --- 2. Simulation and Model Parameters (Tuned for Dual H100) ---
PARAMS = {
    'Nt': 8, 'Nr': 4, 'K': 4, 'Ns': 2, 'tau': 8,
    'fft_size': 64, 'num_ofdm_symbols': 14,
    'subcarrier_spacing': 30e3, 'cp_length': 5,
    'mod_order': 16, 'carrier_freq': 3.5e9, 'delay_spread': 30e-9,
}
PARAMS['bits_per_symbol'] = int(np.log2(PARAMS['mod_order']))
PARAMS['sampling_frequency'] = PARAMS['fft_size'] * PARAMS['subcarrier_spacing']

SL_PARAMS = {
    'embedding_dim': 128,
    'codebook_size': 64,
    'num_epochs': 30,
    'batch_size': 256,
    'dataset_size': 8192,
    'snr_db_train': 20.0,
}

EVAL_PARAMS = {
    'snr_dBs': np.arange(-10, 21, 5),
    'num_channel_realizations': 100,
}

# --- 3. Model Definitions ---
class BeamformingClassifier(tf.keras.Model):
    """A single model that takes channel history and classifies the best beamformer."""
    def __init__(self, embedding_dim, num_actions):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv1D(filters=256, kernel_size=3, padding='same', activation='relu')
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.global_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.embedding_layer = tf.keras.layers.Dense(embedding_dim, activation='relu')
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.logits = tf.keras.layers.Dense(num_actions, dtype='float32')

    @tf.function(jit_compile=True)
    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.global_pool(x)
        embedding = self.embedding_layer(x)
        x = self.dense1(embedding)
        x = self.dense2(x)
        return self.logits(x)

# --- 4. Utility Functions and Baseline Algorithms ---
def create_combiner_codebook(num_matrices, num_streams, num_rx_antennas):
    real = tf.random.normal([num_matrices, num_streams, num_rx_antennas], dtype=tf.float32)
    imag = tf.random.normal([num_matrices, num_streams, num_rx_antennas], dtype=tf.float32)
    W = tf.complex(real, imag)
    norm = tf.norm(W, ord='fro', axis=(-2, -1), keepdims=True)
    return W / tf.cast(norm, tf.complex64)

def calculate_sinr(H_k, V_k_list, W, noise_power, params):
    k_idx = 0
    Nr, Nt = params['Nr'], params['Nt']
    H_k_reshaped = tf.reshape(H_k, (Nr, Nt))
    signal_term = W @ H_k_reshaped @ V_k_list[k_idx]
    signal_power = tf.reduce_sum(tf.square(tf.abs(signal_term)))
    iui_power = 0.0
    for l in range(params['K']):
        if l != k_idx:
            iui_term = W @ H_k_reshaped @ V_k_list[l]
            iui_power += tf.reduce_sum(tf.square(tf.abs(iui_term)))
    noise_out_power = noise_power * tf.reduce_sum(tf.square(tf.abs(W)))
    return signal_power / (iui_power + noise_out_power + 1e-12)

def mrc_combiner(H_k, V_k):
    H_eff = H_k @ V_k
    W_k = tf.linalg.adjoint(H_eff)
    norm = tf.norm(W_k, axis=-1, keepdims=True)
    return tf.math.divide_no_nan(W_k, tf.cast(norm, W_k.dtype))

def mmse_combiner(H_k, V_k_list, noise_variance, params):
    y_k = H_k @ V_k_list[0]
    R_yy_inv = tf.linalg.inv(y_k @ tf.linalg.adjoint(y_k) + tf.cast(noise_variance, tf.complex64) * tf.eye(params['Nr'], dtype=tf.complex64))
    return tf.linalg.adjoint(y_k) @ R_yy_inv

# --- 5. Offline Dataset Generation and Training ---
def generate_dataset(channel_model, params, dataset_size):
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
    return np.array(histories, dtype=np.float32), np.array(channels_H, dtype=np.complex64)

def precompute_labels(channels_H, combiner_codebook, V_k_list, noise_power, params):
    print("--- Pre-computing optimal labels for supervised learning... ---")
    num_states = channels_H.shape[0]
    labels = np.zeros(num_states, dtype=np.int32)
    for i in trange(num_states, desc="Computing Labels"):
        rewards = [np.log2(1 + calculate_sinr(channels_H[i], V_k_list, W, noise_power, params).numpy().real) for W in combiner_codebook]
        labels[i] = np.argmax(rewards)
    return labels

# --- 6. Main Program Execution ---
def main():
    manage_memory()
    strategy = tf.distribute.MirroredStrategy()
    print(f'Number of devices: {strategy.num_replicas_in_sync}')
    global_batch_size = SL_PARAMS['batch_size']
    
    with strategy.scope():
        model = BeamformingClassifier(SL_PARAMS['embedding_dim'], SL_PARAMS['codebook_size'])
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        # CRITICAL FIX: When using a custom training loop with MirroredStrategy,
        # the loss reduction must be set to NONE. The averaging is handled manually
        # using tf.nn.compute_average_loss. This resolves the ValueError.
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE)
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')

    combiner_codebook = create_combiner_codebook(SL_PARAMS['codebook_size'], PARAMS['Ns'], PARAMS['Nr'])
    bs_array = PanelArray(num_rows_per_panel=1, num_cols_per_panel=int(PARAMS['Nt']/2), polarization="dual", polarization_type="VH", antenna_pattern="38.901", carrier_frequency=PARAMS['carrier_freq'])
    ue_array = PanelArray(num_rows_per_panel=1, num_cols_per_panel=PARAMS['Nr'], polarization="single", polarization_type="V", antenna_pattern="omni", carrier_frequency=PARAMS['carrier_freq'])
    channel_model = CDL(model="C", delay_spread=PARAMS['delay_spread'], carrier_frequency=PARAMS['carrier_freq'], ut_array=ue_array, bs_array=bs_array, direction="downlink", min_speed=3.0)
    
    h_sample, d_sample = channel_model(1,1,PARAMS['sampling_frequency'])
    freqs_sample = subcarrier_frequencies(PARAMS['fft_size'], PARAMS['subcarrier_spacing'])
    H_freq_sample = cir_to_ofdm_channel(freqs_sample, h_sample, d_sample)
    H_k_sample = [tf.squeeze(H_freq_sample[..., PARAMS['fft_size']//2]) for _ in range(PARAMS['K'])]
    fixed_V_k = [tf.linalg.svd(h)[2][:, :PARAMS['Ns']] for h in H_k_sample]

    histories_data, channels_data = generate_dataset(channel_model, PARAMS, SL_PARAMS['dataset_size'])
    noise_power_train = 10**(-SL_PARAMS['snr_db_train'] / 10.0)
    labels_data = precompute_labels(channels_data, combiner_codebook, fixed_V_k, noise_power_train, PARAMS)
    manage_memory()

    print(f"\n--- Starting Supervised Training for {SL_PARAMS['num_epochs']} epochs ---")
    dataset = tf.data.Dataset.from_tensor_slices((histories_data, labels_data)).shuffle(SL_PARAMS['dataset_size']).batch(global_batch_size)
    dist_dataset = strategy.experimental_distribute_dataset(dataset)

    @tf.function
    def train_step(dist_inputs):
        batch_histories, batch_labels = dist_inputs
        with tf.GradientTape() as tape:
            logits = model(batch_histories, training=True)
            per_example_loss = loss_fn(batch_labels, logits)
            scaled_loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)
        gradients = tape.gradient(scaled_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_accuracy.update_state(batch_labels, logits)
        return scaled_loss

    train_loss_history = []
    for epoch in range(SL_PARAMS['num_epochs']):
        total_loss = 0.0
        num_batches = 0
        train_accuracy.reset_states()
        for dist_inputs in dist_dataset:
            per_replica_loss = strategy.run(train_step, args=(dist_inputs,))
            total_loss += strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)
            num_batches += 1
        avg_loss = total_loss / num_batches
        train_loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{SL_PARAMS['num_epochs']} finished. Loss: {avg_loss:.4f}, Accuracy: {train_accuracy.result():.4f}")

    print("\n--- Training Finished ---")
    output_dir = "results_v12_supervised"
    os.makedirs(output_dir, exist_ok=True)
    model.save_weights(os.path.join(output_dir, "classifier_weights.h5"))
    manage_memory()

    print("\n--- Starting Evaluation Phase ---")
    eval_model = BeamformingClassifier(SL_PARAMS['embedding_dim'], SL_PARAMS['codebook_size'])
    _ = eval_model(tf.zeros((1, PARAMS['tau'], PARAMS['Nr'] * PARAMS['Nt'] * 2)))
    eval_model.load_weights(os.path.join(output_dir, "classifier_weights.h5"))

    results_ber = {"PPO": [], "MRC": [], "MMSE": []}
    eval_histories, eval_channels = generate_dataset(channel_model, PARAMS, EVAL_PARAMS['num_channel_realizations'])
    
    logits = eval_model(eval_histories, training=False)
    action_indices = tf.argmax(logits, axis=1)
    W_ppo = tf.gather(combiner_codebook, action_indices)

    for snr_db in EVAL_PARAMS['snr_dBs']:
        print(f"--- Evaluating SNR = {snr_db} dB ---")
        noise_var = 10**(-snr_db / 10.0)
        
        W_mrc = tf.stack([mrc_combiner(h, fixed_V_k[0]) for h in eval_channels])
        W_mmse = tf.stack([mmse_combiner(h, fixed_V_k, noise_var, PARAMS) for h in eval_channels])
        
        for name, W_batch in [("PPO", W_ppo), ("MRC", W_mrc), ("MMSE", W_mmse)]:
            sinrs = [calculate_sinr(eval_channels[j], fixed_V_k, W_batch[j], noise_var, PARAMS) for j in range(len(eval_channels))]
            avg_sinr = np.mean([s.numpy().real for s in sinrs])
            ber = (3.0/8.0) * erfc(np.sqrt( (2.0/5.0) * avg_sinr ))
            results_ber[name].append(max(ber, 1e-6))
            
        print(f"  PPO BER: {results_ber['PPO'][-1]:.2e}, MRC BER: {results_ber['MRC'][-1]:.2e}, MMSE BER: {results_ber['MMSE'][-1]:.2e}")

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
    plt.plot(train_loss_history)
    plt.xlabel("Epoch", fontsize=14); plt.ylabel("Training Loss", fontsize=14)
    plt.title("Supervised Learning Curve", fontsize=16); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sl_training_curve.png'))
    print(f"Saved SL training curve plot to '{os.path.join(output_dir, 'sl_training_curve.png')}'")
    plt.close()

    if TSNE_AVAILABLE and len(eval_histories) > 50:
        try:
            eval_embeddings = eval_model.embedding_layer(eval_model.global_pool(eval_model.bn3(eval_model.conv3(eval_model.bn2(eval_model.conv2(eval_model.bn1(eval_model.conv1(eval_histories))))))))
            tsne = TSNE(n_components=2, random_state=SEED, perplexity=min(30, len(eval_embeddings)-1))
            tsne_embeds = tsne.fit_transform(eval_embeddings)
            
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
