# Deep Learning SCA on ASCAD (masked AES, ATMega8515)
# Uses ANSSI's pre-trained MLP to break Boolean-masked AES
# where classical CPA/DPA fail completely.
#
# Model: mlp_best_ascad_desync0_node200_layernb6_epochs200_classes256_batchsize100.h5
# Reference: Benadjila et al., "Study of Deep Learning Techniques for
#            Side-Channel Analysis and Introduction to ASCAD Database", 2019
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

def load_ascad(filepath, target_byte=2):
    """Load attack traces and metadata from ASCAD.h5."""
    with h5py.File(filepath, 'r') as f:
        traces     = f['Attack_traces/traces'][:].astype(np.float32)
        metadata   = f['Attack_traces/metadata']
        key        = metadata['key'][0]
        plaintexts = metadata['plaintext'][:]
        masks      = metadata['masks'][:]  # available for analysis, not used in attack
    return traces, plaintexts, key, masks

SBOX = np.array([
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16,
], dtype=np.uint8)

def rank_key_bytes(predictions, plaintexts_byte):
    """
    Compute accumulated log-likelihood for each key candidate using plaintext.
    For each trace t and key k: score[k] += log P(SBOX[pt[t]^k] | trace_t)
    This maps the 256-class model output to 256 key candidates.
    """
    nb_traces = predictions.shape[0]
    scores = np.zeros(256, dtype=float)
    log_preds = np.log(predictions + 1e-40)   # (nb_traces, 256)
    for k in range(256):
        intermediates = SBOX[plaintexts_byte ^ k]   # (nb_traces,) — class index per trace
        scores[k] = np.sum(log_preds[np.arange(nb_traces), intermediates])
    return scores

def key_rank_evolution(predictions, plaintexts_byte, true_key_byte, n_experiments=100):
    """
    Compute average key rank of the correct byte as a function of trace count.
    """
    nb_traces = predictions.shape[0]
    max_traces = min(nb_traces, 2000)
    ranks = np.zeros((n_experiments, max_traces))

    for exp in range(n_experiments):
        perm = np.random.permutation(nb_traces)[:max_traces]
        preds_s = predictions[perm]
        pt_s    = plaintexts_byte[perm]
        log_acc = np.zeros(256)
        for t in range(max_traces):
            for k in range(256):
                log_acc[k] += np.log(preds_s[t, SBOX[pt_s[t] ^ k]] + 1e-40)
            sorted_scores = np.argsort(log_acc)[::-1]
            rank = int(np.where(sorted_scores == true_key_byte)[0][0])
            ranks[exp, t] = rank

    return np.mean(ranks, axis=0)

def main():
    ascad_file = 'ASCAD.h5'
    model_file = 'mlp_best_ascad_desync0_node200_layernb6_epochs200_classes256_batchsize100.h5'
    TARGET_BYTE = 2

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading ASCAD attack traces...")
    traces, plaintexts, key, masks = load_ascad(ascad_file, TARGET_BYTE)
    nb_traces, nb_samples = traces.shape
    true_key_byte = int(key[TARGET_BYTE])
    print(f"Traces: {nb_traces} x {nb_samples}")
    print(f"True key byte {TARGET_BYTE}: 0x{true_key_byte:02X}")

    # ── Load pre-trained model ─────────────────────────────────────────────────
    # The ASCAD MLP was saved with old Keras (TF 1.x). Rebuild architecture
    # manually and load weights to avoid version incompatibility.
    # Architecture (from ANSSI paper): 6 × Dense(200, ReLU) → Dense(256, softmax)
    print(f"Loading pre-trained MLP weights from {model_file}...")
    try:
        import tensorflow as tf
        import h5py as _h5

        # Build the same MLP architecture
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(nb_samples,)),
        ] + [
            tf.keras.layers.Dense(200, activation='relu') for _ in range(5)
        ] + [
            tf.keras.layers.Dense(256, activation='softmax')
        ])

        # Load weights layer by layer from the h5 file
        # Old Keras format: model_weights/dense_N/dense_N/{kernel:0, bias:0}
        with _h5.File(model_file, 'r') as mf:
            wg = mf['model_weights']
            layer_names = sorted([k for k in wg.keys() if 'dense' in k],
                                  key=lambda x: int(x.split('_')[-1]))
            keras_layers = [l for l in model.layers if len(l.get_weights()) > 0]
            for layer, lname in zip(keras_layers, layer_names):
                inner = wg[lname][lname]
                kernel = np.array(inner['kernel:0'])
                bias   = np.array(inner['bias:0'])
                layer.set_weights([kernel, bias])
        print("Weights loaded successfully.")
        model.summary()
    except ImportError:
        print("TensorFlow not found. Install with: pip install tensorflow-macos")
        return
    except Exception as e:
        print(f"Model load error: {e}")
        import traceback; traceback.print_exc()
        return

    # ── Run inference ──────────────────────────────────────────────────────────
    # The MLP was trained to predict the 8-bit intermediate value
    # SBOX[plaintext[2] ^ key[2]], so its output is a 256-class probability.
    print("Running model inference on attack traces...")
    predictions = model.predict(traces, batch_size=200, verbose=1)
    # predictions shape: (nb_traces, 256)

    # ── Identify correct key from accumulated predictions ──────────────────────
    pt2 = plaintexts[:, TARGET_BYTE]
    scores = rank_key_bytes(predictions, pt2)
    sorted_keys = np.argsort(scores)[::-1]
    best_k = int(sorted_keys[0])
    true_rank = int(np.where(sorted_keys == true_key_byte)[0][0])

    print(f"\n--- DL Attack Result ---")
    print(f"True key byte {TARGET_BYTE}: 0x{true_key_byte:02X}")
    print(f"DL best candidate:          0x{best_k:02X}")
    print(f"Rank of true key:           {true_rank} (0 = best)")
    print(f"DL correct? {'YES' if best_k == true_key_byte else 'NO'}")

    # ── Plot score bar chart ───────────────────────────────────────────────────
    plt.figure(figsize=(14, 4))
    colors = ['green' if i == true_key_byte else ('red' if i == best_k else 'steelblue')
              for i in range(256)]
    plt.bar(range(256), scores, color=colors, width=1.0)
    plt.xlabel("Key byte candidate")
    plt.ylabel("Accumulated log-likelihood")
    plt.title(f"DL Attack on ASCAD (masked AES): byte {TARGET_BYTE}, {nb_traces} traces\n"
              f"True key 0x{true_key_byte:02X} shown in green")
    plt.tight_layout()
    plt.savefig("dl_ascad_scores_byte2.png", dpi=200)
    plt.close()
    print("Saved: dl_ascad_scores_byte2.png")

    # ── Key rank vs trace count ────────────────────────────────────────────────
    print("Computing key rank convergence (100 random orderings)...")
    avg_ranks = key_rank_evolution(predictions, pt2, true_key_byte, n_experiments=100)

    plt.figure(figsize=(10, 4))
    plt.plot(range(1, len(avg_ranks) + 1), avg_ranks, color='darkorange', linewidth=1.0)
    plt.axhline(0, color='green', linestyle='--', linewidth=0.8, label='Rank 0 (correct)')
    plt.xlabel("Number of attack traces")
    plt.ylabel("Average key rank")
    plt.title(f"DL Attack on ASCAD: key rank convergence (byte {TARGET_BYTE})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("dl_ascad_rank_convergence_byte2.png", dpi=200)
    plt.close()
    print("Saved: dl_ascad_rank_convergence_byte2.png")

    # ── Summary comparison ─────────────────────────────────────────────────────
    traces_to_rank0 = next((i for i, r in enumerate(avg_ranks) if r < 0.5), None)
    print(f"\n--- Summary ---")
    print(f"Traces needed for rank-0 recovery: "
          f"~{traces_to_rank0}" if traces_to_rank0 else "not reached within 2000 traces")

if __name__ == "__main__":
    main()
