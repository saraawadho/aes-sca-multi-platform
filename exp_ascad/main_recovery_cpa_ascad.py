# CPA on ASCAD (masked AES, ATMega8515)
# Changes from original:
#   - Loads from ASCAD.h5 (Attack_traces group)
#   - Target: byte 2 only (ASCAD standard target)
#   - Key is fixed — same for all attack traces
#   - Expected: CPA FAILS due to Boolean masking
import numpy as np
import matplotlib.pyplot as plt
import h5py
from aes import AES
from tqdm import tqdm

def hamming_weight(x):
    weight = np.zeros(x.shape, dtype=int)
    for i in range(8):
        weight += (x >> i) & 1
    return weight

def pearson_correlation(x, y):
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    y_mean = np.mean(y)
    y_std = np.std(y)
    cov = np.dot(y - y_mean, x - x_mean[np.newaxis, :]) / len(y)
    corrcoef = cov / (x_std * y_std + 1e-12)
    return corrcoef

def main():
    filepath = 'ASCAD.h5'
    TARGET_BYTE = 2  # ASCAD standard target byte

    print("Loading ASCAD attack traces...")
    with h5py.File(filepath, 'r') as f:
        traces    = f['Attack_traces/traces'][:].astype(float)
        metadata  = f['Attack_traces/metadata']
        key       = metadata['key'][0]          # fixed key, same for all traces
        plaintexts = metadata['plaintext'][:]   # (10000, 16)

    nb_traces, nb_samples = traces.shape
    print(f"Traces: {nb_traces} x {nb_samples} | Target byte: {TARGET_BYTE}")
    print(f"True key byte {TARGET_BYTE}: 0x{key[TARGET_BYTE]:02X}")

    pt2 = plaintexts[:, TARGET_BYTE]

    print("Computing Pearson's correlation (1st order CPA, HW model)...")
    corrcoefs = np.zeros((256, nb_samples), dtype=float)
    for k in tqdm(range(256)):
        hw = hamming_weight(AES.SBOX[pt2 ^ k]).astype(float)
        corrcoefs[k, :] = pearson_correlation(traces, hw)

    best_k = int(np.argmax(np.max(np.abs(corrcoefs), axis=1)))
    peak   = float(np.max(np.abs(corrcoefs[best_k])))
    true_peak = float(np.max(np.abs(corrcoefs[key[TARGET_BYTE]])))

    plt.figure(figsize=(12, 5))
    plt.plot(np.transpose(corrcoefs), linewidth=0.3, alpha=0.5, color='gray')
    plt.plot(corrcoefs[key[TARGET_BYTE]], linewidth=1.2, color='green',
             label=f'True key 0x{key[TARGET_BYTE]:02X} (peak={true_peak:.4f})')
    plt.plot(corrcoefs[best_k], linewidth=1.0, color='red', linestyle='--',
             label=f'Best candidate 0x{best_k:02X} (peak={peak:.4f})')
    plt.xlabel("Time [Samples]")
    plt.ylabel("Pearson's correlation coefficient")
    plt.title(f"CPA on ASCAD (masked AES): byte {TARGET_BYTE}, {nb_traces} traces")
    plt.legend()
    plt.tight_layout()
    plt.savefig("cpa_ascad_byte2.png", dpi=200)
    plt.close()

    recovered = key[TARGET_BYTE] == best_k
    print(f"True key byte {TARGET_BYTE}:  0x{key[TARGET_BYTE]:02X}  (peak |corr| = {true_peak:.4f})")
    print(f"CPA best candidate: 0x{best_k:02X}  (peak |corr| = {peak:.4f})")
    print(f"CPA correct? {'YES' if recovered else 'NO — masking defeated CPA'}")
    print("Saved: cpa_ascad_byte2.png")

if __name__ == "__main__":
    main()