# CPA on AES_HD dataset — CORRECT Hamming Distance model
# Fix: FPGA power leaks HW(state_before XOR state_after) not HW(state_after)
# Model: HW(SBOX_INV[ct^k] XOR (ct^k))  i.e. bits that FLIP during last SubBytes
import numpy as np
import matplotlib.pyplot as plt
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
    print("Loading AES_HD traces and ciphertexts...")
    traces = np.load('../analysis/AES_HD_dataset/attack_traces_AES_HD.npy').astype(float)
    ct     = np.load('../analysis/AES_HD_dataset/attack_ciphertext_AES_HD.npy').astype(np.uint8)

    # Only byte 7 varies across traces — attack that byte of the last-round key
    TARGET_BYTE = 7
    ct7 = ct[:, TARGET_BYTE]
    nb_traces, nb_samples = traces.shape
    print(f"Traces: {nb_traces} x {nb_samples} | Attacking last-round key byte {TARGET_BYTE}")

    print("Computing Pearson's correlation coefficient (last round, Hamming Distance model)...")
    corrcoefs = np.zeros((256, nb_samples), dtype=float)
    for k in tqdm(range(256)):
        intermediate = (ct7 ^ k).astype(np.uint8)
        hd = hamming_weight(AES.SBOX_INV[intermediate] ^ intermediate).astype(float)
        corrcoefs[k, :] = pearson_correlation(traces, hd)

    # Plot all 256 candidates
    best_k = int(np.argmax(np.max(np.abs(corrcoefs), axis=1)))
    peak   = float(np.max(np.abs(corrcoefs[best_k])))
    plt.figure(figsize=(12, 5))
    plt.plot(np.transpose(corrcoefs), linewidth=0.3, alpha=0.6)
    plt.plot(corrcoefs[best_k], linewidth=1.2, color='red',
             label=f'Best k=0x{best_k:02X} (peak={peak:.4f})')
    plt.xlabel("Time [Samples]")
    plt.ylabel("Pearson's correlation coefficient")
    plt.title(f"CPA on AES_HD (FPGA last round): byte {TARGET_BYTE}, {nb_traces} traces")
    plt.legend()
    plt.tight_layout()
    plt.savefig("cpa_aeshd_hd_byte7.png", dpi=200)
    plt.close()
    print(f"Best last-round key byte {TARGET_BYTE}: 0x{best_k:02X}  (peak corr = {peak:.4f})")
    print("Saved: cpa_aeshd_hd_byte7.png")

    # Trace-count convergence
    trace_counts = [100, 250, 500, 1000, 2500, 5000, 10000, nb_traces]
    peaks = []
    for n in trace_counts:
        cc = np.zeros((256, nb_samples))
        for k in range(256):
            intermediate = (ct7[:n] ^ k).astype(np.uint8)
            hd = hamming_weight(AES.SBOX_INV[intermediate] ^ intermediate).astype(float)
            cc[k, :] = pearson_correlation(traces[:n], hd)
        peaks.append(float(np.max(np.abs(cc[best_k]))))

    plt.figure(figsize=(8, 4))
    plt.plot(trace_counts, peaks, 'o-', color='steelblue')
    plt.xlabel("Number of traces")
    plt.ylabel("Peak |correlation| for correct key")
    plt.title("CPA AES_HD: convergence vs trace count")
    plt.xscale('log')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("cpa_aeshd_hd_convergence.png", dpi=200)
    plt.close()
    print("Saved: cpa_aeshd_hd_convergence.png")

if __name__ == "__main__":
    main()