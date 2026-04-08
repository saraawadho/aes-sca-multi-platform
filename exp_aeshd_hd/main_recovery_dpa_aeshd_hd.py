# DPA on AES_HD dataset — CORRECT Hamming Distance model
# Fix: FPGA power leaks HW(state_before XOR state_after) not HW(state_after)
# Model: LSB(SBOX_INV[ct^k] XOR (ct^k))  i.e. LSB of bits that FLIP during last SubBytes
import numpy as np
import matplotlib.pyplot as plt
from aes import AES
from tqdm import tqdm

def main():
    print("Loading AES_HD traces and ciphertexts...")
    traces = np.load('../analysis/AES_HD_dataset/attack_traces_AES_HD.npy').astype(float)
    ct     = np.load('../analysis/AES_HD_dataset/attack_ciphertext_AES_HD.npy').astype(np.uint8)

    TARGET_BYTE = 7
    ct7 = ct[:, TARGET_BYTE]
    nb_traces, nb_samples = traces.shape
    print(f"Traces: {nb_traces} x {nb_samples} | Attacking last-round key byte {TARGET_BYTE}")

    print("Computing Difference of Means (DoMs, last round, Hamming Distance model)...")
    doms = np.zeros((256, nb_samples), dtype=float)
    for k in tqdm(range(256)):
        intermediate = (ct7 ^ k).astype(np.uint8)
        lsb = (AES.SBOX_INV[intermediate] ^ intermediate) & 0x1
        ind = (lsb == 1)
        doms[k, :] = np.mean(traces[ind, :], axis=0) - np.mean(traces[~ind, :], axis=0)

    best_k = int(np.argmax(np.max(np.abs(doms), axis=1)))
    peak   = float(np.max(np.abs(doms[best_k])))

    plt.figure(figsize=(12, 5))
    plt.plot(np.transpose(doms), linewidth=0.3, alpha=0.6)
    plt.plot(doms[best_k], linewidth=1.2, color='red',
             label=f'Best k=0x{best_k:02X} (peak={peak:.4f})')
    plt.xlabel("Time [Samples]")
    plt.ylabel("DoM")
    plt.title(f"DPA on AES_HD (FPGA last round): byte {TARGET_BYTE}, {nb_traces} traces")
    plt.legend()
    plt.tight_layout()
    plt.savefig("dpa_aeshd_hd_byte7.png", dpi=200)
    plt.close()
    print(f"Best last-round key byte {TARGET_BYTE}: 0x{best_k:02X}  (peak DoM = {peak:.4f})")
    print("Saved: dpa_aeshd_hd_byte7.png")

    # Trace-count convergence
    trace_counts = [100, 250, 500, 1000, 2500, 5000, 10000, nb_traces]
    peaks = []
    for n in trace_counts:
        dd = np.zeros((256, nb_samples))
        for k in range(256):
            intermediate = (ct7[:n] ^ k).astype(np.uint8)
            lsb = (AES.SBOX_INV[intermediate] ^ intermediate) & 0x1
            ind = (lsb == 1)
            dd[k, :] = np.mean(traces[:n][ind, :], axis=0) - np.mean(traces[:n][~ind, :], axis=0)
        peaks.append(float(np.max(np.abs(dd[best_k]))))

    plt.figure(figsize=(8, 4))
    plt.plot(trace_counts, peaks, 'o-', color='darkorange')
    plt.xlabel("Number of traces")
    plt.ylabel("Peak |DoM| for correct key")
    plt.title("DPA AES_HD: convergence vs trace count")
    plt.xscale('log')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("dpa_aeshd_hd_convergence.png", dpi=200)
    plt.close()
    print("Saved: dpa_aeshd_hd_convergence.png")

if __name__ == "__main__":
    main()