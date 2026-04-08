# DPA on ASCAD (masked AES, ATMega8515)
# Changes from original:
#   - Loads from ASCAD.h5 (Attack_traces group)
#   - Target: byte 2 only (ASCAD standard target)
#   - Key is fixed — same for all attack traces
#   - Expected: DPA FAILS due to Boolean masking
import numpy as np
import matplotlib.pyplot as plt
import h5py
from aes import AES
from tqdm import tqdm

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

    print("Computing Difference of Means (1st order DPA, LSB model)...")
    doms = np.zeros((256, nb_samples), dtype=float)
    for k in tqdm(range(256)):
        lsb = AES.SBOX[pt2 ^ k] & 0x1
        ind = (lsb == 1)
        doms[k, :] = np.mean(traces[ind, :], axis=0) - np.mean(traces[~ind, :], axis=0)

    best_k = int(np.argmax(np.max(np.abs(doms), axis=1)))
    peak   = float(np.max(np.abs(doms[best_k])))
    true_peak = float(np.max(np.abs(doms[key[TARGET_BYTE]])))

    plt.figure(figsize=(12, 5))
    plt.plot(np.transpose(doms), linewidth=0.3, alpha=0.5, color='gray')
    plt.plot(doms[key[TARGET_BYTE]], linewidth=1.2, color='green',
             label=f'True key 0x{key[TARGET_BYTE]:02X} (peak={true_peak:.4f})')
    plt.plot(doms[best_k], linewidth=1.0, color='red', linestyle='--',
             label=f'Best candidate 0x{best_k:02X} (peak={peak:.4f})')
    plt.xlabel("Time [Samples]")
    plt.ylabel("Difference of Means (DoM)")
    plt.title(f"DPA on ASCAD (masked AES): byte {TARGET_BYTE}, {nb_traces} traces")
    plt.legend()
    plt.tight_layout()
    plt.savefig("dpa_ascad_byte2.png", dpi=200)
    plt.close()

    recovered = key[TARGET_BYTE] == best_k
    print(f"True key byte {TARGET_BYTE}:  0x{key[TARGET_BYTE]:02X}  (peak |DoM| = {true_peak:.4f})")
    print(f"DPA best candidate: 0x{best_k:02X}  (peak |DoM| = {peak:.4f})")
    print(f"DPA correct? {'YES' if recovered else 'NO — masking defeated DPA'}")
    print("Saved: dpa_ascad_byte2.png")

if __name__ == "__main__":
    main()