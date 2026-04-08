# CPA on Cortex M0 unprotected AES dataset (Zenodo 4742593)
# Changes from original:
#   - Loads traces from trace_set_10k.npy and plaintexts from plaintext.txt
#   - No known key — recovered key printed for inspection
#   - Uses all samples (no window slicing needed)
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
    y_mean = np.mean(y, axis=0)
    y_std = np.std(y, axis=0)
    cov = np.matmul(np.transpose(x - x_mean[np.newaxis, :]), y - y_mean[np.newaxis, :])
    cov /= x.shape[0]
    corrcoef = cov / np.outer(x_std, y_std)
    return corrcoef

def load_data():
    traces = np.load('trace_set_10k.npy').astype(float)
    with open('plaintext.txt') as f:
        lines = [l.strip() for l in f if l.strip()]
    plaintexts = np.array([list(bytes.fromhex(l)) for l in lines], dtype=np.uint8)
    n = min(len(traces), len(plaintexts))
    return traces[:n], plaintexts[:n]

def main():
    print("Loading Cortex M0 traces and plaintexts...")
    traces, plaintexts = load_data()
    print(f"Traces: {traces.shape}, Plaintexts: {plaintexts.shape}")

    print("Computing Pearson's correlation coefficient...")
    nb_of_samples = traces.shape[1]
    corrcoefs = np.zeros((16, 256, nb_of_samples), dtype=float)
    for subkey in tqdm(range(256)):
        hw = np.copy(plaintexts)
        hw ^= np.full(plaintexts.shape, subkey, dtype=np.uint8)
        hw = hamming_weight(AES.SBOX[hw]).astype(float)
        cc = pearson_correlation(hw, traces)
        corrcoefs[:,subkey,:] = cc

    print("Recovering subkeys...")
    for i in range(16):
        plt.plot(np.transpose(corrcoefs[i,:,:]), linewidth=0.3)
        plt.xlabel("Time [Samples]")
        plt.ylabel("Pearson's correlation coefficient")
        plt.title("CPA Cortex M0: 256 subkey candidates S-box {:d}".format(i+1))
        plt.savefig("cpa_cortexm0_sbox_{:d}.png".format(i+1), dpi=200)
        plt.close()

    corrcoefs = np.max(abs(corrcoefs), axis=2)
    roundkey = np.argmax(corrcoefs, axis=1)
    print("CPA recovered key: " + " ".join(f"{b:02X}" for b in roundkey))

if __name__ == "__main__":
    main()