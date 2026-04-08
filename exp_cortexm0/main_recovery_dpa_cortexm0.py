# DPA on Cortex M0 unprotected AES dataset (Zenodo 4742593)
# Changes from original:
#   - Loads traces from trace_set_10k.npy and plaintexts from plaintext.txt
#   - No known key — recovered key printed for inspection
#   - Uses all samples (no window slicing needed)
import numpy as np
import matplotlib.pyplot as plt
from aes import AES
from tqdm import tqdm

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

    print("Computing Difference of Means (DoMs)...")
    nb_of_samples = traces.shape[1]
    doms = np.zeros((16, 256, nb_of_samples), dtype=float)
    for subkey in tqdm(range(256)):
        sbox_lsbs = np.copy(plaintexts)
        sbox_lsbs ^= np.full(plaintexts.shape, subkey, dtype=np.uint8)
        sbox_lsbs = AES.SBOX[sbox_lsbs] & 0x1
        for i in range(16):
            ind = (sbox_lsbs[:,i] == 1)
            dom = np.mean(traces[ind,:],axis=0) - np.mean(traces[~ind,:], axis=0)
            doms[i,subkey,:] = dom

    print("Recovering subkeys...")
    for i in range(16):
        plt.plot(np.transpose(doms[i,:,:]), linewidth=0.3)
        plt.xlabel("Time [Samples]")
        plt.ylabel("DoM")
        plt.title("DPA Cortex M0: 256 subkey candidates for S-box {:d}".format(i+1))
        plt.savefig("dpa_cortexm0_sbox_{:d}.png".format(i+1), dpi=200)
        plt.close()

    doms = np.max(abs(doms), axis=2)
    roundkey = np.argmax(doms, axis=1)
    print("DPA recovered key: " + " ".join(f"{b:02X}" for b in roundkey))

    # Compare against CPA-recovered key (ground truth)
    cpa_key = np.array([0xCA,0xFE,0xBA,0xBE,0xDE,0xAD,0xBE,0xAF,
                        0xCA,0xFE,0xBA,0xBE,0xDE,0xAD,0xBE,0xAF], dtype=np.uint8)
    matches = np.count_nonzero(roundkey == cpa_key)
    print(f"Matches vs CPA key: {matches}/16")
    for i in range(16):
        status = "OK" if roundkey[i] == cpa_key[i] else f"WRONG (expected 0x{cpa_key[i]:02X})"
        print(f"  Byte {i:2d}: 0x{roundkey[i]:02X}  {status}")

if __name__ == "__main__":
    main()