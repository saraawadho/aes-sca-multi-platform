import numpy as np
import matplotlib.pyplot as plt
import h5py
from aes import AES
from tqdm import tqdm

def main():
    filepath = 'traces.hdf5'
    first_round_start = 2000
    first_round_stop = 4000

    with h5py.File(filepath, 'r') as f:
        key = f["keys"][0,:]
        plaintexts = f["plaintexts"][0,:,:]
        traces = (f["power"][0,:,first_round_start:first_round_stop]).astype(float)

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
        plt.title("DPA: 256 subkey candidates for S-box {:d}".format(i+1))
        plt.savefig("dpa_dom_sbox_{:d}.png".format(i+1), dpi=300)
        plt.close()

    doms = np.max(abs(doms), axis=2)
    roundkey = np.argmax(doms, axis=1)
    count = np.count_nonzero(key == roundkey)
    print("DPA successfully recovered {:d} out of 16 subkeys.".format(count))
    if (count >= 12) and (count < 16):
        print("A small brute-force search does the rest.")

if __name__ == "__main__":
    main()