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

def template_building():
    filepath = 'traces.hdf5'
    first_round_start = 2000
    first_round_stop = 4000

    with h5py.File(filepath, 'r') as f:
        key = f["keys"][0,:]
        plaintexts = f["plaintexts"][0,:,:]
        traces = (f["power"][0,:,first_round_start:first_round_stop]).astype(float)

    print("Building Hamming weight templates...")
    hw = hamming_weight(AES.SBOX[plaintexts ^ key[np.newaxis,:]])
    nb_of_samples = traces.shape[1]
    nb_of_retained_samples = 10
    templates = np.zeros((16, 9, nb_of_retained_samples), dtype=float)
    indices = np.zeros((16, nb_of_retained_samples), dtype=int)
    for i in tqdm(range(16)):
        means = np.zeros((9, nb_of_samples), dtype=float)
        for j in range(9):
            ind = (hw[:,i] == j)
            means[j, :] = np.mean(traces[ind,:], axis=0)
        corrcoef = np.zeros((nb_of_samples), dtype=float)
        for j in range(nb_of_samples):
            cc = np.corrcoef(traces[:,j], hw[:,i])
            corrcoef[j] = cc[0,1]
        ind = np.argsort(abs(corrcoef))
        ind = np.sort(ind[-nb_of_retained_samples:])
        indices[i,:] = ind
        templates[i, :, :] = means[:, ind]

    for i in tqdm(range(16)):
        plt.plot(indices[i,:], np.transpose(templates[i,:,:]))
        plt.xlabel("Time [Samples]")
        plt.ylabel("Mean")
        plt.title("Templates for S-box {:d}".format(i+1))
        plt.legend(["0","1","2","3","4","5","6","7","8"])
        plt.savefig("templates_build_sbox_{:d}.png".format(i+1), dpi=300)
        plt.close()
    return [indices, templates]

def template_matching(indices, templates):
    filepath = 'traces.hdf5'
    first_round_start = 2000
    first_round_stop = 4000

    with h5py.File(filepath, 'r') as f:
        key = f["keys"][1,:]
        plaintexts = f["plaintexts"][1,:,:]
        traces = (f["power"][1,:,first_round_start:first_round_stop]).astype(float)

    print("Matching Hamming weight templates...")
    match = np.zeros((16,256),dtype=float)
    for i in tqdm(range(16)):
        traces_reduced = traces[:,indices[i,:]]
        for k in range(256):
            hw = hamming_weight(AES.SBOX[plaintexts[:,i] ^ k])
            templates_selected = templates[i,hw,:]
            match[i,k] = -np.sum(np.sum(abs(traces_reduced - templates_selected)))

    for i in tqdm(range(16)):
        plt.plot(np.arange(256), match[i,:], linewidth=0.5)
        plt.xlabel("Subkey")
        plt.ylabel("Match")
        plt.title("Subkey match for S-box {:d}".format(i+1))
        plt.savefig("templates_match_sbox_{:d}.png".format(i+1), dpi=300)
        plt.close()

    roundkey = np.argmax(match, axis=1)
    count = np.count_nonzero(key == roundkey)
    print("Template attack successfully recovered {:d} out of 16 subkeys".format(count))
    if (count >= 12) and (count < 16):
        print("A small brute-force search does the rest.")

def main():
    indices, templates = template_building()
    template_matching(indices, templates)

if __name__ == "__main__":
    main()