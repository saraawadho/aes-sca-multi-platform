from teledynelecroyscope import TeledyneLeCroyScope
from CW308_AES import CW308_STM32F4_AES
import numpy as np
import h5py
from tqdm import tqdm
import time

def main():
    filepath = 'traces.hdf5'
    nb_of_keys = 2 # One key suffices for SPA, DPA, and CPA. For template attacks, two keys are needed.
    nb_of_samples = 20_000
    nb_of_traces = 10_000

    scope = TeledyneLeCroyScope(int16_not_int8=True, float_not_int=False)
    cw308 = CW308_STM32F4_AES()
    cw308.test_correctness()

    print('Collecting AES traces...')
    with h5py.File(filepath, 'w') as f:
        f.create_dataset("trigger", shape=(nb_of_samples), dtype=np.int16)
        f.create_dataset("keys", shape=(nb_of_keys, 16), dtype=np.uint8)
        f.create_dataset("plaintexts", shape=(nb_of_keys, nb_of_traces, 16), dtype=np.uint8)
        f.create_dataset("power", shape=(nb_of_keys, nb_of_traces, nb_of_samples), dtype=np.int16)
        for i in range(nb_of_keys):
            key = np.random.randint(0, high=256, size=16, dtype=np.uint8)
            f["keys"][i,:] = key
            cw308.write_key(key.tobytes())
            for j in tqdm(range(nb_of_traces)):
                plaintext = np.random.randint(0, high=256, size=16, dtype=np.uint8)
                f["plaintexts"][i,:] = plaintext
                cw308.write_plaintext(plaintext.tobytes())
                scope.arm_single_trace()
                time.sleep(0.05)
                cw308.encrypt()
                f["power"][i,j,:] = scope.get_single_trace(channel='C3', nb_of_samples = nb_of_samples)
        f["trigger"][:] = scope.get_single_trace(channel='C1', nb_of_samples = nb_of_samples)

if __name__ == '__main__':
    main()