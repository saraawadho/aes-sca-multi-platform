import numpy as np
import matplotlib.pyplot as plt
import h5py

def main():
    filepath = 'traces.hdf5'

    with h5py.File(filepath, 'r') as f:
        nb_of_traces = f["power"].shape[1]
        nb_of_samples = f["power"].shape[2]
        trace_single = f["power"][0,0,:]
        trace_mean = np.zeros((nb_of_samples), dtype=float)
        for i in range(nb_of_traces):
            trace_mean += f["power"][0,i,:]
    trace_mean /= nb_of_traces

    plt.plot(trace_single, linewidth=0.3)
    plt.xlabel("Time [Samples]")
    plt.ylabel("Power consumption")
    plt.title("A single trace")
    plt.savefig("spa_trace_single.png", dpi=300)
    plt.close()

    plt.plot(trace_mean, linewidth=0.3)
    plt.xlabel("Time [Samples]")
    plt.ylabel("Power consumption")
    plt.title("Mean trace")
    plt.savefig("spa_trace_mean.png", dpi=300)
    plt.close()

    print("In the mean trace, a pattern with 10 repetitions is clearly visible.")
    print("It is reasonable to assume that these are the 10 AES-encryption rounds.")
    print("For first- and last-round attacks, only a small part of the matrix of traces needs to be retained.")
    print("This way, fewer computations have to be performed.")

if __name__ == "__main__":
    main()