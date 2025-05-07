import numpy as np
import os


if __name__ == "__main__":
    optimal_parameters = np.load(os.path.join("/home/bella/Phd/code/quality_control_CMR/params", "params.npy"), allow_pickle=True).item()
    print(optimal_parameters)
