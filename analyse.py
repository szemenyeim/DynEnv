import numpy as np
import h5py
import matplotlib.pyplot as plt

if __name__ == '__main__':
    print("ROBO")
    print("ICM")
    fName = "./log/RoboCup/time_log_2019-11-28 13_58_18.hdf5"
    f = h5py.File(fName, 'r')
    pos0 = np.array(f['ep_pos_rewards']['mean'])
    print(np.max(pos0))
    pos01 = np.array(f['ep_rewards']['mean'])
    print(np.max(pos01))
    print("ICM-TER")
    fName = "./log/RoboCup/time_log_2019-11-28 04_15_58.hdf5"
    f = h5py.File(fName, 'r')
    pos = np.array(f['ep_pos_rewards']['mean'])
    print(np.max(pos))
    pos1 = np.array(f['ep_rewards']['mean'])
    print(np.max(pos1))
    print("RCM")
    fName = "./log/RoboCup/time_log_2019-11-27 18_08_03.hdf5"
    f = h5py.File(fName, 'r')
    pos2 = np.array(f['ep_pos_rewards']['mean'])
    print(np.max(pos2))
    pos3 = np.array(f['ep_rewards']['mean'])
    print(np.max(pos3))

    ep = range(pos3.shape[0])
    plt.figure()
    plt.plot(ep,pos01,ep,pos1,ep,pos3)
    plt.show()

    print("Drive")
    print("RCM")
    fName = "./log/Driving/time_log_2019-11-25 18_27_42.hdf5"
    f = h5py.File(fName, 'r')
    pos1 = np.array(f['ep_rewards']['mean'])
    print(np.max(pos1))
    fName = "./log/Driving/time_log_2019-11-25 13_03_07.hdf5"
    print("ICM")
    f = h5py.File(fName, 'r')
    pos3 = np.array(f['ep_rewards']['mean'])
    print(np.max(pos3))
    fName = "./log/Driving/time_log_2019-11-24 19_37_30.hdf5"
    print("ICM-TER")
    f = h5py.File(fName, 'r')
    pos5 = np.array(f['ep_rewards']['mean'])
    print(np.max(pos5))
