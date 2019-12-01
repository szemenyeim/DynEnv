import numpy as np
import h5py
import matplotlib.pyplot as plt

if __name__ == '__main__':
    print("ROBO")
    print("ICM")
    fName = "../../log/RoboCup/time_log_2019-11-28 13_58_18.hdf5"
    f = h5py.File(fName, 'r')
    pos0 = np.array(f['ep_pos_rewards']['mean'])
    max0 = np.array(f['ep_pos_rewards']['max'])
    print(np.max(pos0),np.max(max0))
    pos01 = np.array(f['ep_rewards']['mean'])
    max01 = np.array(f['ep_rewards']['max'])
    print(np.max(pos01),np.max(max01))
    print("ICM-TER")
    fName = "../../log/RoboCup/time_log_2019-11-28 04_15_58.hdf5"
    f = h5py.File(fName, 'r')
    pos = np.array(f['ep_pos_rewards']['mean'])
    max = np.array(f['ep_pos_rewards']['max'])
    print(np.max(pos),np.max(max))
    pos1 = np.array(f['ep_rewards']['mean'])
    max1 = np.array(f['ep_rewards']['max'])
    print(np.max(pos1),np.max(max1))
    print("RCM")
    fName = "../../log/RoboCup/time_log_2019-11-27 18_08_03.hdf5"
    f = h5py.File(fName, 'r')
    pos2 = np.array(f['ep_pos_rewards']['mean'])
    max2 = np.array(f['ep_pos_rewards']['max'])
    print(np.max(pos2),np.max(max2))
    pos3 = np.array(f['ep_rewards']['mean'])
    max3 = np.array(f['ep_rewards']['max'])
    print(np.max(pos3),np.max(max3))

    ep = range(pos3.shape[0])
    plt.figure()
    plt.plot(ep,pos3)
    plt.xticks(np.arange(0, pos3.shape[0], 50))

    print("Drive")
    print("RCM2")
    fName = "../../log/Driving/time_log_2019-11-30 16_00_09.hdf5"
    f = h5py.File(fName, 'r')
    pos0 = np.array(f['ep_rewards']['mean'])
    avg_pos0 = np.convolve(pos0, np.ones((10,))/10, mode='same')
    max0 = np.array(f['ep_rewards']['max'])
    print(np.max(avg_pos0),np.max(pos0),np.max(max0))
    print("RCM")
    fName = "../../log/Driving/time_log_2019-11-25 18_27_42.hdf5"
    f = h5py.File(fName, 'r')
    pos1 = np.array(f['ep_rewards']['mean'])
    max1 = np.array(f['ep_rewards']['max'])
    print(np.max(pos1),np.max(max1))
    fName = "../../log/Driving/time_log_2019-11-25 13_03_07.hdf5"
    print("ICM")
    f = h5py.File(fName, 'r')
    pos3 = np.array(f['ep_rewards']['mean'])
    max3 = np.array(f['ep_rewards']['max'])
    print(np.max(pos3),np.max(max3))
    fName = "../../log/Driving/time_log_2019-11-24 19_37_30.hdf5"
    print("ICM-TER")
    f = h5py.File(fName, 'r')
    pos5 = np.array(f['ep_rewards']['mean'])
    max5 = np.array(f['ep_rewards']['max'])
    print(np.max(pos5),np.max(max5))

    ep = range(pos1.shape[0])
    plt.figure()
    plt.plot(ep,pos1,ep,avg_pos0)
    plt.xticks(np.arange(0, pos1.shape[0], 50))
    plt.show()
