import numpy as np
import h5py
import matplotlib.pyplot as plt

if __name__ == '__main__':

    N = 100

    print("ROBO")
    print("NoRec")
    fName = "../../log/RoboCup/time_log_2020-05-04 17_44_50.hdf5"
    f = h5py.File(fName, 'r')
    pos1 = np.array(f['ep_pos_rewards']['mean'])
    avg_pos1 = np.convolve(pos1, np.ones((N,))/N, mode='valid')
    max1 = np.array(f['ep_pos_rewards']['max'])
    print(np.max(pos1),np.max(avg_pos1),np.max(max1))
    pos11 = np.array(f['ep_rewards']['mean'])
    avg_pos11 = np.convolve(pos11, np.ones((N,))/N, mode='valid')
    max11 = np.array(f['ep_rewards']['max'])
    print(np.max(pos11),np.max(avg_pos11),np.max(max11))

    print("ReconFalse")
    fName = "../../log/RoboCup/time_log_2020-05-04 17_44_59.hdf5"
    f = h5py.File(fName, 'r')
    pos0 = np.array(f['ep_pos_rewards']['mean'])
    avg_pos0 = np.convolve(pos0, np.ones((N,))/N, mode='valid')
    max0 = np.array(f['ep_pos_rewards']['max'])
    print(np.max(pos0),np.max(avg_pos0),np.max(max0))
    pos01 = np.array(f['ep_rewards']['mean'])
    avg_pos01 = np.convolve(pos01, np.ones((N,))/N, mode='valid')
    max01 = np.array(f['ep_rewards']['max'])
    print(np.max(pos01),np.max(avg_pos01),np.max(max01))

    print("ReconPret")
    fName = "../../log/RoboCup/time_log_2020-05-05 17_24_54.hdf5"
    f = h5py.File(fName, 'r')
    pos2 = np.array(f['ep_pos_rewards']['mean'])
    avg_pos2 = np.convolve(pos2, np.ones((N,))/N, mode='valid')
    max2 = np.array(f['ep_pos_rewards']['max'])
    print(np.max(pos2),np.max(avg_pos2),np.max(max2))
    pos21 = np.array(f['ep_rewards']['mean'])
    avg_pos21 = np.convolve(pos21, np.ones((N,))/N, mode='valid')
    max21 = np.array(f['ep_rewards']['max'])
    print(np.max(pos21),np.max(avg_pos21),np.max(max21))

    ep = range(avg_pos0.shape[0])
    plt.figure()
    plt.plot(ep,avg_pos0,ep,avg_pos1,ep,avg_pos2)
    plt.xticks(np.arange(0, pos0.shape[0], 50))
    plt.show()

    '''print("Drive")
    print("RCM")
    fName = "../../log/Driving/time_log_2019-12-02 14_21_17.hdf5"
    f = h5py.File(fName, 'r')
    pos0 = np.array(f['ep_rewards']['mean'])
    avg_pos0 = np.convolve(pos0, np.ones((N,))/N, mode='valid')
    max0 = np.array(f['ep_rewards']['max'])
    print(np.max(avg_pos0),np.max(pos0),np.max(max0))

    print("ICM")
    fName = "../../log/Driving/time_log_2019-12-04 09_17_56.hdf5"
    f = h5py.File(fName, 'r')
    pos1 = np.array(f['ep_rewards']['mean'])
    avg_pos1 = np.convolve(pos1, np.ones((N,))/N, mode='valid')
    max1 = np.array(f['ep_rewards']['max'])
    print(np.max(avg_pos1),np.max(pos1),np.max(max1))

    print("ICM-TER")
    fName = "../../log/Driving/time_log_2019-12-04 09_18_04.hdf5"
    f = h5py.File(fName, 'r')
    pos2 = np.array(f['ep_rewards']['mean'])
    avg_pos2 = np.convolve(pos2, np.ones((N,))/N, mode='valid')
    max2 = np.array(f['ep_rewards']['max'])
    print(np.max(avg_pos2),np.max(pos2),np.max(max2))

    ep = range(avg_pos0.shape[0])   
    plt.figure()
    plt.plot(ep,avg_pos0,ep,avg_pos1,ep,avg_pos2)
    plt.xticks(np.arange(0, avg_pos0.shape[0], 50))
    plt.show()'''
