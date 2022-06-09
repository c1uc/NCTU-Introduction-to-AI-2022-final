import matplotlib.pyplot as plt
import numpy as np
import os
import argparse


def initialize_plot():
    plt.figure(figsize=(10, 5))
    plt.title('BreakoutNoFrameskip-v4')
    plt.xlabel('epoch')
    plt.ylabel('rewards')


def DQN():
    DQN_Rewards = np.load("./Rewards/DQN_rewards.npy")
    DQN_test_Rewards = np.load("./Rewards/DQN_test_rewards.npy")
    avg = [sum([x for x in DQN_Rewards[0][i:i+100]]) / 100 for i in range(50000 - 100)]

    initialize_plot()

    plt.plot([i for i in range(50000 - 100)], avg,
             label='DQN', color='lightblue')
    plt.legend(loc="best")
    plt.savefig("./Graphs/DQN.png")
    plt.show()
    plt.close()

    initialize_plot()
    plt.plot([i for i in range(100)], DQN_test_Rewards, label='test', color='blue')
    plt.savefig("./Graphs/DQN_test.png")
    plt.show()
    plt.close()

def DDQN():
    DDQN_Rewards = np.load("./Rewards/DDQN_rewards.npy")
    DDQN_test_Rewards = np.load("./Rewards/DDQN_test_rewards.npy")
    avg = [sum([x for x in DDQN_Rewards[0][i:i+100]]) / 100 for i in range(50000 - 100)]

    initialize_plot()

    plt.plot([i for i in range(50000 - 100)], avg,
             label='DDQN', color='lightblue')
    plt.legend(loc="best")
    plt.savefig("./Graphs/DDQN.png")
    plt.show()
    plt.close()

    initialize_plot()
    plt.plot([i for i in range(100)], DDQN_test_Rewards, label='test', color='blue')
    plt.savefig("./Graphs/DDQN_test.png")
    plt.show()
    plt.close()


def compare():
    DQN_Rewards = np.load("./Rewards/DQN_rewards.npy")
    DQN_test_Rewards = np.load("./Rewards/DQN_test_rewards.npy")
    DQN_avg = [sum([x for x in DQN_Rewards[0][i:i + 100]]) / 100 for i in range(50000 - 100)]
    DDQN_Rewards = np.load("./Rewards/DDQN_rewards.npy")
    DDQN_test_Rewards = np.load("./Rewards/DDQN_test_rewards.npy")
    DDQN_avg = [sum([x for x in DDQN_Rewards[0][i:i + 100]]) / 100 for i in range(50000 - 100)]
    initialize_plot()
    plt.plot([i for i in range(50000 - 100)], DQN_avg, label='DQN', color='blue')
    plt.plot([i for i in range(50000 - 100)],
             DDQN_avg, label='DDQN', color='orange')
    plt.legend(loc="best")
    plt.savefig("./Graphs/compare.png")
    plt.show()
    plt.close()

    initialize_plot()
    plt.plot([i for i in range(100)], DQN_test_Rewards, label='DQN', color='blue')
    plt.plot([i for i in range(100)],
             DDQN_test_Rewards, label='DDQN', color='orange')
    plt.legend(loc="best")
    plt.savefig("./Graphs/compare_test.png")
    plt.show()
    plt.close()


if __name__ == "__main__":
    '''
    Plot the trend of Rewards
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--DQN", action="store_true")
    parser.add_argument("--DDQN", action="store_true")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    if not os.path.exists("./Graphs"):
        os.mkdir("./Graphs")

    if args.DQN:
        DQN()
    elif args.DDQN:
        DDQN()
    elif args.compare:
        compare()
