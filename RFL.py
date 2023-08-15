import Q_learning
import SARSA                                                                            
import numpy as np
import matplotlib.pyplot as plt
import sys

select = sys.argv[1]

try:
    total_episodes = int(sys.argv[2])
except:
    total_episodes = 10000

try:
    max_moves = int(sys.argv[3])
except:
    max_moves = 99


if select == 'ql': # For plotting Q-learning------------------------------------------------------------------------------------------------------------------------------

    q_episodes, q_rewards = Q_learning.q_learning(total_episodes,max_moves, True)

    avg_q_eps = []
    avg_q_rew = []
    count = 0
    avg_q = 0

    for i in range(len(q_episodes)):

        avg_q +=q_rewards[i]

        if count == 0:
            avg_q_eps.append(q_episodes[i])
            avg_q_rew.append(avg_q)


        if count == 100:
            count = 0
            avg_q = avg_q / 100
            avg_q_eps.append(q_episodes[i])
            avg_q_rew.append(avg_q)
            qf = avg_q
            avg_q = 0
            
        count+=1

    print("Q-learning average score of the last 100 episodes: ", qf)
    fig, axis = plt.subplots(1,2)
    fig.suptitle('Q-learning Rewards over Time')

    axis[0].plot(q_episodes,q_rewards)
    axis[0].set_title("Total rewards per episode")
    axis[1].plot(avg_q_eps,avg_q_rew)
    axis[1].set_title("Avg per 100 episodes")
    axis[0].set_xlabel("# of Trials")
    axis[0].set_ylabel("Rewards per trial")
    plt.show()



if select == 'sarsa': # For plotting SARSA------------------------------------------------------------------------------------------------------------------------------

    s_episodes, s_rewards = SARSA.sarsa(total_episodes,max_moves, True)

    avg_s_eps = []
    avg_s_rew = []
    count = 0
    avg_s = 0

    for i in range(len(s_episodes)):

        avg_s +=s_rewards[i]

        if count == 0:
            avg_s_eps.append(s_episodes[i])
            avg_s_rew.append(avg_s)


        if count == 100:
            count = 0
            avg_s = avg_s / 100
            avg_s_eps.append(s_episodes[i])
            avg_s_rew.append(avg_s)
            sf = avg_s
            avg_s = 0
        count+=1

    print("SARSA average score of the last 100 episodes: ", sf)    
    fig, axis = plt.subplots(1,2)
    fig.suptitle('SARSA Rewards over Time')

    axis[0].plot(s_episodes,s_rewards)
    axis[0].set_title("Total rewards per episode")
    axis[1].plot(avg_s_eps,avg_s_rew)
    axis[1].set_title("Avg per 100 episodes")
    axis[0].set_xlabel("# of Trials")
    axis[0].set_ylabel("Rewards per trial")
    plt.show()


if select == 'both': # For plotting Both------------------------------------------------------------------------------------------------------------------------------

    q_episodes, q_rewards = Q_learning.q_learning(total_episodes,max_moves, False)
    s_episodes, s_rewards = SARSA.sarsa(total_episodes,max_moves, False)

    avg_q_eps = []
    avg_q_rew = []
    count = 0
    avg_q = 0

    for i in range(len(q_episodes)):

        avg_q +=q_rewards[i]

        if count == 0:
            avg_q_eps.append(q_episodes[i])
            avg_q_rew.append(avg_q)


        if count == 100:
            count = 0
            avg_q = avg_q / 100
            avg_q_eps.append(q_episodes[i])
            avg_q_rew.append(avg_q)
            qf = avg_q
            avg_q = 0
            
        count+=1


    avg_s_eps = []
    avg_s_rew = []
    count = 0
    avg_s = 0

    for i in range(len(s_episodes)):

        avg_s +=s_rewards[i]

        if count == 0:
            avg_s_eps.append(s_episodes[i])
            avg_s_rew.append(avg_s)


        if count == 100:
            count = 0
            avg_s = avg_s / 100
            avg_s_eps.append(s_episodes[i])
            avg_s_rew.append(avg_s)
            sf = avg_s
            avg_s = 0
        count+=1

    print("Q-learning average score of the last 100 episodes: ", qf)    
    print("SARSA average score of the last 100 episodes: ", sf)    

    fig, axis = plt.subplots(1,2)
    fig.suptitle('Q-learning vs SARSA Rewards over Time')
   
    axis[0].plot(q_episodes,q_rewards, c='b', label='Q_learning')
    axis[0].plot(s_episodes,s_rewards, c='r', label='SARSA')
    axis[0].set_title("Total rewards per episode")
    axis[1].plot(avg_q_eps,avg_q_rew, c='b', label='Q_learning')
    axis[1].plot(avg_s_eps,avg_s_rew, c='r', label='SARSA')
    axis[1].set_title("Avg per 100 episodes")
    axis[0].set_xlabel("# of Trials")
    axis[0].set_ylabel("Rewards per trial")

    plt.legend(loc="lower right")
    
    plt.show()

