import gym
import random                                                                               
import numpy as np
import matplotlib.pyplot as plt
import time


def q_learning(total_ep, max_moves, to_show):


    env = gym.make('Taxi-v3')
    env.reset()
    #env.render()

    action_space_size = env.action_space.n                      #Actions:  0 - South move, 1 - North move, 2 - East move, 3 - West Move                 
    state_space_size = env.observation_space.n                  #           4 - Pickup,     5 - Dropoff 

    curr_ep = 0

    learning_rate = 0.3  #defined to be aplha
    discount_factor = 0.9  #defined to be gamma

    e = 1.00
    min_e = 0.01

    q_table = np.zeros((state_space_size, action_space_size))


    rewards = []
    episodes = []



    for _ in range(total_ep):
    
        state = env.reset()
        game_success = False
        total_reward = 0
        
        for moves in range(max_moves):
            
                if curr_ep == total_ep-1 and to_show:       # Block to show the agent running on the final iteration
                    env.render()
                    print('Current Score:', total_reward)
                    time.sleep(0.75)

                random_eps = random.uniform(0,1)

                if random_eps > e  :                                             #If random_eps > eps then we exploit q-table (choose largest value)
                    action = np.argmax(q_table[state,:])

                else:
                    action = env.action_space.sample()                          # Else, we take a random step, aka explore

                next_state, reward, game_success, prob = env.step(action)       #Step returns : Position of next state, reward value, game is complete, adittional info

                total_reward += reward

            #   new Q(S, A) = Q(S, A) + α (reward + γ * (Max Q'(S', A') - Q(S, A))
                q_table[state, action] = q_table[state,action] + learning_rate * (reward + discount_factor * (np.max(q_table[next_state,:]) ) -q_table[state, action] )

                state = next_state
                if game_success == True:
                    if curr_ep == total_ep-1 and to_show:
                        env.render()
                        print("Game completed successfully with a final score of: ",total_reward)
                        time.sleep(2)
                    break

        rewards.append(total_reward)
        episodes.append(curr_ep)       
        curr_ep += 1

        #Episolon decay function: Eps starts out at max to promote exploration, then quickly decays to promote q-table exploitation 
        e = e*0.95
        if e < min_e:
            e = min_e

    env.close()
    return(episodes, rewards)
   