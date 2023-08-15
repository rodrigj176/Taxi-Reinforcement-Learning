# Taxi-Reinforcement-Learning
Implementation of reinforcement learning algorithms Q-learning and Sarsa on the OpenAI gym Taxi environment. 

How to run:

The following is the command to run the program with default settings. 

You must specify which algorithm you would like to run:

Q-learning:
python rfl.py ql 

SARSA:
python rfl.py sarsa

BOTH:
python rfl.py both

You can also run it with different max amounts of episodes and moves:

python rfl.py ql 3000 50

will run the q-learning algorithm with 3000 episodes and 50 max moves per episode.
