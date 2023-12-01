"""# script to visualize how agent plays a game
# useful to study different iterations

import numpy as np
from agent import DeepQLearningAgent, PolicyGradientAgent, \
        AdvantageActorCriticAgent, HamiltonianCycleAgent, BreadthFirstSearchAgent
from game_environment import Snake, SnakeNumpy
from utils import visualize_game
import json
# import keras.backend as K

# some global variables
version = 'v17.1'

with open('model_config/v17.1.json', 'r') as f:
    m = json.loads(f.read())
    board_size = m['board_size']
    frames = m['frames'] # keep frames >= 2
    max_time_limit = m['max_time_limit']
    supervised = bool(m['supervised'])
    n_actions = m['n_actions']
    obstacles = bool(m['obstacles'])

iteration_list = [163500]
max_time_limit = 398

# setup the environment
env = Snake(board_size=board_size, frames=frames, max_time_limit=max_time_limit,
            obstacles=obstacles, version=version)
s = env.reset()
n_actions = env.get_num_actions()

# setup the agent
# K.clear_session()
agent = DeepQLearningAgent(board_size=board_size, frames=frames,
                           n_actions=n_actions, buffer_size=10, version=version)
# agent = PolicyGradientAgent(board_size=board_size, frames=frames, n_actions=n_actions, buffer_size=10)
# agent = AdvantageActorCriticAgent(board_size=board_size, frames=frames, n_actions=n_actions, buffer_size=10)
# agent = HamiltonianCycleAgent(board_size=board_size, frames=frames, n_actions=n_actions, buffer_size=10)
# agent = BreadthFirstSearchAgent(board_size=board_size, frames=frames, n_actions=n_actions, buffer_size=10)

for iteration in iteration_list:
    agent.load_model(file_path='models/v17.1', iteration=iteration)

    for i in range(5):
        visualize_game(env, agent,
            path='images/game_visual_v17.1_{:d}_14_ob_{:d}.mp4'.format(iteration, i),
            debug=False, animate=True, fps=12)
"""

import numpy as np
from agent2 import DeepQLearningAgent, PolicyGradientAgent, \
    AdvantageActorCriticAgent, HamiltonianCycleAgent, BreadthFirstSearchAgent
from game_environment import Snake, SnakeNumpy
from utils import visualize_game
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from agent3 import DeepQLearningAgent  # Import your agent class
from game_environment import Snake  # Import your environment class
from utils import visualize_game, anim_frames_func
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import json
# import keras.backend as K


# ... (your existing code)

# some global variables
version = 'v17.1'

with open('model_config/v17.1.json', 'r') as f:
    m = json.loads(f.read())
    board_size = m['board_size']
    frames = m['frames']  # keep frames >= 2
    max_time_limit = m['max_time_limit']
    supervised = bool(m['supervised'])
    n_actions = m['n_actions']
    obstacles = bool(m['obstacles'])

iteration_list = [163500]
max_time_limit = 398

# setup the environment
env = Snake(board_size=board_size, frames=frames, max_time_limit=max_time_limit,
            obstacles=obstacles, version=version)
s = env.reset()
n_actions = env.get_num_actions()
color_map = {0: 'lightgray', 1: 'g', 2: 'lightgreen', 3: 'r', 4: 'darkgray'}


# setup the agent
# K.clear_session()
agent = DeepQLearningAgent(board_size=board_size, frames=frames,
                           n_actions=n_actions, buffer_size=10, version=version)
# agent = PolicyGradientAgent(board_size=board_size, frames=frames, n_actions=n_actions, buffer_size=10)
# agent = AdvantageActorCriticAgent(board_size=board_size, frames=frames, n_actions=n_actions, buffer_size=10)
# agent = HamiltonianCycleAgent(board_size=board_size, frames=frames, n_actions=n_actions, buffer_size=10)
# agent = BreadthFirstSearchAgent(board_size=board_size, frames=frames, n_actions=n_actions, buffer_size=10)

# Set the path for saving the video
video_path = 'videos/game_visual_v17.1_{:d}_14_ob_{:d}.mp4'

# Iterate over the selected iterations
for iteration in iteration_list:
    agent.load_model(file_path='models/v17.1', iteration=iteration)

    for i in range(5):
        # Collect game frames
        game_images = []
        s = env.reset()
        game_images.append([s[:, :, 0], 0])
        done = 0
        while not done:
            legal_moves = env.get_legal_moves()
            a = agent.move(s, legal_moves, env.get_values())
            next_s, _, done, _, _ = env.step(a)
            game_images.append([next_s[:, :, 0], env.get_time()])
            s = next_s.copy()

        # Append a few static frames in the end for a pause effect
        for _ in range(5):
            game_images.append(game_images[-1])

        # Plot the game and save it as a video
        fig, axs = plt.subplots(1, 1, figsize=(board_size // 2 + 1, board_size // 2 + 1))
        anim = animation.FuncAnimation(fig, anim_frames_func,
                                       frames=game_images,
                                       blit=False, interval=10,
                                       repeat=True, init_func=None,
                                       fargs=(axs, color_map, [], []))  # Adjust the last two arguments if needed

        # Save the animation as a video
        anim.save(video_path.format(iteration, i), writer=animation.writers['ffmpeg'](fps=12, metadata=dict(artist='Me'), bitrate=1800))

"""# script to visualize how agent plays a game
# useful to study different iterations

import numpy as np
from agent import DeepQLearningAgent, PolicyGradientAgent, AdvantageActorCriticAgent
from game_environment import Snake, SnakeNumpy
from utils import visualize_game
import json

# some global variables
version = 'v17.1'

with open('model_config/{:s}.json'.format(version), 'r') as f:
    m = json.loads(f.read())
    board_size = m['board_size']
    frames = m['frames']  # keep frames >= 2
    max_time_limit = m['max_time_limit']
    supervised = bool(m['supervised'])
    n_actions = m['n_actions']
    obstacles = bool(m['obstacles'])

iteration_list = [163500]
max_time_limit = 398

# setup the environment
env = Snake(board_size=board_size, frames=frames, max_time_limit=max_time_limit,
            obstacles=obstacles, version=version)
s = env.reset()
n_actions = env.get_num_actions()

# setup the agent
# K.clear_session()
agent = DeepQLearningAgent(board_size=board_size, frames=frames,
                           n_actions=n_actions, buffer_size=10, version=version)
# agent = PolicyGradientAgent(board_size=board_size, frames=frames, n_actions=n_actions, buffer_size=10)
# agent = AdvantageActorCriticAgent(board_size=board_size, frames=frames, n_actions=n_actions, buffer_size=10)


for iteration in iteration_list:
    agent.load_model(file_path='models/{:s}'.format(version), iteration=iteration)
    for i in range(5):
        visualize_game(env, agent,
                       path='images/game_visual_{:s}_{:d}_14_ob_{:d}.mp4'.format(version, iteration, i),
                       debug=False, animate=True, fps=12)"""