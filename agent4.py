"""
store all the agents here
"""
from charset_normalizer import models

import replay_buffer
from replay_buffer import ReplayBuffer, ReplayBufferNumpy
import numpy as np
import time
import pickle
from collections import deque
import json
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import tensorflow as tf


def huber_loss(y_true, y_pred, delta=1):
    """Keras implementation for huber loss
    loss = {
        0.5 * (y_true - y_pred)**2 if abs(y_true - y_pred) < delta
        delta * (abs(y_true - y_pred) - 0.5 * delta) otherwise
    }
    Parameters
    ----------
    y_true : Tensor
        The true values for the regression data
    y_pred : Tensor
        The predicted values for the regression data
    delta : float, optional
        The cutoff to decide whether to use quadratic or linear loss

    Returns
    -------
    loss : Tensor
        loss values for all points
    """

    error = y_true - y_pred
    quad_error = 0.5 * error ** 2
    lin_error = delta * (torch.abs(error) - 0.5 * delta)
    # Quadratic error if abs(error) < delta, linear error otherwise
    return torch.where(torch.abs(error) < delta, quad_error, lin_error)


def mean_huber_loss(y_true, y_pred, delta=1):
    """Calculates the mean value of huber loss

    Parameters
    ----------
    y_true : Tensor
        The true values for the regression data
    y_pred : Tensor
        The predicted values for the regression data
    delta : float, optional
        The cutoff to decide whether to use quadratic or linear loss

    Returns
    -------
    loss : Tensor
        average loss across points
     return torch.mean(huber_loss(y_true, y_pred, delta))"""
    loss = F.smooth_l1_loss(y_pred, y_true, reduction='mean', beta=delta)
    return loss


class Agent(nn.Module):
    """Base class for all agents
    This class extends to the following classes
    DeepQLearningAgent
    HamiltonianCycleAgent
    BreadthFirstSearchAgent

    Attributes
    ----------
    _board_size : int
        Size of board, keep greater than 6 for useful learning
        should be the same as the env board size
    _n_frames : int
        Total frames to keep in history when making prediction
        should be the same as env board size
    _buffer_size : int
        Size of the buffer, how many examples to keep in memory
        should be large for DQN
    _n_actions : int
        Total actions available in the env, should be same as env
    _gamma : float
        Reward discounting to use for future rewards, useful in policy
        gradient, keep < 1 for convergence
    _use_target_net : bool
        If use a target network to calculate next state Q values,
        necessary to stabilise DQN learning
    _input_shape : tuple
        Tuple to store individual state shapes
    _board_grid : Numpy array
        A square filled with values from 0 to board size **2,
        Useful when converting between row, col and int representation
    _version : str
        model version string
    """

    def __init__(self, board_size=10, frames=2, buffer_size=10000, gamma=0.99, n_actions=3, use_target_net=True,
                 max_size=10000,
                 version='v17.1'):
        """ initialize the agent

        Parameters
        ----------
        board_size : int, optional
            The env board size, keep > 6
        frames : int, optional
            The env frame count to keep old frames in state
        buffer_size : int, optional
            Size of the buffer, keep large for DQN
        gamma : float, optional
            Agent's discount factor, keep < 1 for convergence
        n_actions : int, optional
            Count of actions available in env
        use_target_net : bool, optional
            Whether to use target network, necessary for DQN convergence
            version : str, optional except NN based models
            path to the model architecture json
        """
        super().__init__()
        self._board_size = board_size
        self._n_frames = frames
        self._buffer_size = buffer_size
        self._n_actions = n_actions
        self._current_size = 0
        self._current_index = 0
        self._max_size = max_size
        self._gamma = gamma
        self._use_target_net = use_target_net
        self._input_shape = (self._board_size, self._board_size, self._n_frames)
        # reset buffer also initializes the buffer
        self.reset_buffer()
        self._board_grid = np.arange(0, self._board_size ** 2) \
            .reshape(self._board_size, -1)
        self._version = version

    def get_gamma(self):
        """Returns the agent's gamma value

        Returns
        -------
        _gamma : float
            Agent's gamma value
        """
        return self._gamma

    def reset_buffer(self, buffer_size=None):
        """Reset current buffer

        Parameters
        ----------
        buffer_size : int, optional
            Initialize the buffer with buffer_size, if not supplied,
            use the original value
        """
        if (buffer_size is not None):
            self._buffer_size = buffer_size
        self._buffer = ReplayBuffer(self._buffer_size, self._board_size,
                                    self._n_frames, self._n_actions)

    def get_buffer_size(self):
        """Get the current buffer size

        Returns
        -------
        buffer size : int
            Current size of the buffer
        """
        return self._buffer.get_current_size()

    def add_to_buffer(self, board, action, reward, next_board, done, legal_moves):
        """Add current game step to the replay buffer

        Parameters
        ----------
        board : Numpy array
            Current state of the board, can contain multiple games
        action : Numpy array or int
            Action that was taken, can contain actions for multiple games
        reward : Numpy array or int
            Reward value(s) for the current action on current states
        next_board : Numpy array
            State obtained after executing action on current state
        done : Numpy array or int
            Binary indicator for game termination
        legal_moves : Numpy array
            Binary indicators for actions which are allowed at next states
"""
        """self._buffer.add_to_buffer(board, action, reward, next_board,
                                   done, legal_moves)"""
        """Add current game step to the replay buffer."""
        """Add current game step to the replay buffer.        if self._buffer_size < self.get_buffer_size():
            # Convert legal_moves to a one-hot encoded array
            legal_moves_one_hot = np.zeros((legal_moves.shape[0], self._n_actions), dtype=np.float32)
            legal_moves_one_hot[np.arange(legal_moves.shape[0]), legal_moves[:, 0]] = 1
        else:
            # If the buffer is full, overwrite the oldest entry
            legal_moves_one_hot = np.zeros((legal_moves.shape[0], self._n_actions), dtype=np.float32)
            legal_moves_one_hot[np.arange(legal_moves.shape[0]), legal_moves[:, 0]] = 1

        idx = self._current_index % self._max_size
        # Extract a specific board, action, reward, and next_board from the larger arrays
        self._board[idx] = board[0]  # Assuming you want the first board from the batch
        self._action[idx] = action[0] if len(action) > 0 else 0  # Assuming you want the first value from action
        self._reward[idx] = reward[0] if len(reward) > 0 else 0  # Assuming you want the first value from reward
        self._next_board[idx] = next_board[0]  # Assuming you want the first next_board from the batch
        self._done[idx] = bool(done[0])  # Assuming you want the first value from done
        self._legal_moves[idx] = legal_moves_one_hot[0]  # Assuming you want the first legal_moves from the batch

        if self._buffer_size < self.get_buffer_size():
            self._current_size += 1
        self._current_index += 1"""
        if self._buffer_size < self.get_buffer_size():
            # Convert legal_moves to a one-hot encoded array
            legal_moves_one_hot = np.zeros((legal_moves.shape[0], self._n_actions), dtype=np.float32)
            legal_moves_one_hot[np.arange(legal_moves.shape[0]), legal_moves[:, 0]] = 1
        else:
            # If the buffer is full, overwrite the oldest entry
            legal_moves_one_hot = np.zeros((legal_moves.shape[0], self._n_actions), dtype=np.float32)
            legal_moves_one_hot[np.arange(legal_moves.shape[0]), legal_moves[:, 0]] = 1

        idx = self._current_index % self._max_size
        # Extract a specific board, action, reward, and next_board from the larger arrays
        self._board[idx] = board[0]  # Assuming you want the first board from the batch
        self._action[idx] = action[0] if len(action) > 0 else 0  # Assuming you want the first value from action
        self._reward[idx] = reward[0] if len(reward) > 0 else 0  # Assuming you want the first value from reward
        self._next_board[idx] = next_board[0]  # Assuming you want the first next_board from the batch
        self._done[idx] = bool(done[0])  # Assuming you want the first value from done
        self._legal_moves[idx] = legal_moves_one_hot[0]  # Assuming you want the first legal_moves from the batch

        if self._buffer_size < self.get_buffer_size():
            self._current_size += 1
        self._current_index += 1

    def save_buffer(self, file_path='', iteration=None):
        """Save the buffer to disk

        Parameters
        ----------
        file_path : str, optional
            The location to save the buffer at
        iteration : int, optional
            Iteration number to tag the file name with, if None, iteration is 0
        """
        if (iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        with open("{}/buffer_{:04d}".format(file_path, iteration), 'wb') as f:
            pickle.dump(self._buffer, f)

    def load_buffer(self, file_path='', iteration=None):
        """Load the buffer from disk

        Parameters
        ----------
        file_path : str, optional
            Disk location to fetch the buffer from
        iteration : int, optional
            Iteration number to use in case the file has been tagged
            with one, 0 if iteration is None

        Raises
        ------
        FileNotFoundError
            If the requested file could not be located on the disk
        """
        if (iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        with open("{}/buffer_{:04d}".format(file_path, iteration), 'rb') as f:
            self._buffer = pickle.load(f)

    def _point_to_row_col(self, point):
        """Covert a point value to row, col value
        point value is the array index when it is flattened

        Parameters
        ----------
        point : int
            The point to convert

        Returns
        -------
        (row, col) : tuple
            Row and column values for the point
        """
        return (point // self._board_size, point % self._board_size)

    def _row_col_to_point(self, row, col):
        """Covert a (row, col) to value
        point value is the array index when it is flattened

        Parameters
        ----------
        row : int
            The row number in array
        col : int
            The column number in array
        Returns
        -------
        point : int
            point value corresponding to the row and col values
        """
        return row * self._board_size + col


class DeepQLearningAgent(Agent):
    """This agent learns the game via Q learning
    model outputs everywhere refers to Q values
    This class extends to the following classes
    PolicyGradientAgent
    AdvantageActorCriticAgent

    Attributes
    ----------
    _model : TensorFlow Graph
        Stores the graph of the DQN model
    _target_net : TensorFlow Graph
        Stores the target network graph of the DQN model


    def __init__(self, board_size=10, frames=4, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=True,
                 version=''):
        Initializer for DQN agent, arguments are same as Agent class
        except use_target_net is by default True and we call and additional
        reset models method to initialize the DQN networks

        Agent.__init__(self, board_size=board_size, frames=frames, buffer_size=buffer_size,
                       gamma=gamma, n_actions=n_actions, use_target_net=use_target_net,
                       version=version)
        self.reset_models()"""

    def __init__(self, board_size=10, frames=4, n_actions=3, buffer_size=10000, gamma=0.99, use_target_net=True,
                 version='v17.1'):

        super(DeepQLearningAgent, self).__init__()

        # Your neural network layers here
        self.conv1 = nn.Conv2d(10, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(64 * board_size * board_size, 64)
        # self.fc1 = nn.Linear(640, 64)
        self.fc1 = nn.Linear(640, 64)
        self.fc2 = nn.Linear(64, n_actions)

        # Other attributes
        self._board_size = board_size
        self._frames = frames
        self._n_actions = n_actions
        self._buffer_size = buffer_size
        self._gamma = gamma
        # self._target_net = None
        self._use_target_net = use_target_net
        self._version = version
        # self._model = self._agent_model()

        # Initialize the optimizer
        self.optimizer = optim.RMSprop(self.parameters(), lr=0.0005)

        # New attributes for replay buffer
        self._buffer = None

        # Call the reset_models method to initialize the DQN networks
        # self.reset_models()

        # New attributes for replay buffer
        self._board = np.zeros((buffer_size, board_size, board_size, frames), dtype=np.float32)
        self._action = np.zeros(buffer_size, dtype=np.int32)
        self._reward = np.zeros(buffer_size, dtype=np.float32)
        self._next_board = np.zeros((buffer_size, board_size, board_size, frames), dtype=np.float32)
        self._done = np.zeros(buffer_size, dtype=np.float32)
        self._legal_moves = np.zeros((buffer_size, n_actions), dtype=np.float32)
        self._max_size = buffer_size
        self._current_size = 0
        self._current_index = 0
        self._buffer = replay_buffer.ReplayBufferNumpy(buffer_size, board_size, frames, n_actions)

    def forward(self, x):
        # print("Input Shape:", x.shape)
        x = F.relu(self.conv1(x))
        # print("After Conv1 Shape:", x.shape)
        x = F.relu(self.conv2(x))
        # print("After Conv2 Shape:", x.shape)
        x = self.flatten(x)  # Add this line to flatten the output
        # print("After Flatten Shape:", x.shape)
        x = F.relu(self.fc1(x))
        # print("After FC1 Shape:", x.shape)
        x = self.fc2(x)
        # print("After FC2 Shape:", x.shape)
        return x

    def parameters(self):
        return list(self.conv1.parameters()) + list(self.conv2.parameters()) + list(self.conv3.parameters()) + \
            list(self.fc1.parameters()) + list(self.fc2.parameters())

    def reset_models(self):
        """ Reset all the models by creating new graphs"""
        self._model = self._agent_model()
        if (self._use_target_net):
            self._target_net = self._agent_model()
            self.update_target_net()

        """self._buffer = ReplayBufferNumpy(self._buffer_size, self._board_size, self._frames, self._n_actions)
        self._model = self._create_model()
        if self._use_target_net:
            self._target_net = self._create_model()
            self.update_target_net()"""

    """def _create_model(self):
        model = DeepQLearningAgent(self._board_size, self._frames, self._n_actions, self._buffer_size, self._gamma)
        optimizer = optim.RMSprop(model.parameters(), lr=0.0005)
        return model
        return DeepQLearningAgent(self._board_size, self._frames, self._n_actions, self._buffer_size, self._gamma,
                                  False, self._version)"""

    def _init_weights(self, layer):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _prepare_input(self, board):
        """Reshape input and normalize

            Parameters
            ----------
            board : PyTorch tensor
                The board state to process

            Returns
            -------
            board : PyTorch tensor
                Processed and normalized board
            if len(board.shape) == 3:
            # Assuming the shape is [height, width, channels], change it to [channels, height, width]
            board = np.transpose(board, (2, 0, 1))
            board = board.reshape(1, *board.shape)  # Add batch dimension

        board = self._normalize_board(board)
        return board.clone()        if len(board.shape) == 3:
            board = board.unsqueeze(0)
        board = self._normalize_board(board)
        return board.clone()"""
        if len(board.shape) == 3:
            # Assuming the shape is [height, width, channels], change it to [channels, height, width]
            board = np.transpose(board, (2, 0, 1))
            board = board.reshape(1, *board.shape)  # Add batch dimension

        board = self._normalize_board(board)
        return board.clone()

    def _get_model_outputs(self, board, model=None):
        """Get action values from the DQN model

        Parameters
        ----------
        board : Numpy array
            The board state for which to predict action values
        model : TensorFlow Graph, optional
            The graph to use for prediction, model or target network

        Returns
        -------
        model_outputs : Numpy array
            Predicted model outputs on board,
            of shape board.shape[0] * num actions

         # to correct dimensions and normalize
        board = self._prepare_input(board)
        # the default model to use
        if model is None:
            model = self._model
        model_outputs = model.predict_on_batch(board)

        return model_outputs"""

        # to correct dimensions and normalize
        board = self._prepare_input(board)
        # the default model to use
        if model is None:
            model = self._agent_model()

        # Use the PyTorch model for prediction
        model_outputs = model(board)

        # Convert the PyTorch tensor to a NumPy array
        model_outputs = model_outputs.detach().numpy()

        return model_outputs

    def _normalize_board(self, board):
        """Normalize the board before input to the network

        Parameters
        ----------
        board : Numpy array
            The board state to normalize

        Returns
        -------
        board : Numpy array
            The copy of board state after normalization
        """
        # return board.copy()
        # return((board/128.0 - 1).copy())
        # return board.astype(np.float32) / 4.0
        # Convert NumPy array to PyTorch tensor and perform normalization
        return torch.tensor(board, dtype=torch.float32) / 4.0

    def move(self, board, legal_moves, value=None):
        """Get the action with maximum Q value

        Parameters
        ----------
        board : Numpy array
            The board state on which to calculate best action
        value : None, optional
            Kept for consistency with other agent classes

        Returns
        -------
        output : Numpy array
            Selected action using the argmax function
        # use the agent model to make the predictions
        model_outputs = self._get_model_outputs(board, self._agent_model())
        return np.argmax(np.where(legal_moves == 1, model_outputs, -np.inf), axis=1)"""
        # use the agent model to make the predictions
        model_outputs = self._get_model_outputs(board, self._agent_model())

        # Flatten the model_outputs to have the same shape as legal_moves
        flat_model_outputs = model_outputs.reshape(model_outputs.shape[0], -1)

        # Use np.argmax with flattened model_outputs
        action = np.argmax(np.where(legal_moves == 1, flat_model_outputs, -np.inf), axis=1)

        return action


    def _agent_model(self):
        """Returns the model which evaluates Q values for a given state input

        Returns
        -------
        model : TensorFlow Graph
            DQN model graph

        # define the input layer; shape is dependent on the board size and frames
        with open('model_config/{:s}.json'.format(self._version), 'r') as f:
            m = json.loads(f.read())

        input_board = Input((self._board_size, self._board_size, self._n_frames,), name='input')
        x = input_board
        for layer in m['model']:
            l = m['model'][layer]
            if ('Conv2D' in layer):
                # add convolutional layer
                x = Conv2D(**l)(x)
            if ('Flatten' in layer):
                x = Flatten()(x)
            if ('Dense' in layer):
                x = Dense(**l)(x)
        out = Dense(self._n_actions, activation='linear', name='action_values')(x)
        model = Model(inputs=input_board, outputs=out)
        model.compile(optimizer=RMSprop(0.0005), loss=mean_huber_loss)
        input_board = Input((self._board_size, self._board_size, self._n_frames,), name='input')
        x = Conv2D(16, (3,3), activation='relu', data_format='channels_last')(input_board)
        x = Conv2D(32, (3,3), activation='relu', data_format='channels_last')(x)
        x = Conv2D(64, (6,6), activation='relu', data_format='channels_last')(x)
        x = Flatten()(x)
        x = Dense(64, activation = 'relu', name='action_prev_dense')(x)
        # this layer contains the final output values, activation is linear since
        # the loss used is huber or mse
        out = Dense(self._n_actions, activation='linear', name='action_values')(x)
        # compile the model
        model = Model(inputs=input_board, outputs=out)
        model.compile(optimizer=RMSprop(0.0005), loss=mean_huber_loss)
        # model.compile(optimizer=RMSprop(0.0005), loss='mean_squared_error')
        # Load model configuration from JSON
        with open('model_versions.json'.format(self._version), 'r') as f:
            model_config = json.loads(f.read())

        # Assuming "model_versions" contains the version-specific configurations
        version_config = model_config.get("model_versions", {}).get(self._version, {})

        # Extract relevant configuration parameters
        input_shape = version_config.get("input_shape", [10, 10, 2])  # Adjust based on your input shape
        output_size = version_config.get("output_size", 3)  # Adjust based on your number of actions

        # Instantiate the model (using PyTorch)
        # model = DeepQLearningAgentModel(input_shape, output_size)
        model = DeepQLearningAgent(self._board_size, self._frames, self._n_actions)
        optimizer = optim.RMSprop(model.parameters(), lr=0.0005)

        return model"""

        self.model = DeepQLearningAgentNet(version='v17.1', frames=self._n_frames, n_actions=self._n_actions,
                                           board_size=self._board_size, buffer_size=self._buffer_size,
                                           gamma=self._gamma, use_target_net=self._use_target_net)
        return self.model

    def set_weights_trainable(self):
        """Set selected layers to non trainable and compile the model"""
        for layer in self._model.layers:
            layer.trainable = False
        # the last dense layers should be trainable
        for s in ['action_prev_dense', 'action_values']:
            self._model.get_layer(s).trainable = True
        self._model.compile(optimizer=self._model.optimizer,
                            loss=self._model.loss)

    def get_action_proba(self, board, values=None):
        """Returns the action probability values using the DQN model

        Parameters
        ----------
        board : Numpy array
            Board state on which to calculate action probabilities
        values : None, optional
            Kept for consistency with other agent classes

        Returns
        -------
        model_outputs : Numpy array
            Action probabilities, shape is board.shape[0] * n_actions
        """
        model_outputs = self._get_model_outputs(board, self._model)
        # subtracting max and taking softmax does not change output
        # do this for numerical stability
        model_outputs = np.clip(model_outputs, -10, 10)
        model_outputs = model_outputs - model_outputs.max(axis=1).reshape((-1, 1))
        model_outputs = np.exp(model_outputs)
        model_outputs = model_outputs / model_outputs.sum(axis=1).reshape((-1, 1))
        return model_outputs

    def save_model(self, file_path='', iteration=None):
        """Save the current models to disk using tensorflow's
        inbuilt save model function (saves in h5 format)
        saving weights instead of model as cannot load compiled
        model with any kind of custom object (loss or metric)

        Parameters
        ----------
        file_path : str, optional
            Path where to save the file
        iteration : int, optional
            Iteration number to tag the file name with, if None, iteration is 0
        """
        if (iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        self._model.save_weights("{}/model_{:04d}.h5".format(file_path, iteration))
        if (self._use_target_net):
            self._target_net.save_weights("{}/model_{:04d}_target.h5".format(file_path, iteration))

    def load_model(self, file_path='', iteration=None):
        """ load any existing models, if available """
        """Load models from disk using tensorflow's
        inbuilt load model function (model saved in h5 format)

        Parameters
        ----------
        file_path : str, optional
            Path where to find the file
        iteration : int, optional
            Iteration number the file is tagged with, if None, iteration is 0

        Raises
        ------
        FileNotFoundError
            The file is not loaded if not found and an error message is printed,
            this error does not affect the functioning of the program
                if (iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        self._model.load_weights("{}/model_{:04d}.h5".format(file_path, iteration))
        if (self._use_target_net):
            self._target_net.load_weights("{}/model_{:04d}_target.h5".format(file_path, iteration))
        # print("Couldn't locate models at {}, check provided path".format(file_path))"""
        if (iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
            # before: self._model.load_weights("{}/model_{:04d}.h5".format(file_path, itera tion))
        self._model.load_state_dict((torch.load("{}/model_{:04d}.h5".format(file_path, iteration))))

        if (self._use_target_net):
            # before: self._target_net.load_weights("{}/model_{:04d}_target.h5".format(file_path, iteration))
            self._target_net.load_state_dict(torch.load("{}/model_{:04d}_target.h5".format(file_path, iteration)))
        # print("Couldn't locate models at {}, check provided path".format(file_path))

    def print_models(self):
        """Print the current models using summary method"""
        print('Training Model')
        print(self._model.summary())
        if (self._use_target_net):
            print('Target Network')
            print(self._target_net.summary())

    def train_agent(self, batch_size=32, num_games=1, reward_clip=False):
        """Train the model by sampling from buffer and return the error.
        We are predicting the expected future discounted reward for all
        actions with our model. The target for training the model is calculated
        in two parts:
        1) dicounted reward = current reward +
                        (max possible reward in next state) * gamma
           the next reward component is calculated using the predictions
           of the target network (for stability)
        2) rewards for only the action take are compared, hence while
           calculating the target, set target value for all other actions
           the same as the model predictions

        Parameters
        ----------
        batch_size : int, optional
            The number of examples to sample from buffer, should be small
        num_games : int, optional
            Not used here, kept for consistency with other agents
        reward_clip : bool, optional
            Whether to clip the rewards using the numpy sign command
            rewards > 0 -> 1, rewards <0 -> -1, rewards == 0 remain same
            this setting can alter the learned behaviour of the agent

        Returns
        -------
            loss : float
            The current error (error metric is defined in reset_models)
        """
        """s, a, r, next_s, done, legal_moves = self._buffer.sample(batch_size)
        if (reward_clip):
            r = np.sign(r)
        # calculate the discounted reward, and then train accordingly
        current_model = self._target_net if self._use_target_net else self._model
        next_model_outputs = self._get_model_outputs(next_s, current_model)
        # our estimate of expexted future discounted reward
        discounted_reward = r + \
                            (self._gamma * np.max(np.where(legal_moves == 1, next_model_outputs, -np.inf),
                                                  axis=1) \
                             .reshape(-1, 1)) * (1 - done)
        # create the target variable, only the column with action has different value
        target = self._get_model_outputs(s)
        # we bother only with the difference in reward estimate at the selected action
        target = (1 - a) * target + a * discounted_reward
        # fit
        loss = self._model.train_on_batch(self._normalize_board(s), target)
        # loss = round(loss, 5)
        return loss"""

        s, a, r, next_s, done, legal_moves = self._buffer.sample(batch_size)
        if reward_clip:
            r = torch.sign(torch.tensor(r, dtype=torch.float32)).numpy()

        legal_moves = torch.tensor(legal_moves, dtype=torch.float32)
        current_model = self

        # Convert to PyTorch tensors
        s_tensor = torch.tensor(s, dtype=torch.float32)
        next_s_tensor = torch.tensor(next_s, dtype=torch.float32)

        # Forward pass
        predictions = current_model(s_tensor)

        # Calculate loss
        # target = self._calculate_target(current_model, next_s_tensor, done, legal_moves, r)
        loss = huber_loss(torch.tensor(r, dtype=torch.float32), predictions, delta=1.0)

        # Backward pass and optimization
        self.optimizer.zero_grad()
        # loss.backward()
        self.optimizer.step()

        # Return loss
        return loss

    """def _calculate_target(self, model, next_s, done, legal_moves, r):
        # Calculate the target for training the model
        with torch.no_grad():
            next_model_outputs = model(next_s)
            discounted_reward_component = (
                                                  self._gamma * torch.max(
                                              torch.where(legal_moves == 1, next_model_outputs,
                                                          torch.tensor(-np.inf, dtype=torch.float32)),
                                              dim=1
                                          ).values
                                          ).view(-1, 1) * (1 - done)

            # Expand discounted_reward_component to match the number of actions
            discounted_reward_component = discounted_reward_component.expand(-1, next_model_outputs.shape[1])

            # Overwrite the relevant columns of the target with discounted_reward_component
            target = model(s).clone().detach()
            target[:, :discounted_reward_component.shape[1]] = discounted_reward_component

            # We bother only with the difference in reward estimate at the selected action
            target = (1 - a) * target + a * discounted_reward_component

        return target"""

    def update_target_net(self):
        """Update the weights of the target network, which is kept
        static for a few iterations to stabilize the other network.
        This should not be updated very frequently
        self._target_net.load_state_dict(self._model.state_dict())"""
        if (self._use_target_net):
            self._target_net.load_state_dict(self._model.state_dict())

    def compare_weights(self):
        """Simple utility function to heck if the model and target
        network have the same weights or not
        """
        for i in range(len(self._model.layers)):
            for j in range(len(self._model.layers[i].weights)):
                c = (self._model.layers[i].weights[j].numpy() == \
                     self._target_net.layers[i].weights[j].numpy()).all()
                print('Layer {:d} Weights {:d} Match : {:d}'.format(i, j, int(c)))

    def copy_weights_from_agent(self, agent_for_copy):
        """Update weights between competing agents which can be used
        in parallel training
        """
        assert isinstance(agent_for_copy, self), "Agent type is required for copy"

        self._model.set_weights(agent_for_copy._model.get_weights())
        self._target_net.set_weights(agent_for_copy._model_pred.get_weights())


class DeepQLearningAgentModel(nn.Module):
    """PyTorch model for DeepQLearningAgent"""

    def __init__(self, input_shape, output_size):
        super(DeepQLearningAgentModel, self).__init__()

        # Define your neural network layers here
        self.conv1 = nn.Conv2d(10, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        # print("Flatten Shape:", self.flatten)
        self.fc1 = nn.Linear(64, 64)
        # print("FC1 Shape:", self.fc1)
        self.fc2 = nn.Linear(64, output_size)
        # print("FC2 Shape:", self.fc2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def parameters(self):
        return list(self.conv1.parameters()) + list(self.conv2.parameters()) + \
            list(self.conv3.parameters()) + list(self.fc1.parameters()) + list(self.fc2.parameters())


class DeepQLearningAgentNet(nn.Module):
    def __init__(self, version='', frames=4, board_size=10, n_actions=3, buffer_size=10000,
                 gamma=0.99, use_target_net=True) -> None:
        super(DeepQLearningAgentNet, self).__init__()
        self.version = version
        self._n_frames = frames
        self._n_actions = n_actions
        self._board_size = board_size
        # define the input layer, shape is dependent on the board size and frames
        with open('model_config/{:s}.json'.format(self.version), 'r') as f:
            m = json.loads(f.read())
        out_channels_prev = m['frames']
        layers = []
        for layer in m['model']:
            l = m['model'][layer]
            if ('Conv2D' in layer):
                # add convolutional layer
                # x = Conv2D(**l)(x)
                if "padding" in l:  # Check if "padding is in the dict
                    padding = l["padding"]
                    layers.append(
                        nn.Conv2d(in_channels=out_channels_prev, out_channels=l["filters"],
                                  kernel_size=l["kernel_size"],
                                  padding=padding))
                else:
                    layers.append(
                        nn.Conv2d(in_channels=out_channels_prev, out_channels=l["filters"],
                                  kernel_size=l["kernel_size"]))
                    # padding = "valid"  # valid= the same as no padding
                if "activation" in l:  # check if  relu activation in l
                    if l['activation'] == "relu":
                        layers.append(nn.ReLU())
                out_channels_prev = l["filters"]  # variable to store how many out channels, for the next in_channels
            if 'Flatten' in layer:
                # x = Flatten()(x)
                layers.append(nn.Flatten())
                out_channels_prev = 64 * 4 * 4
            if 'Dense' in layer:
                # x = Dense(**l)(x)
                layers.append(nn.Linear(out_channels_prev, l['units']))
                if "activation" in l:  # check if activation
                    if l['activation'] == "relu":
                        layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)
        self.out = nn.Linear(64, self._n_actions)
        self.criterion = mean_huber_loss
        self.optimizer = optim.RMSprop(self.parameters(), lr=0.0005)
        # self.to('cuda')

    def forward(self, x: torch.Tensor):
        # x.to('cuda')
        # x = self.conv(x)
        output = self.out(x)
        return output
