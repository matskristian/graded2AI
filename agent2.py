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

"""from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Softmax, MaxPool2D
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2"""


# from tensorflow.keras.losses import Huber

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
     # Use TensorFlow session to evaluate symbolic tensors
    with tf.compat.v1.Session() as sess:
        y_true_np = sess.run(y_true)
        y_pred_np = sess.run(y_pred)

    # Convert NumPy arrays to PyTorch tensors
    y_true_torch = torch.tensor(y_true_np, dtype=torch.float32)
    y_pred_torch = torch.tensor(y_pred_np, dtype=torch.float32)

    error = y_true_torch - y_pred_torch
    quad_error = 0.5 * error ** 2
    lin_error = delta * (torch.abs(error) - 0.5 * delta)
    # quadratic error, linear error
    return torch.where(torch.abs(error) < delta, quad_error, lin_error)"""

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
                 version='v17.1', *args, **kwargs):
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
        super().__init__(*args, **kwargs)
        self._buffer = None
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
        if buffer_size is not None:
            self._buffer_size = buffer_size
        self._buffer = ReplayBufferNumpy(self._buffer_size, self._board_size,
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
        """Add current game step to the replay buffer."""
        """if self._buffer_size < self.get_buffer_size():
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
        """if self._buffer_size < self.get_buffer_size():
            # Convert legal_moves to a one-hot encoded tensor
            legal_moves_one_hot = np.zeros((legal_moves.shape[0], self._n_actions), dtype=np.float32)

            legal_moves_one_hot[np.arange(legal_moves.shape[0]), legal_moves.astype(int)] = 1


        else:
            # If the buffer is full, overwrite the oldest entry
            legal_moves_one_hot = torch.zeros((legal_moves.shape[0], self._n_actions), dtype=torch.float32)
            legal_moves_one_hot[np.arange(legal_moves.shape[0]), legal_moves.astype(int)] = 1



        idx = self._current_index % self._max_size
        # Convert everything to PyTorch tensors
        self._board[idx] = torch.tensor(board[0])  # Assuming you want the first board from the batch
        self._action[idx] = torch.tensor(action[0]) if len(
            action) > 0 else 0  # Assuming you want the first value from action
        self._reward[idx] = torch.tensor(reward[0]) if len(
            reward) > 0 else 0  # Assuming you want the first value from reward
        self._next_board[idx] = torch.tensor(next_board[0])  # Assuming you want the first next_board from the batch
        self._done[idx] = bool(done[0])  # Assuming you want the first value from done
        self._legal_moves[idx] = legal_moves_one_hot[0]  # Assuming you want the first legal_moves from the batch

        if self._buffer_size < self.get_buffer_size():
            self._current_size += 1
        self._current_index += 1"""
        # Convert board and next_board to PyTorch tensors
        board = torch.tensor(board, dtype=torch.float32)
        next_board = torch.tensor(next_board, dtype=torch.float32)

        # Ensure legal_moves is a NumPy array
        legal_moves = np.array(legal_moves, dtype=np.float32)

        # Ensure legal_moves_one_hot has the correct shape [batch_size, n_actions]
        legal_moves_one_hot = np.zeros((legal_moves.shape[0], self._n_actions), dtype=np.float32)

        # Convert legal_moves to integer and use it for one-hot encoding
        legal_moves_indices = legal_moves.astype(int)

        # Check if legal_moves_indices has the correct shape
        if legal_moves_indices.ndim == 1:
            legal_moves_indices = legal_moves_indices.reshape(-1, 1)

        legal_moves_one_hot[np.arange(legal_moves.shape[0]), legal_moves_indices.flatten()] = 1

        # Add the transition to the replay buffer
        self._buffer.add_transition(board, action, reward, next_board, done, legal_moves_one_hot)


    def save_buffer(self, file_path='', iteration=None):
        """Save the buffer to disk

        Parameters
        ----------
        file_path : str, optional
            The location to save the buffer at
        iteration : int, optional
            Iteration number to tag the file name with, if None, iteration is 0
        """
        if iteration is not None:
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
        if iteration is not None:
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
        return point // self._board_size, point % self._board_size

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


    Def __init__(self, board_size=10, frames=4, buffer_size=10000,
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
                 version='v17.1', target_net=None, in_channels=2, device='cuda', *args, **kwargs):

        super(DeepQLearningAgent, self).__init__()

        # Other attributes
        self._board_size = board_size
        self._frames = frames
        self._n_actions = n_actions
        self._buffer_size = buffer_size
        self._gamma = gamma
        self._use_target_net = use_target_net
        self._version = version
        self._target_net = target_net
        self._in_channels = in_channels
        self.device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.reset_models()

        # Initialize the optimizer
        self.optimizer = optim.RMSprop(self._model.parameters(), lr=0.0005)

        if use_target_net:
            self._target_net = self._agent_model()
            self._target_net.to(self.device)  # Move the target network to the same device as the main network
            self._target_net.load_state_dict(self._model.state_dict())  # Initialize target network
        else:
            self._target_net = None

        self._buffer = replay_buffer.ReplayBufferNumpy(buffer_size, board_size, frames, n_actions)

        self.update_target_net()

        # New attributes for replay buffer
        """self._board = np.zeros((buffer_size, board_size, board_size, frames), dtype=np.float32)
        self._action = np.zeros(buffer_size, dtype=np.int32)
        self._reward = np.zeros(buffer_size, dtype=np.float32)
        self._next_board = np.zeros((buffer_size, board_size, board_size, frames), dtype=np.float32)
        self._done = np.zeros(buffer_size, dtype=np.float32)
        self._legal_moves = np.zeros((buffer_size, n_actions), dtype=np.float32)
        self._max_size = buffer_size
        self._current_size = 0
        self._current_index = 0"""

    """def forward(self, x):
        print("Input Shape:", x.shape)
        x = F.relu(self.conv1(x))
        print("After Conv1 Shape:", x.shape)
        x = F.relu(self.conv2(x))
        print("After Conv2 Shape:", x.shape)

        # Flatten the tensor before passing through fully connected layers
        x = self.flatten(x)
        print("After Flatten Shape:", x.shape)

        x = F.relu(self.fc1(x))
        print("After FC1 Shape:", x.shape)
        x = self.fc2(x)
        print("After FC2 Shape:", x.shape)
        return x"""

    """def forward(self, x):
        print("Input Shape:", x.shape)
        x = F.relu(self.conv1(x))
        print("After Conv1 Shape:", x.shape)
        x = F.relu(self.conv2(x))
        print("After Conv2 Shape:", x.shape)
        x = self.flatten(x)  # Add this line to flatten the output
        print("After Flatten Shape:", x.shape)
        x = F.relu(self.fc1(x))
        print("After FC1 Shape:", x.shape)
        x = self.fc2(x)
        print("After FC2 Shape:", x.shape)
        return x"""

    def forward(self, x):
        layers = self._agent_model()
        for layer in layers:
            x = layer(x)
        return x

    def parameters(self):
        return super(DeepQLearningAgent, self).parameters()

    def reset_models(self):
        """Reset the DQN model and target network"""
        self._model = self._agent_model()
        if self._use_target_net:
            self._target_net = self._agent_model()
        self.update_target_net()
        """self._model = self._agent_model()
        if self._use_target_net:
            self._target_net = self._agent_model()
        self.update_target_net()  # copy weights from model to a target network"""

    def _prepare_input(self, board):
        assert board.shape[0] == 10 and board.shape[1] == 10 and board.shape[2] == 2

        if len(board.shape) == 3:
            # Assuming the shape is [height, width, channels], change it to [channels, height, width]
            board = np.transpose(board, (2, 0, 1))
            board = board.reshape(1, *board.shape)  # Add batch dimension

        board = self._normalize_board(board)
        return board.clone()

    def _get_model_outputs(self, board, model=None):
        # to correct dimensions and normalize
        board = self._prepare_input(board)
        # the default model to use
        if model is None:
            model = self

        # Use the PyTorch model for prediction
        model_outputs = model(board)

        # Convert the PyTorch tensor to a NumPy array
        model_outputs = model_outputs.detach().numpy()

        return model_outputs

    def _normalize_board(self, board):
        return torch.tensor(board, dtype=torch.float32) / 4.0

    def move(self, board, legal_moves, value=None):
        model_outputs = self._get_model_outputs(board, self)
        return np.argmax(np.where(legal_moves == 1, model_outputs, -np.inf), axis=1)

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
        with open('model_config/{:s}.json'.format(self._version), 'r') as f:
            m = json.loads(f.read())

        print("Loaded JSON content:", m)

        # Check if 'input' key is present in the model configuration
        if 'input' not in m['model']:
            raise KeyError("Key 'input' not found in the model configuration.")

        # Input size
        input_shape = m['model']['input']['shape']
        input_size = (input_shape[2], input_shape[0], input_shape[1])  # PyTorch uses (C, H, W) convention

        # Output size
        output_size = m['model']['action_values']['out_features']

        # Instantiate the model
        model = DeepQLearningAgent(self._board_size, self._frames, self._n_actions)
        optimizer = optim.RMSprop(model.parameters(), lr=0.0005)
        return model, optimizer
        # Load model configuration from JSON
        with open('model_config/{:s}.json'.format(self._version), 'r') as f:
            config = json.loads(f.read())

        layers = []
        # Inside the _agent_model function
        for layer_name, layer_params in config['model'].items():
            if 'Conv2D' in layer_name:
                # Change 'filters' to 'out_channels'
                layer_params['out_channels'] = layer_params.pop('filters')
                layers.append(nn.Conv2d(**layer_params))
                layers.append(nn.ReLU())  # Applying ReLU activation separately
            elif 'Flatten' in layer_name:
                layers.append(nn.Flatten())
            elif 'Dense' in layer_name:
                layers.append(nn.Linear(**layer_params))

        # Remove the last Linear layer added
        layers.pop()  # Remove the last Linear layer added

        # Append the ReLU activation after the last Dense layer
        layers.append(nn.ReLU())

        # Final output layer
        layers.append(nn.Linear(64, self._n_actions))

        # Combine all layers into a Sequential model
        model = nn.Sequential(*layers)

        return model"""

        layers = []

        # Load model configuration from JSON
        with open('model_config/{:s}.json'.format(self._version), 'r') as f:
            config = json.loads(f.read())

        # Inside the _agent_model function
        for layer_name, layer_params in config['model'].items():
            if 'Conv2D' in layer_name:
                # Change 'filters' to 'out_channels'
                layer_params['out_channels'] = layer_params.pop('filters')
                # Add 'in_channels' to layer_params
                layer_params['in_channels'] = self._in_channels
                # Remove 'activation' and 'data_format' from layer_params
                activation = layer_params.pop('activation', None)
                layer_params.pop('data_format', None)
                layers.append(nn.Conv2d(**layer_params))
                if activation:
                    layers.append(nn.ReLU())  # Applying ReLU activation separately
            elif 'Flatten' in layer_name:
                layers.append(nn.Flatten())
            elif 'Dense' in layer_name:
                # Change 'units' to 'out_features'
                layer_params['out_features'] = layer_params.pop('units')
                # Remove 'activation' from layer_params
                activation = layer_params.pop('activation', None)
                # Remove 'name' from layer_params
                layer_params.pop('name', None)
                # Add 'in_features' to layer_params
                layer_params['in_features'] = 64  # You need to set the correct in_features value
                dense_layer = nn.Linear(**layer_params)
                layers.append(dense_layer)
                # Apply activation separately if provided
                if activation:
                    layers.append(nn.ReLU())

        # Remove the last Linear layer added
        layers.pop()  # Remove the last Linear layer added

        # Append the ReLU activation after the last Dense layer
        layers.append(nn.ReLU())

        # Final output layer
        layers.append(nn.Linear(64, self._n_actions))

        # Combine all layers into a Sequential model
        model = nn.Sequential(*layers)

        return model

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
        """model_outputs = self._get_model_outputs(board, self._model)
        # subtracting max and taking softmax does not change output
        # do this for numerical stability
        model_outputs = np.clip(model_outputs, -10, 10)
        model_outputs = model_outputs - model_outputs.max(axis=1).reshape((-1, 1))
        model_outputs = np.exp(model_outputs)
        model_outputs = model_outputs / model_outputs.sum(axis=1).reshape((-1, 1))
        return model_outputs"""
        model_outputs = self._get_model_outputs(board, self._model)
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
        if iteration is not None:
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0

        torch.save(self.state_dict(), "{}/model_{:04d}.pth".format(file_path, iteration))

        if self._use_target_net:
            torch.save(self.state_dict(), "{}/model_{:04d}_target.pth".format(file_path, iteration))

        """if (iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        self._model.save_weights("{}/model_{:04d}.h5".format(file_path, iteration))
        if (self._use_target_net):
            self._target_net.save_weights("{}/model_{:04d}_target.h5".format(file_path, iteration))
"""

    def load_model(self, file_path='', iteration=None):
        if iteration is not None:
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0

        try:
            # Load the model state_dict
            model_state_dict = torch.load("{}/model_{:04d}.pth".format(file_path, iteration))

            # Filter out unexpected keys
            unexpected_keys = [k for k in model_state_dict.keys() if k not in self.state_dict()]
            model_state_dict = {k: v for k, v in model_state_dict.items() if k in self.state_dict()}

            # Handle conv1.weight separately to account for input channel mismatch
            if 'conv1.weight' in model_state_dict and model_state_dict['conv1.weight'].shape[
                1] != self.conv1.in_channels:
                common_channels = min(model_state_dict['conv1.weight'].shape[1], self.conv1.in_channels)
                self.conv1.weight.data[:, :common_channels, :, :] = model_state_dict['conv1.weight'][:,
                                                                    :common_channels, :, :]
                self.conv1.weight.data[:, common_channels:, :, :].uniform_(-0.1, 0.1)

            # Update the state_dict
            self.load_state_dict(model_state_dict, strict=False)

            if self._use_target_net:
                # Load the target_net state_dict
                target_net_state_dict = torch.load("{}/model_{:04d}_target.pth".format(file_path, iteration))
                target_net_state_dict = {k: v for k, v in target_net_state_dict.items() if
                                         k in self._target_net.state_dict()}
                self._target_net.load_state_dict(target_net_state_dict, strict=False)

            if unexpected_keys:
                print(f"Unexpected keys found in the model state_dict: {unexpected_keys}. They are ignored.")

        except FileNotFoundError:
            print("Couldn't locate models at {}, check the provided path")
        except RuntimeError as e:
            print(f"Error loading model: {e}")

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
            r = torch.sign(torch.tensor(r, dtype=torch.float32)).numpy()  # Convert PyTorch tensor to NumPy array

        legal_moves = torch.tensor(legal_moves, dtype=torch.float32)  # Convert NumPy array to PyTorch tensor

        current_model = self  # No need for _target_net in PyTorch
        next_model_outputs = self._get_model_outputs(next_s, current_model)
        next_model_outputs = torch.tensor(next_model_outputs, dtype=torch.float32).numpy()

        # If r has more than 2 dimensions, squeeze it to 2D
        r = np.squeeze(r)

        # Create the target variable, only the column with the action has a different value
        target = self._get_model_outputs(s)

        # Our estimate of expected future discounted reward
        discounted_reward_component = (
                                              self._gamma * np.max(
                                          np.where(legal_moves == 1, next_model_outputs, -np.inf), axis=1)
                                      ).reshape(-1, 1) * (1 - done)

        # Expand discounted_reward_component to match the number of actions
        discounted_reward_component = np.tile(discounted_reward_component, (1, target.shape[1]))

        # Overwrite the relevant columns of the target with discounted_reward_component
        target[:, :discounted_reward_component.shape[1]] = discounted_reward_component

        # We bother only with the difference in reward estimate at the selected action
        target = (1 - a) * target + a * discounted_reward_component

        # Convert to PyTorch tensors
        s_tensor = torch.tensor(s, dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.float32)

        # Forward pass
        predictions = self(s_tensor)

        # Calculate loss
        loss = F.mse_loss(predictions, target_tensor)

        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Return loss
        return loss.item()

    def update_target_net(self):
        """Update the weights of the target network."""
        """self._target_net.load_state_dict(self._model.state_dict())"""
        """if self._use_target_net and self._target_net is not None:
            # Filter out unnecessary keys before loading the state_dict
            source_state_dict = self.state_dict()
            target_state_dict = self._target_net.state_dict()
            filtered_state_dict = {key: source_state_dict[key] for key in target_state_dict.keys()}
            self._target_net.load_state_dict(filtered_state_dict)"""
        if self._use_target_net and self._target_net is not None:
            # Extract the state dictionaries from the source and target models
            source_state_dict = self.state_dict()
            target_state_dict = self._target_net.state_dict()

            # Iterate through the layers and load the weights
            for (source_key, source_value), (target_key, target_value) in zip(source_state_dict.items(),
                                                                              target_state_dict.items()):
                # Check if the layer name matches and load the weights
                if source_key.split('.')[1:] == target_key.split('.')[1:]:
                    target_state_dict[target_key] = source_value

            # Load the updated state_dict into the target network
            self._target_net.load_state_dict(target_state_dict)

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
        assert isinstance(agent_for_copy, DeepQLearningAgent), "Agent type is required for copy"

        self._model.load_state_dict(agent_for_copy._model.state_dict())
        self._target_net.load_state_dict(agent_for_copy._model.state_dict())


class PolicyGradientAgent(DeepQLearningAgent):
    """This agent learns via Policy Gradient method

    Attributes
    ----------
    _update_function : function
        defines the policy update function to use while training
    """

    """def __init__(self, board_size=10, frames=4, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=False,
                 version=''):
        Initializer for PolicyGradientAgent, similar to DeepQLearningAgent
        but does an extra assignment to the training function
        
        DeepQLearningAgent.__init__(self, board_size=board_size, frames=frames,
                                    buffer_size=buffer_size, gamma=gamma,
                                    n_actions=n_actions, use_target_net=False,
                                    version=version)
        self._actor_optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-6)
        # tf.keras.optimizer.Adam(1e-6)"""

    def __init__(self, board_size=10, frames=4, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=False,
                 version=''):
        super(PolicyGradientAgent, self).__init__(board_size, frames,
                                                  buffer_size, gamma,
                                                  n_actions, use_target_net,
                                                  version)
        self._actor_optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-6)

    def _agent_model(self):
        """Returns the model which evaluates prob values for a given state input
        Model is compiled in a different function
        Overrides parent
        
        Returns
        -------
        model : TensorFlow Graph
            Policy Gradient model graph
        """
        """input_board = Input((self._board_size, self._board_size, self._n_frames,))
        x = Conv2D(16, (4, 4), activation='relu', data_format='channels_last', kernel_regularizer=l2(0.01))(input_board)
        x = Conv2D(32, (4, 4), activation='relu', data_format='channels_last', kernel_regularizer=l2(0.01))(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
        out = Dense(self._n_actions, activation='linear', name='action_logits', kernel_regularizer=l2(0.01))(x)

        model = Model(inputs=input_board, outputs=out)
        # do not compile the model here, but rather use the outputs separately
        # in a training function to create any custom loss function
        # model.compile(optimizer = RMSprop(0.0005), loss = 'mean_squared_error')
        return model"""
        input_channels = self._n_frames
        model = DeepQLearningAgent.Sequential(
            DeepQLearningAgent.Conv2d(input_channels, 16, kernel_size=4, stride=4),
            DeepQLearningAgent.ReLU(),
            DeepQLearningAgent.Conv2d(16, 32, kernel_size=4, stride=4),
            DeepQLearningAgent.ReLU(),
            DeepQLearningAgent.Flatten(),
            DeepQLearningAgent.Linear(32 * 4 * 4, 64),
            DeepQLearningAgent.ReLU(),
            DeepQLearningAgent.Linear(64, self._n_actions)
        )
        return model

    def train_agent(self, batch_size=32, beta=0.1, normalize_rewards=False,
                    num_games=1, reward_clip=False):
        """Train the model by sampling from buffer and return the error
        The buffer is assumed to contain all states of a finite set of games
        and is fully sampled from the buffer
        Overrides parent
        
        Parameters
        ----------
        batch_size : int, optional
            Not used here, kept for consistency with other agents
        beta : float, optional
            The weight for the entropy loss
        normalize_rewards : bool, optional
            Whether to normalize rewards for stable training
        num_games : int, optional
            Total games played in the current batch
        reward_clip : bool, optional
            Not used here, kept for consistency with other agents

        Returns
        -------
        error : list
            The current loss (total loss, classification loss, entropy)
        """
        # in policy gradient, only complete episodes are used for training
        s, a, r, _, _, _ = self._buffer.sample(self._buffer.get_current_size())
        # unlike DQN, the discounted reward is not estimated but true one
        # we have defined custom policy graident loss function above
        # use that to train to agent model
        # normzlize the rewards for training stability
        if (normalize_rewards):
            r = (r - np.mean(r)) / (np.std(r) + 1e-8)
        target = np.multiply(a, r)
        loss = actor_loss_update(self._prepare_input(s), target, self._model,
                                 self._actor_optimizer, beta=beta, num_games=num_games)
        return loss[0] if len(loss) == 1 else loss


class AdvantageActorCriticAgent(PolicyGradientAgent):
    """This agent uses the Advantage Actor Critic method to train
    the reinforcement learning agent, we will use Q actor critic here

    Attributes
    ----------
    _action_values_model : Tensorflow Graph
        Contains the network for the action values calculation model
    _actor_update : function
        Custom function to prepare the 
    """

    """def __init__(self, board_size=10, frames=4, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=True,
                 version=''):
        super(AdvantageActorCriticAgent, self).__init__(board_size, frames,
                                                        buffer_size, gamma,
                                                        n_actions, use_target_net,
                                                        version)
        self._model_logits, self._model_full, self._model_values = self._agent_model()
        self._optimizer = torch.optim.RMSprop(self.parameters(), lr=5e-4)"""

    def __init__(self, board_size=10, frames=4, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=True,
                 version=''):
        super(AdvantageActorCriticAgent, self).__init__(board_size, frames,
                                                        buffer_size, gamma,
                                                        n_actions, use_target_net,
                                                        version)
        self._model = None
        self._optimizer = torch.optim.RMSprop(self._model.parameters(), lr=5e-4)
        if use_target_net:
            self._target_net, _ = self._agent_model()
            self.update_target_net()  # copy weights from model to a target network

    """def __init__(self, board_size=10, frames=4, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=True,
                 version=''):
        DeepQLearningAgent.__init__(self, board_size=board_size, frames=frames,
                                    buffer_size=buffer_size, gamma=gamma,
                                    n_actions=n_actions, use_target_net=use_target_net,
                                    version=version)
        self._optimizer = torch.optim.RMSprop(self.parameters(), lr=5e-4)
        # tf.keras.optimizers.RMSprop(5e-4)"""

    def _agent_model(self):
        """Returns the models which evaluate prob logits and action values
        for a given state input, Model is compiled in a different function
        Overrides parent

        Returns
        -------
        model_logits : TensorFlow Graph
            A2C model graph for action logits
        model_full : TensorFlow Graph
            A2C model complete graph
        """
        """input_board = Input((self._board_size, self._board_size, self._n_frames,))
        x = Conv2D(16, (3, 3), activation='relu', data_format='channels_last')(input_board)
        x = Conv2D(32, (3, 3), activation='relu', data_format='channels_last')(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu', name='dense')(x)
        action_logits = Dense(self._n_actions, activation='linear', name='action_logits')(x)
        state_values = Dense(1, activation='linear', name='state_values')(x)

        model_logits = Model(inputs=input_board, outputs=action_logits)
        model_full = Model(inputs=input_board, outputs=[action_logits, state_values])
        model_values = Model(inputs=input_board, outputs=state_values)
        # updates are calculated in the train_agent function

        return model_logits, model_full, model_values"""
        input_channels = self._n_frames
        shared_layers = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 6 * 6, 64),
            nn.ReLU()
        )

        actor = nn.Sequential(
            shared_layers,
            nn.Linear(64, self._n_actions)
        )

        critic = nn.Sequential(
            shared_layers,
            nn.Linear(64, 1)
        )

        return actor, nn.ModuleList([actor, critic]), critic

    def reset_models(self):
        """ Reset all the models by creating new graphs"""
        self._model, _ = self._agent_model()

        if self._use_target_net:
            self._target_net, _ = self._agent_model()
            self.update_target_net()  # copy weights from model to a target network

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
        if iteration is not None:
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0

        torch.save(self._model.state_dict(), "{}/model_{:04d}.pth".format(file_path, iteration))

        if self._use_target_net:
            torch.save(self._target_net.state_dict(), "{}/model_{:04d}_target.pth".format(file_path, iteration))

        """if (iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        self._model.save_weights("{}/model_{:04d}.h5".format(file_path, iteration))
        self._full_model.save_weights("{}/model_{:04d}_full.h5".format(file_path, iteration))
        if (self._use_target_net):
            self._values_model.save_weights("{}/model_{:04d}_values.h5".format(file_path, iteration))
            self._target_net.save_weights("{}/model_{:04d}_target.h5".format(file_path, iteration))
"""

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
        """
        """if iteration is not None:
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        self._model.load_weights("{}/model_{:04d}.h5".format(file_path, iteration))
        self._full_model.load_weights("{}/model_{:04d}_full.h5".format(file_path, iteration))
        if self._use_target_net:
            self._values_model.load_weights("{}/model_{:04d}_values.h5".format(file_path, iteration))
            self._target_net.load_weights("{}/model_{:04d}_target.h5".format(file_path, iteration))"""
        if iteration is not None:
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0

        try:
            self._model.load_state_dict(torch.load("{}/model_{:04d}.pth".format(file_path, iteration)))

            if self._use_target_net:
                self._target_net.load_state_dict(torch.load("{}/model_{:04d}_target.pth".format(file_path, iteration)))

        except FileNotFoundError:
            print("Couldn't locate models at {}, check provided path".format(file_path))

        """if iteration is not None:
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0

        try:
            self._model.load_weights("{}/model_{:04d}.h5".format(file_path, iteration))
            if self._use_target_net:
                self._target_net.load_weights("{}/model_{:04d}_target.h5".format(file_path, iteration))
        except FileNotFoundError:
            print("Models not found. Check the provided path.")
"""

    def update_target_net(self):
        """Update the weights of the target network."""
        if self._use_target_net and self._target_net is not None:
            # Filter out unnecessary keys before loading the state_dict
            source_state_dict = self.state_dict()
            target_state_dict = self._target_net.state_dict()
            filtered_state_dict = {key: source_state_dict[key] for key in target_state_dict.keys()}
            self._target_net.load_state_dict(filtered_state_dict)

    def train_agent(self, batch_size=32, beta=0.001, normalize_rewards=False,
                    num_games=1, reward_clip=False):
        """Train the model by sampling from buffer and return the error
        The buffer is assumed to contain all states of a finite set of games
        and is fully sampled from the buffer
        Overrides parent
        
        Parameters
        ----------
        batch_size : int, optional
            Not used here, kept for consistency with other agents
        beta : float, optional
            The weight for the policy gradient entropy loss
        normalize_rewards : bool, optional
            Whether to normalize rewards for stable training
        num_games : int, optional
            Not used here, kept for consistency with other agents
        reward_clip : bool, optional
            Not used here, kept for consistency with other agents

        Returns
        -------
        error : list
            The current loss (total loss, actor loss, critic loss)
        """
        # in policy gradient, only one complete episode is used for training
        s, a, r, next_s, done, _ = self._buffer.sample(self._buffer.get_current_size())
        s_prepared = self._prepare_input(s)
        next_s_prepared = self._prepare_input(next_s)
        # unlike DQN, the discounted reward is not estimated
        # we have defined custom actor and critic losses functions above
        # use that to train to an agent model

        # convert to PyTorch tensors
        s_prepared = torch.FloatTensor(s_prepared)
        a = torch.FloatTensor(a)
        r = torch.FloatTensor(r)
        next_s_prepared = torch.FloatTensor(next_s_prepared)
        done = torch.FloatTensor(done)

        # normalize the rewards for training stability, does not work in practice
        if normalize_rewards:
            if r.std() == 0:
                # std dev is zero
                r -= r
            else:
                r = (r - r.mean()) / r.std()

        if reward_clip:
            r = torch.sign(r)

            # calculate V values
            if self._use_target_net:
                next_s_pred = self._target_net(next_s_prepared)
            else:
                next_s_pred = self._values_model(next_s_prepared)
            s_pred = self._values_model(s_prepared)

            # prepare target
            future_reward = self._gamma * next_s_pred * (1 - done)

            # calculate target for actor (uses advantage), similar to Policy Gradient
            advantage = a * (r + future_reward - s_pred)

            # calculate target for critic, simply current reward + future expected reward
            critic_target = r + future_reward

            model = self._full_model

            # PyTorch does not have a native function for entropy, so we need to implement it
            policy = F.softmax(model(s_prepared)[0], dim=1)
            log_policy = F.log_softmax(model(s_prepared)[0], dim=1)

            # calculate loss
            J = torch.sum(advantage * log_policy) / num_games
            entropy = -torch.sum(policy * log_policy) / num_games
            actor_loss = -J - beta * entropy
            critic_loss = mean_huber_loss(critic_target, model(s_prepared)[1])
            loss = actor_loss + critic_loss

            # zero the gradients before backward pass
            self._optimizer.zero_grad()

            # backward pass
            loss.backward()

            # clip gradients if needed
            # for param in self._full_model.parameters():
            #    param.grad.data.clamp_(-5, 5)

            # run the optimizer
            self._optimizer.step()

            return [loss.item(), actor_loss.item(), critic_loss.item()]

        """# normzlize the rewards for training stability, does not work in practice
        if (normalize_rewards):
            if ((r == r[0][0]).sum() == r.shape[0]):
                # std dev is zero
                r -= r
            else:
                r = (r - np.mean(r)) / np.std(r)

        if (reward_clip):
            r = np.sign(r)"""

        """# calculate V values
        if (self._use_target_net):
            next_s_pred = self._target_net.predict_on_batch(next_s_prepared)
        else:
            next_s_pred = self._values_model.predict_on_batch(next_s_prepared)
        s_pred = self._values_model.predict_on_batch(s_prepared)

        # prepare target
        future_reward = self._gamma * next_s_pred * (1 - done)
        # calculate target for actor (uses advantage), similar to Policy Gradient
        advantage = a * (r + future_reward - s_pred)

        # calculate target for critic, simply current reward + future expected reward
        critic_target = r + future_reward

        model = self._full_model
        with tf.GradientTape() as tape:
            model_out = model(s_prepared)
            policy = tf.nn.softmax(model_out[0])
            log_policy = tf.nn.log_softmax(model_out[0])
            # calculate loss
            J = tf.reduce_sum(tf.multiply(advantage, log_policy)) / num_games
            entropy = -tf.reduce_sum(tf.multiply(policy, log_policy)) / num_games
            actor_loss = -J - beta * entropy
            critic_loss = mean_huber_loss(critic_target, model_out[1])
            loss = actor_loss + critic_loss
        # get the gradients
        grads = tape.gradient(loss, model.trainable_weights)
        # grads = [tf.clip_by_value(grad, -5, 5) for grad in grads]
        # run the optimizer
        self._optimizer.apply_gradients(zip(grads, model.trainable_variables))

        loss = [loss.numpy(), actor_loss.numpy(), critic_loss.numpy()]
        return loss[0] if len(loss) == 1 else loss"""


class HamiltonianCycleAgent(Agent):
    """This agent prepares a Hamiltonian Cycle through the board and then
    follows it to reach the food, inherits Agent

    Attributes
    ----------
        board_size (int): side length of the board
        frames (int): no of frames available in one board state
        n_actions (int): no of actions available in the action space
    """

    def __init__(self, board_size=10, frames=4, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=False,
                 version=''):
        assert board_size % 2 == 0, "Board size should be odd for hamiltonian cycle"
        Agent.__init__(self, board_size=board_size, frames=frames, buffer_size=buffer_size,
                       gamma=gamma, n_actions=n_actions, use_target_net=use_target_net,
                       version=version)
        # self._get_cycle()
        self._get_cycle_square()

    def _get_neighbors(self, point):
        """
        point is a single integer such that 
        row = point//self._board_size
        col = point%self._board_size
        """
        row, col = point // self._board_size, point % self._board_size
        neighbors = []
        for delta_row, delta_col in [[-1, 0], [1, 0], [0, 1], [0, -1]]:
            new_row, new_col = row + delta_row, col + delta_col
            if (1 <= new_row and new_row <= self._board_size - 2 and \
                    1 <= new_col and new_col <= self._board_size - 2):
                neighbors.append(new_row * self._board_size + new_col)
        return neighbors

    def _hamil_util(self):
        neighbors = self._get_neighbors(self._cycle[self._index])
        if (self._index == ((self._board_size - 2) ** 2) - 1):
            if (self._start_point in neighbors):
                # end of path and cycle
                return True
            else:
                # end of path but not cycle
                return False
        else:
            for i in neighbors:
                if (i not in self._cycle_set):
                    self._index += 1
                    self._cycle[self._index] = i
                    self._cycle_set.add(i)
                    ret = self._hamil_util()
                    if (ret):
                        return True
                    else:
                        # remove the element and backtrack
                        self._cycle_set.remove(self._cycle[self._index])
                        self._index -= 1
            # if all neighbors in cycle set
            return False

    def _get_cycle(self):
        """
        given a square board size, calculate a hamiltonian cycle through
        the graph, use it to follow the board, the _cycle variable is a list
        of tuples which tells the next coordinates to go to
        note that the board starts at row 1, col 1
        """
        self._start_point = 1 * self._board_size + 1
        self._cycle = np.zeros(((self._board_size - 2) ** 2,))
        # calculate the cycle path, start at 0, 0
        self._index = 0
        self._cycle[self._index] = self._start_point
        self._cycle_set = set([self._start_point])
        cycle_possible = self._hamil_util()

    def _get_cycle_square(self):
        """
        simple implementation to get the hamiltonian cycle
        for square board, by traversing in a up and down fashion
        all movement code is based on this implementation
        """
        self._cycle = np.zeros(((self._board_size - 2) ** 2,), dtype=np.int64)
        index = 0
        sp = 1 * self._board_size + 1
        while (index < self._cycle.shape[0]):
            if (index == 0):
                # put as is
                pass
            elif ((sp // self._board_size) == 2 and (sp % self._board_size) == self._board_size - 2):
                # at the point where we go up and then left to
                # complete the cycle, go up once
                sp = ((sp // self._board_size) - 1) * self._board_size + (sp % self._board_size)
            elif (index != 1 and sp // self._board_size == 1):
                # keep going left to complete cycle
                sp = ((sp // self._board_size)) * self._board_size + ((sp % self._board_size) - 1)
            elif ((sp % self._board_size) % 2 == 1):
                # go down till possible
                sp = ((sp // self._board_size) + 1) * self._board_size + (sp % self._board_size)
                if (sp // self._board_size == self._board_size - 1):
                    # should have turned right instead of goind down
                    sp = ((sp // self._board_size) - 1) * self._board_size + ((sp % self._board_size) + 1)
            else:
                # go up till the last but one row
                sp = ((sp // self._board_size) - 1) * self._board_size + (sp % self._board_size)
                if (sp // self._board_size == 1):
                    # should have turned right instead of goind up
                    sp = ((sp // self._board_size) + 1) * self._board_size + ((sp % self._board_size) + 1)
            self._cycle[index] = sp
            index += 1

    def move(self, board, legal_moves, values):
        """ get the action using agent policy """
        cy_len = (self._board_size - 2) ** 2
        curr_head = np.sum(self._board_grid * \
                           (board[:, :, 0] == values['head']).reshape(self._board_size, self._board_size))
        index = 0
        while (1):
            if (self._cycle[index] == curr_head):
                break
            index = (index + 1) % cy_len
        prev_head = self._cycle[(index - 1) % cy_len]
        next_head = self._cycle[(index + 1) % cy_len]
        # get the next move
        if (board[prev_head // self._board_size, prev_head % self._board_size, 0] == 0):
            # check if snake is in line with the hamiltonian cycle or not
            if (next_head > curr_head):
                return 3
            else:
                return 1
        else:
            # calcualte intended direction to get move
            curr_head_row, curr_head_col = self._point_to_row_col(curr_head)
            prev_head_row, prev_head_col = self._point_to_row_col(prev_head)
            next_head_row, next_head_col = self._point_to_row_col(next_head)
            dx, dy = next_head_col - curr_head_col, -next_head_row + curr_head_row
            if (dx == 1 and dy == 0):
                return 0
            elif (dx == 0 and dy == 1):
                return 1
            elif (dx == -1 and dy == 0):
                return 2
            elif (dx == 0 and dy == -1):
                return 3
            else:
                return -1

            """
            # calculate vectors representing current and new directions
            # to get the direction in which to turn
            d1 = (curr_head_row - prev_head_row, curr_head_col - prev_head_col)
            d2 = (next_head_row - curr_head_row, next_head_col - curr_head_col)
            # take cross product
            turn_dir = d1[0]*d2[1] - d1[1]*d2[0]
            if(turn_dir == 0):
                return 1
            elif(turn_dir == -1):
                return 0
            else:
                return 2
            """

    def get_action_proba(self, board, values):
        """ for compatibility """
        move = self.move(board, values)
        prob = [0] * self._n_actions
        prob[move] = 1
        return prob

    def _get_model_outputs(self, board=None, model=None):
        """ for compatibility """
        return [[0] * self._n_actions]

    def load_model(self, **kwargs):
        """ for compatibility """
        pass


class SupervisedLearningAgent(DeepQLearningAgent):
    """This agent learns in a supervised manner. A close to perfect
    agent is first used to generate training data, playing only for
    a few frames at a time, and then the actions taken by the perfect agent
    are used as targets. This helps learning of feature representation
    and can speed up training of DQN agent later.

    Attributes
    ----------
    _model_action_out : PyTorch Softmax layer
        A softmax layer on top of the DQN model to train as a classification
        problem (instead of regression)
    _model_action : PyTorch Model
        The model that will be trained and is simply DQN model + softmax
    """

    def __init__(self, board_size=10, frames=2, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=True,
                 version=''):
        """Initializer for SupervisedLearningAgent, similar to DeepQLearningAgent
        but creates extra layer and model for classification training
        """
        super(SupervisedLearningAgent, self).__init__(board_size=board_size, frames=frames,
                                                      buffer_size=buffer_size, gamma=gamma,
                                                      n_actions=n_actions, use_target_net=use_target_net,
                                                      version=version)

        # Define model with softmax activation, and use action as target
        # instead of the reward value
        self._model_action = nn.Sequential(
            nn.Conv2d(self._n_frames, 16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 2 * 2, 64),
            nn.ReLU(),
            nn.Linear(64, self._n_actions)
        )

        self._model_action_out = nn.Softmax(dim=1)
        self._optimizer_action = optim.Adam(self._model_action.parameters(), lr=0.0005)

    def train_agent(self, batch_size=32, num_games=1, epochs=5,
                    reward_clip=False):
        """Train the model by sampling from buffer and return the error.
        _model_action is trained as a classification problem to learn weights
        for all the layers of the DQN model

        Parameters
        ----------
        batch_size : int, optional
            The number of examples to sample from buffer, should be small
        num_games : int, optional
            Not used here, kept for consistency with other agents
        epochs : int, optional
            Number of epochs to train the model for
        reward_clip : bool, optional
            Not used here, kept for consistency with other agents

        Returns
        -------
        loss : float
            The current error (error metric is cross entropy)
        """
        s, a, _, _, _, _ = self._buffer.sample(self.get_buffer_size())

        # Convert to PyTorch tensors
        s = torch.FloatTensor(self._normalize_board(s))
        a = torch.LongTensor(np.argmax(a, axis=1))

        # Train the classification model
        for epoch in range(epochs):
            logits = self._model_action(s)
            loss = F.cross_entropy(logits, a)

            self._optimizer_action.zero_grad()
            loss.backward()
            self._optimizer_action.step()

        return loss.item()

    def get_max_output(self):
        """Get the maximum output of Q values from the model
        This value is used to later divide the weights of the output layer
        of DQN model since the values can be unexpectedly high because
        we are training the classification model (which disregards the relative
        magnitudes of the linear outputs)

        Returns
        -------
        max_value : int
            The maximum output produced by the network (_model)
        """
        s, _, _, _, _, _ = self._buffer.sample(self.get_buffer_size())
        max_value = np.max(np.abs(self._model_action(s).detach().numpy()))
        return max_value

    def normalize_layers(self, max_value=None):
        """Use the max value to divide the weights of the last layer
        of the DQN model, this helps stabilize the initial training of DQN

        Parameters
        ----------
        max_value : int, optional
            Value by which to divide, assumed to be 1 if None
        """
        # normalize output layers by this value
        if max_value is None or np.isnan(max_value):
            max_value = 1.0
        # don't normalize all layers as that will shrink the
        # output proportional to the no of layers
        last_layer = list(self._model_action.children())[-1]
        if isinstance(last_layer, nn.Linear):
            last_layer.weight.data /= max_value


class BreadthFirstSearchAgent(Agent):
    """
    finds the shortest path from head to food
    while avoiding the borders and body
    """

    def _get_neighbors(self, point, values, board):
        """
        point is a single integer such that 
        row = point//self._board_size
        col = point%self._board_size
        """
        row, col = self._point_to_row_col(point)
        neighbors = []
        for delta_row, delta_col in [[-1, 0], [1, 0], [0, 1], [0, -1]]:
            new_row, new_col = row + delta_row, col + delta_col
            if (board[new_row][new_col] in \
                    [values['board'], values['food'], values['head']]):
                neighbors.append(new_row * self._board_size + new_col)
        return neighbors

    def _get_shortest_path(self, board, values):
        # get the head coordinate
        board = board[:, :, 0]
        head = ((self._board_grid * (board == values['head'])).sum())
        points_to_search = deque()
        points_to_search.append(head)
        path = []
        row, col = self._point_to_row_col(head)
        distances = np.ones((self._board_size, self._board_size)) * np.inf
        distances[row][col] = 0
        visited = np.zeros((self._board_size, self._board_size))
        visited[row][col] = 1
        found = False
        while (not found):
            if (len(points_to_search) == 0):
                # complete board has been explored without finding path
                # take any arbitrary action
                path = []
                break
            else:
                curr_point = points_to_search.popleft()
                curr_row, curr_col = self._point_to_row_col(curr_point)
                n = self._get_neighbors(curr_point, values, board)
                if (len(n) == 0):
                    # no neighbors available, explore other paths
                    continue
                # iterate over neighbors and calculate distances
                for p in n:
                    row, col = self._point_to_row_col(p)
                    if (distances[row][col] > 1 + distances[curr_row][curr_col]):
                        # update shortest distance
                        distances[row][col] = 1 + distances[curr_row][curr_col]
                    if (board[row][col] == values['food']):
                        # reached food, break
                        found = True
                        break
                    if (visited[row][col] == 0):
                        visited[curr_row][curr_col] = 1
                        points_to_search.append(p)
        # create the path going backwards from the food
        curr_point = ((self._board_grid * (board == values['food'])).sum())
        path.append(curr_point)
        while (1):
            curr_row, curr_col = self._point_to_row_col(curr_point)
            if (distances[curr_row][curr_col] == np.inf):
                # path is not possible
                return []
            if (distances[curr_row][curr_col] == 0):
                # path is complete
                break
            n = self._get_neighbors(curr_point, values, board)
            for p in n:
                row, col = self._point_to_row_col(p)
                if (distances[row][col] != np.inf and \
                        distances[row][col] == distances[curr_row][curr_col] - 1):
                    path.append(p)
                    curr_point = p
                    break
        return path

    def move(self, board, legal_moves, values):
        if (board.ndim == 3):
            board = board.reshape((1,) + board.shape)
        board_main = board.copy()
        a = np.zeros((board.shape[0],), dtype=np.uint8)
        for i in range(board.shape[0]):
            board = board_main[i, :, :, :]
            path = self._get_shortest_path(board, values)
            if (len(path) == 0):
                a[i] = 1
                continue
            next_head = path[-2]
            curr_head = (self._board_grid * (board[:, :, 0] == values['head'])).sum()
            # get prev head position
            if (((board[:, :, 0] == values['head']) + (board[:, :, 0] == values['snake']) \
                 == (board[:, :, 1] == values['head']) + (board[:, :, 1] == values['snake'])).all()):
                # we are at the first frame, snake position is unchanged
                prev_head = curr_head - 1
            else:
                # we are moving
                prev_head = (self._board_grid * (board[:, :, 1] == values['head'])).sum()
            curr_head_row, curr_head_col = self._point_to_row_col(curr_head)
            prev_head_row, prev_head_col = self._point_to_row_col(prev_head)
            next_head_row, next_head_col = self._point_to_row_col(next_head)
            dx, dy = next_head_col - curr_head_col, -next_head_row + curr_head_row
            if (dx == 1 and dy == 0):
                a[i] = 0
            elif (dx == 0 and dy == 1):
                a[i] = 1
            elif (dx == -1 and dy == 0):
                a[i] = 2
            elif (dx == 0 and dy == -1):
                a[i] = 3
            else:
                a[i] = 0
        return a
        """
        d1 = (curr_head_row - prev_head_row, curr_head_col - prev_head_col)
        d2 = (next_head_row - curr_head_row, next_head_col - curr_head_col)
        # take cross product
        turn_dir = d1[0]*d2[1] - d1[1]*d2[0]
        if(turn_dir == 0):
            return 1
        elif(turn_dir == -1):
            return 0
        else:
            return 2
        """

    def get_action_proba(self, board, values):
        """ for compatibility """
        move = self.move(board, values)
        prob = [0] * self._n_actions
        prob[move] = 1
        return prob

    def _get_model_outputs(self, board=None, model=None):
        """ for compatibility """
        return [[0] * self._n_actions]

    def load_model(self, **kwargs):
        """ for compatibility """
        pass
