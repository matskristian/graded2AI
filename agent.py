"""
store all the agents here
"""

import replay_buffer
from replay_buffer import ReplayBufferNumpy
import numpy as np
import pickle
import json
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn


def huber_loss(y_true, y_pred, delta=1):
    """Parameters
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
        loss values for all points"""

    error = y_true - y_pred  # error is of shape (batch_size, num_actions)
    quad_error = 0.5 * error ** 2  # shape (batch_size, num_actions)
    lin_error = delta * (torch.abs(error) - 0.5 * delta)  # shape (batch_size, num_actions)
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
        average loss across points"""
    return torch.mean(huber_loss(y_true, y_pred, delta))  # returns a scalar


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
        # Other attributes
        self._board_size = board_size
        self._n_frames = frames
        self._buffer_size = buffer_size
        self._n_actions = n_actions
        self._gamma = gamma
        self._use_target_net = use_target_net
        self._input_shape = (self._board_size, self._board_size, self._n_frames)
        self.reset_buffer()
        self._board_grid = np.arange(0, self._board_size ** 2).reshape(self._board_size, -1)
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
            Binary indicators for actions which are allowed at next states"""
        self._buffer.add_to_buffer(board, action, reward, next_board, done, legal_moves)

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
        with open(f"{file_path}/buffer_{iteration:04d}", 'wb') as f:
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
        with open(f"{file_path}/buffer_{iteration:04d}", 'rb') as f:
            self._buffer = pickle.load(f)

    def _point_to_row_col(self, row, col):
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
        return row * self._board_size + col

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
    AdvantageActorCriticAgent"""

    def __init__(self, board_size=10, frames=4, n_actions=3, buffer_size=10000, gamma=0.99, use_target_net=True,
                 version='v17.1',
                 target_net=None):  # Hardcoded the version in order to eliminate errors when loading the model

        super(DeepQLearningAgent, self).__init__()
        # Your neural network layers here
        self.conv1 = nn.Conv2d(10, 16, kernel_size=3, stride=1, padding=1)  # 10 input channels, 16 output channels
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # 16 input channels, 32 output channels
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 32 input channels, 64 output channels
        self.flatten = nn.Flatten()  # Flatten the output of the last convolutional layer
        self.fc1 = nn.Linear(640, 64)  # 640 input features, 64 output features
        self.fc2 = nn.Linear(64, n_actions)  # 64 input features, 3 output features (one for each action)

        # Other attributes
        self._board_size = board_size
        self._frames = frames
        self._n_actions = n_actions
        self._buffer_size = buffer_size
        self._gamma = gamma
        self._use_target_net = use_target_net
        self._version = version

        # Initialize the optimizer
        self.optimizer = optim.RMSprop(self.parameters(), lr=0.0005)  # Use RMSprop optimizer with learning rate 0.0005

        # Initialize the target network separately
        if use_target_net:
            self._target_net = DeepQLearningAgent(board_size, frames, n_actions, use_target_net=False)
        else:
            self._target_net = None

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

    # Forwards the input through the network
    def forward(self, x):
        x = F.relu(self.conv1(x))  # Apply relu activation to the output of the first convolutional layer
        x = F.relu(self.conv2(x))  # Apply relu activation to the output of the second convolutional layer
        x = self.flatten(x)  # Flatten the output of the last convolutional layer
        x = F.relu(self.fc1(x))  # Apply relu activation to the output of the first dense layer
        x = self.fc2(x)  # Apply relu activation to the output of the second dense layer
        return x  # Return the output of the network

    # Returns the parameters of the network
    def parameters(self):
        return list(self.conv1.parameters()) + list(self.conv2.parameters()) + list(self.conv3.parameters()) + \
            list(self.fc1.parameters()) + list(self.fc2.parameters())

    def reset_models(self):
        """ Reset all the models by creating new graphs"""
        self._model, _ = self._agent_model()  # create a new model
        if self._use_target_net:
            _, self._target_net = self._agent_model()  # create a new target network
        self.update_target_net()  # copy weights from model to a target network

    def _prepare_input(self, board):
        """Reshape input and normalize the board"""
        if len(board.shape) == 3:  # If the board has only one channel
            # The shape is [height, width, channels], change it to [channels, height, width]
            board = np.transpose(board, (2, 0, 1))  # PyTorch uses (C, H, W) convention
            board = board.reshape(1, *board.shape)  # Add batch dimension

        board = self._normalize_board(board)  # Normalize the board
        return board.clone()  # Return a copy of the board

    def _get_model_outputs(self, board, model=None):
        # Prepare the input for the model (reshape and normalize)
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
        """
        # use the agent model to make the predictions for the board
        model_outputs = self._get_model_outputs(board, self)
        return np.argmax(np.where(legal_moves == 1, model_outputs, -np.inf), axis=1)

    def _agent_model(self):
        # Load model configuration from JSON
        with open('model_config/{:s}.json'.format(self._version), 'r') as f:  # Load the model configuration from JSON
            m = json.loads(f.read())

        # Input size (C, H, W)
        input_shape = m['model']['input']['shape']
        input_size = (input_shape[2], input_shape[0], input_shape[1])

        # Output size
        output_size = m['model']['action_values']['out_features']

        # Instantiate the model and optimizer
        model = DeepQLearningAgent(self._board_size, self._frames, self._n_actions)
        optimizer = optim.RMSprop(model.parameters(), lr=0.0005)
        return model, optimizer

    def set_weights_trainable(self):
        """Set selected layers to non-trainable and compile the model"""
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
            Kept for consistency with other agent classes"""

        model_outputs = self._get_model_outputs(board, self._model)
        # subtracting max and taking softmax does not change output
        # do this for numerical stability
        model_outputs = np.clip(model_outputs, -10, 10)
        model_outputs = model_outputs - model_outputs.max(axis=1).reshape((-1, 1))
        model_outputs = np.exp(model_outputs)
        model_outputs = model_outputs / model_outputs.sum(axis=1).reshape((-1, 1))
        return model_outputs

    def save_model(self, file_path='', iteration=None):

        if iteration is not None:
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        # Save the model weights
        torch.save(self.state_dict(), "{}/model_{:04d}.pth".format(file_path, iteration))

        # Save the target network weights
        if self._use_target_net:
            torch.save(self.state_dict(), "{}/model_{:04d}_target.pth".format(file_path, iteration))

    def load_model(self, file_path='', iteration=None):

        # Load the model from disk
        if iteration is not None:
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0

        # Load the model weights
        try:
            self.load_state_dict(torch.load("{}/model_{:04d}.pth".format(file_path, iteration)))

        except FileNotFoundError:
            print("Couldn't locate models at {}, check provided path".format(file_path))

    def print_models(self):
        """Print the current models using summary method"""
        print('Training Model')
        print(self._model.summary())
        if (self._use_target_net):
            print('Target Network')
            print(self._target_net.summary())

    def train_agent(self, batch_size=32, num_games=1, reward_clip=False):

        # Sample from the buffer and convert to PyTorch tensors for training
        s, a, r, next_s, done, legal_moves = self._buffer.sample(batch_size)
        if reward_clip:
            r = torch.sign(torch.tensor(r, dtype=torch.float32)).numpy()  # Convert PyTorch tensor to NumPy array

        # Convert to PyTorch tensors
        legal_moves = torch.tensor(legal_moves, dtype=torch.float32)  # Convert NumPy array to PyTorch tensor

        # Convert to PyTorch tensors
        current_model = self  # No need for _target_net in PyTorch
        next_model_outputs = self._get_model_outputs(next_s, current_model)  # Get the model outputs for next state
        next_model_outputs = torch.tensor(next_model_outputs,
                                          dtype=torch.float32).numpy()  # Convert PyTorch tensor to NumPy array

        # If r has more than 2 dimensions, squeeze it to 2D
        r = np.squeeze(r)

        # Create the target variable, only the column with the action has a different value
        target = self._get_model_outputs(s)

        # Estimate of expected future discounted reward
        discounted_reward_component = (
                                              self._gamma * np.max(
                                          np.where(legal_moves == 1, next_model_outputs, -np.inf), axis=1)
                                      ).reshape(-1, 1) * (1 - done)

        # Expand discounted_reward_component to match the number of actions
        discounted_reward_component = np.tile(discounted_reward_component, (1, target.shape[1]))

        # Overwrite the relevant columns of the target with discounted_reward_component
        target[:, :discounted_reward_component.shape[1]] = discounted_reward_component

        # Bother only with the difference in reward estimate at the selected action
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
        """Update the weights of the target network, which is kept
        static for a few iterations to stabilize the other network.
        This should not be updated very frequently
        """
        # Update the target network
        if self._use_target_net and self._target_net is not None:
            # Filter out unnecessary keys before loading the state_dict
            source_state_dict = self.state_dict()
            target_state_dict = self._target_net.state_dict()
            filtered_state_dict = {key: source_state_dict[key] for key in target_state_dict.keys()}
            self._target_net.load_state_dict(filtered_state_dict)

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
