"""
ann_controller.py
-----------------
Simple feedforward ANN controller (baseline for comparison with SNN).

Architecture: Input → Hidden (tanh) → Output (argmax)
Same weight count as the SNN for fair comparison.
"""

import numpy as np


class ANNController:
    """
    2-layer feedforward ANN controller.

    Parameters
    ----------
    n_inputs  : number of input neurons
    n_hidden  : number of hidden neurons
    n_outputs : number of output neurons (actions)
    weights   : flat numpy array. If None, random init.
    """

    def __init__(self, n_inputs=10, n_hidden=10, n_outputs=5, weights=None):
        self.n_inputs  = n_inputs
        self.n_hidden  = n_hidden
        self.n_outputs = n_outputs

        self.w_ih_size = n_hidden * n_inputs
        self.w_ho_size = n_outputs * n_hidden
        self.total_weights = self.w_ih_size + self.w_ho_size

        if weights is None:
            weights = np.random.uniform(-5.0, 5.0, self.total_weights)
        self.set_weights(weights)

    def set_weights(self, flat_weights):
        assert len(flat_weights) == self.total_weights
        self.W_ih = flat_weights[:self.w_ih_size].reshape(self.n_hidden, self.n_inputs)
        self.W_ho = flat_weights[self.w_ih_size:].reshape(self.n_outputs, self.n_hidden)

    def get_weights(self):
        return np.concatenate([self.W_ih.flatten(), self.W_ho.flatten()])

    def reset_state(self):
        pass   # ANN has no persistent state between steps

    def forward(self, sensor_readings):
        """
        Parameters
        ----------
        sensor_readings : np.ndarray shape (n_inputs,)

        Returns
        -------
        action       : int
        activations  : np.ndarray shape (n_outputs,)  output layer values
        """
        hidden = np.tanh(self.W_ih @ sensor_readings)
        output = np.tanh(self.W_ho @ hidden)
        action = int(np.argmax(output))
        return action, output
