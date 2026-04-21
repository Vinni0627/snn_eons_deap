"""
snn_controller.py - FIXED VERSION 3
-------------------------------------
Key insight: The original LIF implementation has a fundamental problem.
The hidden layer spikes are used directly as binary (0/1) inputs to the
output layer via W_ho @ hidden_spikes. But hidden_spikes resets to zeros
every timestep, so the output layer only gets signal when hidden neurons
fire in THAT EXACT timestep. With random weights most hidden neurons fire
randomly and incoherently, so the output layer rarely gets consistent signal.

FIX: Use accumulated hidden spike RATES (not per-timestep binary spikes)
as the input to the output layer. This gives the output layer a stable,
graded signal to work with - much easier for the EA to evolve.

Also: simplify neuron parameters to make firing much more likely.
"""

import numpy as np


class LIFNeuron:
    """
    Leaky Integrate-and-Fire neuron with simplified parameters
    that make firing reliably achievable.
    """
    def __init__(self, tau_m=10.0, v_rest=0.0, v_thresh=1.0,
                 v_reset=0.0, dt=1.0):
        # Simplified: v_rest=0, v_thresh=1 (normalised units)
        # No need for biological mV values - same math, easier to tune
        self.tau_m    = tau_m
        self.v_rest   = v_rest
        self.v_thresh = v_thresh
        self.v_reset  = v_reset
        self.dt       = dt
        self.v        = v_rest
        self.refractory = 0

    def step(self, I_input):
        if self.refractory > 0:
            self.refractory -= 1
            self.v = self.v_reset
            return 0
        dv = (-(self.v - self.v_rest) + I_input) * (self.dt / self.tau_m)
        self.v += dv
        if self.v >= self.v_thresh:
            self.v = self.v_reset
            self.refractory = 2
            return 1
        return 0

    def reset(self):
        self.v = self.v_rest
        self.refractory = 0


class SNNController:
    """
    3-layer SNN: Input -> Hidden -> Output

    KEY FIX: Output layer receives CUMULATIVE hidden firing rates
    (normalised spike counts over sim_steps), not per-timestep binary spikes.
    This gives a stable graded signal that the EA can reliably shape.
    """

    def __init__(self, n_inputs=10, n_hidden=10, n_outputs=5,
                 sim_steps=50, weights=None):
        self.n_inputs   = n_inputs
        self.n_hidden   = n_hidden
        self.n_outputs  = n_outputs
        self.sim_steps  = sim_steps

        self.hidden_neurons = [LIFNeuron() for _ in range(n_hidden)]
        self.output_neurons = [LIFNeuron() for _ in range(n_outputs)]

        self.w_ih_size     = n_hidden * n_inputs
        self.w_ho_size     = n_outputs * n_hidden
        self.total_weights = self.w_ih_size + self.w_ho_size

        if weights is None:
            weights = np.random.uniform(-2.0, 2.0, self.total_weights)
        self.set_weights(weights)

    def set_weights(self, flat_weights):
        assert len(flat_weights) == self.total_weights
        self.W_ih = flat_weights[:self.w_ih_size].reshape(self.n_hidden, self.n_inputs)
        self.W_ho = flat_weights[self.w_ih_size:].reshape(self.n_outputs, self.n_hidden)

    def get_weights(self):
        return np.concatenate([self.W_ih.flatten(), self.W_ho.flatten()])

    def reset_state(self):
        for n in self.hidden_neurons:
            n.reset()
        for n in self.output_neurons:
            n.reset()

    def forward(self, sensor_readings):
        """
        Two-phase forward pass:
        Phase 1: Run input -> hidden for sim_steps, accumulate hidden rates
        Phase 2: Feed hidden rates (stable signal) -> output layer
        """
        # Clip inputs to [0, 1] range for stable currents
        sensor_readings = np.clip(sensor_readings, -1.0, 1.0)

        # Phase 1: Input -> Hidden
        # Input current scaled to reliably drive hidden neurons
        # With v_thresh=1.0, need I > 1.0 to fire
        # sensor in [0,1], weight in [-2,2], so I = w*s in [-2,2]
        # Positive weights on positive sensors = firing
        input_currents = sensor_readings  # shape (n_inputs,)

        hidden_spike_counts = np.zeros(self.n_hidden)

        for _ in range(self.sim_steps):
            hidden_inputs = self.W_ih @ input_currents  # shape (n_hidden,)
            for i, neuron in enumerate(self.hidden_neurons):
                spike = neuron.step(hidden_inputs[i])
                hidden_spike_counts[i] += spike

        # Normalise to firing RATES [0, 1]
        # Max possible spikes = sim_steps / (1 + refractory) ~ sim_steps/3
        max_spikes = self.sim_steps / 3.0
        hidden_rates = np.clip(hidden_spike_counts / max_spikes, 0.0, 1.0)

        # Phase 2: Hidden rates -> Output
        # Now output neurons get a STABLE graded input, not noisy binary spikes
        output_spike_counts = np.zeros(self.n_outputs)

        for _ in range(self.sim_steps):
            output_inputs = self.W_ho @ hidden_rates  # stable signal
            for i, neuron in enumerate(self.output_neurons):
                spike = neuron.step(output_inputs[i])
                output_spike_counts[i] += spike

        # Decode action
        if output_spike_counts.sum() == 0:
            # Fallback: use hidden rates to pick action via W_ho dot product
            # This gives the EA signal even when output neurons don't fire
            scores = self.W_ho @ hidden_rates
            action = int(np.argmax(scores))
        else:
            action = int(np.argmax(output_spike_counts))

        return action, output_spike_counts