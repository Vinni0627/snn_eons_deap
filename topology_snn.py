"""
topology_snn.py
---------------
Variable-topology LIF controller driven by a neuro.Network topology JSON.

The network JSON from net.as_json() has:
  "Inputs"  : [node_id, ...] ordered list of input node IDs
  "Outputs" : [node_id, ...] ordered list of output node IDs
  "Nodes"   : [{"id": int, "values": []}, ...]
  "Edges"   : [{"from": int, "to": int, "values": []}, ...]

Weights are passed as a flat list/array with one value per edge (same order
as "Edges"). EONS manages topology; DEAP manages these weights.

Forward pass mirrors snn_controller.py's two-phase rate approach:
  Phase 1: input -> hidden for sim_steps  → hidden spike rates
  Phase 2: hidden rates -> output for sim_steps → output spike counts → action
Direct input->output edges are included in Phase 2.
"""

import numpy as np


class LIFNeuron:
    def __init__(self, tau_m=10.0, v_thresh=1.0, dt=1.0):
        self.tau_m    = tau_m
        self.v_thresh = v_thresh
        self.dt       = dt
        self.v        = 0.0
        self.refractory = 0

    def step(self, I_input):
        if self.refractory > 0:
            self.refractory -= 1
            self.v = 0.0
            return 0
        self.v += (-self.v + I_input) * (self.dt / self.tau_m)
        if self.v >= self.v_thresh:
            self.v = 0.0
            self.refractory = 2
            return 1
        return 0

    def reset(self):
        self.v = 0.0
        self.refractory = 0


class TopologySNN:
    """
    LIF network with topology defined by a neuro.Network JSON snapshot.
    One weight per edge in topology_json["Edges"], indexed by position.
    """

    def __init__(self, topology_json, weights, sim_steps=50):
        self.sim_steps = sim_steps

        # Parse node roles from "Inputs"/"Outputs" lists
        self.input_ids  = list(topology_json["Inputs"])   # ordered: input[i] → sensor[i]
        self.output_ids = list(topology_json["Outputs"])  # ordered: output[i] → action[i]
        input_set  = set(self.input_ids)
        output_set = set(self.output_ids)

        all_ids = [n["id"] for n in topology_json["Nodes"]]
        self.hidden_ids = [nid for nid in all_ids
                           if nid not in input_set and nid not in output_set]

        # Edge weight mapping: (from_id, to_id) → weight value
        edges = topology_json["Edges"]
        self.edge_weights = {(e["from"], e["to"]): float(weights[i])
                             for i, e in enumerate(edges)}

        # Precompute incoming connections per non-input node
        lif_ids = self.hidden_ids + self.output_ids
        self.incoming = {nid: [] for nid in lif_ids}
        for (fr, to), w in self.edge_weights.items():
            if to in self.incoming:
                self.incoming[to].append((fr, w))

        # LIF neurons for hidden and output nodes
        self.lif_neurons = {nid: LIFNeuron() for nid in lif_ids}

    # ------------------------------------------------------------------

    def reset_state(self):
        for n in self.lif_neurons.values():
            n.reset()

    def forward(self, sensor_readings):
        sensor_readings = np.clip(sensor_readings, -1.0, 1.0)
        input_vals = {nid: float(sensor_readings[i])
                      for i, nid in enumerate(self.input_ids)}

        # Phase 1: input → hidden, collect spike rates
        hidden_counts = {nid: 0.0 for nid in self.hidden_ids}
        for _ in range(self.sim_steps):
            for nid in self.hidden_ids:
                current = sum(w * input_vals[src]
                              for src, w in self.incoming[nid]
                              if src in input_vals)
                spike = self.lif_neurons[nid].step(current)
                hidden_counts[nid] += spike

        max_spikes = self.sim_steps / 3.0
        hidden_rates = {nid: min(cnt / max_spikes, 1.0)
                        for nid, cnt in hidden_counts.items()}

        # Phase 2: hidden rates (+ direct inputs) → output, collect spike counts
        output_counts = {nid: 0.0 for nid in self.output_ids}
        for _ in range(self.sim_steps):
            for nid in self.output_ids:
                current = 0.0
                for src, w in self.incoming[nid]:
                    if src in hidden_rates:
                        current += w * hidden_rates[src]
                    elif src in input_vals:
                        current += w * input_vals[src]
                spike = self.lif_neurons[nid].step(current)
                output_counts[nid] += spike

        counts = [output_counts[nid] for nid in self.output_ids]

        if sum(counts) == 0:
            # Fallback: linear score from incoming weights
            scores = []
            for nid in self.output_ids:
                s = sum(w * hidden_rates.get(src, input_vals.get(src, 0.0))
                        for src, w in self.incoming[nid])
                scores.append(s)
            action = int(np.argmax(scores)) if any(s != 0 for s in scores) else 0
        else:
            action = int(np.argmax(counts))

        return action, counts
