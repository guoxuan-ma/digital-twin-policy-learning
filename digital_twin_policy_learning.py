
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

"""
Generic trajectory-based sequential decision learning interface.

Main user-facing classes:

TrajectoryDataset
    Converts long-format trajectory data into padded arrays for sequence modeling
    and per-patient artifacts for simulation and reinforcement learning.

MicrosimQLearner
    Trains or loads a sequence model, builds a patient-level simulation environment,
    trains tabular Q-learning, and evaluates policies.

Expected input data:

The core input is a long-format pandas DataFrame with one row per subject per time step.
Users must specify:
- a patient identifier column,
- a time-ordering column,
- an action column,
- columns used as RNN covariates,
- columns used as RNN outcomes,
- columns used as discrete RL states.

Optional hooks:

Users may optionally supply:
- reward_fn
- action_constraint_fn
- episode_start_fn
- transition_fn
- terminal_fn

Default behavior:

If reward_fn is None, the reward is the negative predicted value of reward_outcome_col.
If action_constraint_fn is None, all actions in action_space are allowed.
If transition_fn is None, the action column is updated automatically.
If terminal_fn is None, no custom early termination rule is applied.
"""

State = Tuple[int, ...]
RewardFn = Callable[[Dict[str, Any]], float]
ActionConstraintFn = Callable[[Dict[str, Any]], Sequence[int]]
EpisodeStartFn = Callable[[Dict[str, Any]], int]
TransitionFn = Callable[[Dict[str, Any]], np.ndarray]
TerminalFn = Callable[[Dict[str, Any]], bool]
PolicyFn = Callable[[State, Dict[str, Any]], int]


class SequenceDataset(Dataset):
    """Internal PyTorch dataset for sequence-model training.

    Parameters:
    x : np.ndarray
        Padded covariate array of shape (n_patients, max_seq_len, input_size).
    y : np.ndarray
        Padded outcome array of shape (n_patients, max_seq_len, output_size).
    seq_length : np.ndarray
        True sequence lengths before padding.

    Notes:
    The dataset builds a mask over valid time steps so that padded positions do not
    contribute to the sequence-model loss.
    """
    def __init__(self, x: np.ndarray, y: np.ndarray, seq_length: np.ndarray) -> None:
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_length = np.asarray(seq_length, dtype=np.int64)
        self.n, self.t, self.p = self.x.shape
        self.output_size = self.y.shape[2]
        self.seq_mask_y = self._make_mask()

    def _make_mask(self) -> torch.Tensor:
        """Return a mask that marks observed (non-padded) outcome positions."""
        mask = np.zeros((self.n, self.t, self.output_size), dtype=np.float32)
        for i, length in enumerate(self.seq_length):
            mask[i, :length, :] = 1.0
        return torch.tensor(mask, dtype=torch.float32)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        """Return one patient's padded inputs, padded outcomes, and outcome mask."""
        return self.x[idx], self.y[idx], self.seq_mask_y[idx]


class RNNModel(nn.Module):
    """LSTM-based sequence model used as the digital twin engine.

    Parameters:
    input_size : int
        Number of covariates at each time step.
    output_size : int
        Number of predicted outcomes at each time step.
    hidden_size : int, default=128
        Hidden dimension of the LSTM.
    num_layers : int, default=2
        Number of stacked LSTM layers.
    dropout : float, default=0.2
        Dropout applied between LSTM layers when num_layers > 1.

    Notes:
    The forward method returns logits. Use predict_proba to obtain sigmoid-transformed
    probabilities for binary outcomes.
    """
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-time-step outcome logits for an input trajectory batch."""
        out, _ = self.lstm(x)
        return self.fc(out)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-time-step outcome probabilities after sigmoid transformation."""
        return torch.sigmoid(self.forward(x))


class TabularQLearner:
    """Tabular Q-learning agent over a discrete state-action space.

    Parameters:
    state_levels : Sequence[int]
        Number of levels for each discrete state dimension. The Q-table shape becomes
        (*state_levels, n_actions).
    action_space : Sequence[int], default=(0, 1)
        Available discrete actions.
    gamma : float, default=0.99
        Discount factor.
    learning_rate : float, default=0.01
        Initial Q-learning step size.
    learning_rate_decay : float, default=0.998
        Multiplicative decay applied every `decay_every` steps.
    min_learning_rate : float, default=1e-5
        Lower bound for the learning rate.
    epsilon : float, default=0.5
        Initial epsilon for epsilon-greedy exploration.
    epsilon_decay : float, default=0.99
        Multiplicative decay applied every `decay_every` steps.
    decay_every : int, default=5000
        Number of update steps between decay events.
    seed : int, default=2024
        Random seed for action selection.

    Notes:
    The agent only works with discrete RL states. Any application with continuous state
    variables must discretize them before calling tabular Q-learning.
    """
    def __init__(
        self,
        state_levels: Sequence[int],
        action_space: Sequence[int] = (0, 1),
        gamma: float = 0.99,
        learning_rate: float = 0.01,
        learning_rate_decay: float = 0.998,
        min_learning_rate: float = 1e-5,
        epsilon: float = 0.5,
        epsilon_decay: float = 0.99,
        decay_every: int = 5000,
        seed: int = 2024,
    ) -> None:
        self.state_levels = tuple(int(v) for v in state_levels)
        self.action_space = tuple(int(a) for a in action_space)
        self.action_to_index = {a: i for i, a in enumerate(self.action_space)}
        self.q_table = np.zeros(self.state_levels + (len(self.action_space),), dtype=np.float32)

        self.gamma = float(gamma)
        self.learning_rate = float(learning_rate)
        self.learning_rate_decay = float(learning_rate_decay)
        self.min_learning_rate = float(min_learning_rate)
        self.epsilon = float(epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.decay_every = int(decay_every)
        self.steps = 0
        self.rng = np.random.default_rng(seed)

    def select_action(
        self,
        state: State,
        valid_actions: Sequence[int],
        greedy_only: bool = False,
    ) -> int:
        """Choose an action from valid_actions using epsilon-greedy exploration."""

        valid_actions = [int(a) for a in valid_actions]
        if len(valid_actions) == 0:
            raise ValueError("valid_actions cannot be empty.")

        valid_idx = [self.action_to_index[a] for a in valid_actions]

        if (not greedy_only) and (self.rng.uniform() < self.epsilon):
            return int(self.rng.choice(valid_actions))

        q_vals = self.q_table[state]
        best_local_idx = valid_idx[int(np.argmax(q_vals[valid_idx]))]
        return int(self.action_space[best_local_idx])

    def update(self, cur_state: State, cur_action: int, reward: float, next_state: State) -> None:
        """Apply one tabular Q-learning update to the Q-table."""
        a_idx = self.action_to_index[int(cur_action)]
        self.q_table[cur_state + (a_idx,)] = (
            self.q_table[cur_state + (a_idx,)]
            + self.learning_rate
            * (
                reward
                + self.gamma * np.max(self.q_table[next_state])
                - self.q_table[cur_state + (a_idx,)]
            )
        )
        self.steps += 1
        if self.steps % self.decay_every == 0:
            self.learning_rate = max(self.min_learning_rate, self.learning_rate * self.learning_rate_decay)
            self.epsilon *= self.epsilon_decay



@dataclass
class TrajectoryPatientArtifacts:
    """Per-patient trajectory representation used during simulation.

    Attributes:
    patient_id : Any
        Subject identifier.
    times : np.ndarray
        Time values for the subject trajectory.
    rnn_inputs : np.ndarray
        Per-time-step covariates used by the sequence model.
    rnn_outcomes : np.ndarray
        Observed outcomes aligned with the trajectory.
    observed_actions : np.ndarray
        Observed historical action sequence.
    rl_state_raw : pd.DataFrame
        Raw RL state columns before mapping to integer indices.
    rl_state_idx_seq : np.ndarray
        Integer-encoded RL state sequence.
    action_positions : np.ndarray
        Indices where the observed action equals 1.
    history_start_idx : int
        Index at which simulation begins for this patient.
    """
    patient_id: Any
    times: np.ndarray
    rnn_inputs: np.ndarray
    rnn_outcomes: np.ndarray
    observed_actions: np.ndarray
    rl_state_raw: pd.DataFrame
    rl_state_idx_seq: np.ndarray
    action_positions: np.ndarray
    history_start_idx: int


@dataclass
class EnvironmentConfig:
    """Configuration object used by GenericTrajectoryEnv.

    This object collects column mappings, RL state metadata, action-space settings,
    and optional hook functions so that the environment can be constructed consistently
    for each patient trajectory.
    """
    action_col: str
    action_col_idx: int
    rl_state_cols: Sequence[str]
    rl_state_maps: Dict[str, Dict[Any, int]]
    outcome_indices: Dict[str, int]
    reward_outcome_col: str
    reward_outcome_idx: int
    feature_col_index: Dict[str, int]
    action_space: Sequence[int] = (0, 1)
    cumulative_action_col: Optional[str] = None
    reward_fn: Optional[RewardFn] = None
    action_constraint_fn: Optional[ActionConstraintFn] = None
    episode_start_fn: Optional[EpisodeStartFn] = None
    transition_fn: Optional[TransitionFn] = None
    terminal_fn: Optional[TerminalFn] = None


@dataclass
class PolicyEvaluationResult:
    """Container for policy evaluation outputs."""
    rewards: np.ndarray
    q_table: Optional[np.ndarray] = None


class GenericTrajectoryEnv:
    """Patient-level microsimulation environment driven by a trained sequence model.

    Parameters:
    rnn_model : RNNModel
        Trained sequence model used to predict next-step outcomes.
    patient : TrajectoryPatientArtifacts
        One patient's trajectory artifacts.
    config : EnvironmentConfig
        Environment metadata and optional hook functions.
    device : str, default="cpu"
        Torch device used for model prediction.
    action_history : np.ndarray, optional
        If provided, overrides the observed action history used to initialize the
        simulation path.

    Notes:
    The environment maintains a simulated action history and covariate path. At each
    step it updates the path, obtains next-step outcome predictions from the sequence
    model, computes a reward, and optionally checks a terminal rule.
    """
    def __init__(
        self,
        rnn_model: RNNModel,
        patient: TrajectoryPatientArtifacts,
        config: EnvironmentConfig,
        device: str = "cpu",
        action_history: Optional[np.ndarray] = None,
    ) -> None:
        self.rnn = rnn_model
        self.patient = patient
        self.config = config
        self.device = device

        self.total_steps = patient.rnn_inputs.shape[0]
        self.history_len = int(patient.history_start_idx + 1)

        if action_history is None:
            self.action_history = patient.observed_actions[: self.history_len].astype(int).copy()
        else:
            self.action_history = np.asarray(action_history, dtype=int).copy()

        self.path_x = patient.rnn_inputs[: self.history_len].copy().astype(np.float32)
        self.current_step = self.history_len - 1

        self.last_predicted_risk: Optional[np.ndarray] = None
        self.last_reward: Optional[float] = None
        self.done: bool = False
        self._refresh_state_from_last_row()

    def _map_state_value(self, col: str, value: Any) -> int:
        mapping = self.config.rl_state_maps[col]
        if value in mapping:
            return int(mapping[value])

        keys = list(mapping.keys())
        try:
            numeric_keys = np.array([float(k) for k in keys], dtype=float)
            numeric_value = float(value)
            nearest = keys[int(np.argmin(np.abs(numeric_keys - numeric_value)))]
            return int(mapping[nearest])
        except Exception:
            return int(mapping[keys[0]])

    def _refresh_state_from_last_row(self) -> None:
        raw_row = self.patient.rl_state_raw.iloc[self.current_step]
        state_list = []

        for col in self.config.rl_state_cols:
            if col == self.config.cumulative_action_col:
                state_list.append(self._map_state_value(col, int(self.action_history.sum())))

            elif col == self.config.action_col:
                state_list.append(self._map_state_value(col, int(self.action_history[-1])))

            else:
                state_list.append(self._map_state_value(col, raw_row[col]))

        self.tq_state = tuple(state_list)

    def _default_transition(self, base_next_row: np.ndarray, action: int) -> np.ndarray:
        row = base_next_row.copy()
        row[self.config.action_col_idx] = float(action)

        if self.config.cumulative_action_col is not None:
            idx = self.config.feature_col_index[self.config.cumulative_action_col]
            row[idx] = float(self.action_history.sum())

        return row

    def get_valid_actions(self) -> Sequence[int]:
        if self.config.action_constraint_fn is None:
            return list(self.config.action_space)

        context = {
            "env": self,
            "patient": self.patient,
            "current_step": self.current_step,
            "action_history": self.action_history.copy(),
            "state": self.tq_state,
        }
        return list(self.config.action_constraint_fn(context))

    def step(self, action: int) -> Tuple[np.ndarray, State, float, bool]:
        """Advance the environment by one time step.

        Parameters:
        action : int
            Action applied at the current decision point.

        Returns:
        next_row : np.ndarray
            Updated covariate row appended to the simulated path.
        next_state : State
            Discrete RL state after the transition.
        reward : float
            Reward assigned to the transition.
        done : bool
            Whether the episode terminates after this step.

        Default behavior:
        - If transition_fn is None, the environment updates the action column automatically
          and also updates cumulative_action_col when that column is configured.
        - If reward_fn is None, the reward is `-predicted_outcomes[reward_outcome_idx]`.
        - If terminal_fn is None, no custom early termination rule is applied.
        """
        if self.current_step >= self.total_steps - 1:
            self.done = True
            return self.path_x[-1].copy(), self.tq_state, 0.0, True

        next_step = self.current_step + 1
        base_next_row = self.patient.rnn_inputs[next_step].copy().astype(np.float32)

        self.action_history = np.append(self.action_history, int(action))

        if self.config.transition_fn is None:
            next_row = self._default_transition(base_next_row, int(action))
        else:
            next_row = self.config.transition_fn(
                {
                    "env": self,
                    "base_next_row": base_next_row,
                    "action": int(action),
                    "next_step": next_step,
                }
            ).astype(np.float32)

        self.path_x = np.vstack([self.path_x, next_row.reshape(1, -1)])
        self.current_step = next_step
        self._refresh_state_from_last_row()

        with torch.no_grad():
            x = torch.tensor(self.path_x, dtype=torch.float32, device=self.device).unsqueeze(0)
            risk = self.rnn.predict_proba(x)[0, -1, :].detach().cpu().numpy()

        self.last_predicted_risk = risk.copy()

        reward_context = {
            "env": self,
            "patient": self.patient,
            "action": int(action),
            "predicted_outcomes": risk.copy(),
            "reward_outcome_idx": self.config.reward_outcome_idx,
            "reward_outcome_col": self.config.reward_outcome_col,
            "current_step": self.current_step,
            "state": self.tq_state,
        }

        # Default reward: negative predicted value of the selected reward_outcome_col.
        if self.config.reward_fn is None:
            reward = -float(risk[self.config.reward_outcome_idx])
        else:
            reward = float(self.config.reward_fn(reward_context))

        self.last_reward = reward

        if self.config.terminal_fn is None:
            done = False
        else:
            done = bool(
                self.config.terminal_fn(
                    {
                        "env": self,
                        "patient": self.patient,
                        "action": int(action),
                        "predicted_outcomes": risk.copy(),
                        "current_step": self.current_step,
                        "state": self.tq_state,
                    }
                )
            )

        self.done = done
        return self.path_x[-1].copy(), self.tq_state, reward, done


class TrajectoryDataset:
    """Container for trajectory data prepared for sequence modeling and RL.

    This class stores both:
    1. padded arrays for sequence-model training
    2. per-patient trajectory artifacts used by the simulation environment

    Users can create this object through `TrajectoryDataset.from_long_format(...)`.
    """
    def __init__(self, seed: int = 2024) -> None:
        self.seed = int(seed)
        self.long_format_df: Optional[pd.DataFrame] = None
        self.patients: List[TrajectoryPatientArtifacts] = []

        self.covariates_rnn: Optional[np.ndarray] = None
        self.outcomes_rnn: Optional[np.ndarray] = None
        self.seq_length: Optional[np.ndarray] = None

        self.patient_id_col: Optional[str] = None
        self.time_col: Optional[str] = None
        self.action_col: Optional[str] = None

        self.rnn_covariate_cols: List[str] = []
        self.rnn_outcome_cols: List[str] = []
        self.rl_state_cols: List[str] = []

        self.feature_col_index: Dict[str, int] = {}
        self.outcome_col_index: Dict[str, int] = {}
        self.rl_state_maps: Dict[str, Dict[Any, int]] = {}

        self.cumulative_action_col: Optional[str] = None
        self.reward_outcome_col: Optional[str] = None

    @staticmethod
    def _build_state_maps(df: pd.DataFrame, rl_state_cols: Sequence[str]) -> Dict[str, Dict[Any, int]]:
        """Map each unique RL state value to an integer index for tabular Q-learning."""
        state_maps: Dict[str, Dict[Any, int]] = {}
        for col in rl_state_cols:
            vals = pd.Series(df[col]).dropna().unique().tolist()
            try:
                vals = sorted(vals)
            except Exception:
                vals = list(vals)
            state_maps[col] = {v: i for i, v in enumerate(vals)}
        return state_maps

    @staticmethod
    def _default_history_start_idx(actions: np.ndarray) -> int:
        """Default rule for where simulation begins within a trajectory.

        By default, simulation begins at index 0 so each episode starts from the
        beginning of the available trajectory history.
        """
        return 0

    @classmethod
    def from_long_format(
        cls,
        df: pd.DataFrame,
        patient_id_col: str,
        time_col: str,
        action_col: str,
        rnn_covariate_cols: Sequence[str],
        rnn_outcome_cols: Sequence[str],
        rl_state_cols: Sequence[str],
        cumulative_action_col: Optional[str] = None,
        reward_outcome_col: Optional[str] = None,
        episode_start_fn: Optional[EpisodeStartFn] = None,
        seed: int = 2024,
    ) -> "TrajectoryDataset":
        """Build a TrajectoryDataset from a long-format pandas DataFrame.

        Parameters:
        df : pd.DataFrame
            Long-format input data with one row per patient per time step.
        patient_id_col : str
            Column identifying the patient / subject.
        time_col : str
            Column defining within-patient time order.
        action_col : str
            Action column. This column must also be included in `rnn_covariate_cols`.
        rnn_covariate_cols : Sequence[str]
            Columns used as input covariates for the sequence model.
        rnn_outcome_cols : Sequence[str]
            Columns used as outcomes for the sequence model.
        rl_state_cols : Sequence[str]
            Columns defining the discrete RL state used by tabular Q-learning.
        cumulative_action_col : str, optional
            Covariate column that should track cumulative actions and be updated during
            default simulation transitions.
        reward_outcome_col : str, optional
            Outcome column used by the default reward function. If omitted, the first
            outcome column in `rnn_outcome_cols` is used.
        episode_start_fn : callable, optional
            Custom function that determines the simulation start index for each patient.
        seed : int, default=2024
            Random seed stored on the dataset object.

        Returns:
        TrajectoryDataset
            Prepared dataset containing padded arrays, patient-level artifacts, column
            mappings, and discrete RL state maps.
        """
        obj = cls(seed=seed)

        required = {patient_id_col, time_col, action_col, *rnn_covariate_cols, *rnn_outcome_cols, *rl_state_cols}
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if action_col not in rnn_covariate_cols:
            raise ValueError("action_col must be included in rnn_covariate_cols.")

        work = df.copy().sort_values([patient_id_col, time_col]).reset_index(drop=True)

        obj.long_format_df = work.copy()
        obj.patient_id_col = patient_id_col
        obj.time_col = time_col
        obj.action_col = action_col
        obj.rnn_covariate_cols = list(rnn_covariate_cols)
        obj.rnn_outcome_cols = list(rnn_outcome_cols)
        obj.rl_state_cols = list(rl_state_cols)
        obj.cumulative_action_col = cumulative_action_col
        obj.reward_outcome_col = reward_outcome_col or rnn_outcome_cols[0]

        obj.feature_col_index = {c: i for i, c in enumerate(obj.rnn_covariate_cols)}
        obj.outcome_col_index = {c: i for i, c in enumerate(obj.rnn_outcome_cols)}
        obj.rl_state_maps = cls._build_state_maps(work, obj.rl_state_cols)

        seq_x_list = []
        seq_y_list = []
        seq_len = []

        for patient_id, pat_df in work.groupby(patient_id_col, sort=False):
            pat_df = pat_df.sort_values(time_col).reset_index(drop=True)
            x = pat_df[obj.rnn_covariate_cols].to_numpy(dtype=np.float32)
            y = pat_df[obj.rnn_outcome_cols].to_numpy(dtype=np.float32)
            a = pat_df[action_col].to_numpy(dtype=int)

            rl_state_idx = np.column_stack(
                [pat_df[col].map(obj.rl_state_maps[col]).to_numpy(dtype=int) for col in obj.rl_state_cols]
            )

            if episode_start_fn is None:
                history_start_idx = cls._default_history_start_idx(a)
            else:
                history_start_idx = int(
                    episode_start_fn(
                        {
                            "patient_id": patient_id,
                            "patient_df": pat_df.copy(),
                            "actions": a.copy(),
                        }
                    )
                )

            obj.patients.append(
                TrajectoryPatientArtifacts(
                    patient_id=patient_id,
                    times=pat_df[time_col].to_numpy(),
                    rnn_inputs=x,
                    rnn_outcomes=y,
                    observed_actions=a,
                    rl_state_raw=pat_df[obj.rl_state_cols].copy(),
                    rl_state_idx_seq=rl_state_idx,
                    action_positions=np.where(a == 1)[0].astype(int),
                    history_start_idx=history_start_idx,
                )
            )

            seq_x_list.append(x)
            seq_y_list.append(y)
            seq_len.append(x.shape[0])

        max_len = max(seq_len)
        n = len(seq_x_list)
        p = len(obj.rnn_covariate_cols)
        q = len(obj.rnn_outcome_cols)

        obj.covariates_rnn = np.zeros((n, max_len, p), dtype=np.float32)
        obj.outcomes_rnn = np.zeros((n, max_len, q), dtype=np.float32)
        obj.seq_length = np.asarray(seq_len, dtype=np.int64)

        for i, (x, y) in enumerate(zip(seq_x_list, seq_y_list)):
            obj.covariates_rnn[i, : x.shape[0], :] = x
            obj.outcomes_rnn[i, : y.shape[0], :] = y

        return obj

    def summary(self) -> Dict[str, Any]:
        """Return basic dataset dimensions useful for debugging and reporting."""
        return {
            "n_patients": len(self.patients),
            "max_seq_len": int(self.covariates_rnn.shape[1]),
            "input_size": int(self.covariates_rnn.shape[2]),
            "output_size": int(self.outcomes_rnn.shape[2]),
            "rl_state_levels": tuple(len(self.rl_state_maps[c]) for c in self.rl_state_cols),
        }


class MicrosimQLearner:
    """Main user-facing interface for sequence modeling, simulation, and Q-learning.

    Parameters:
    dataset : TrajectoryDataset
        Prepared trajectory dataset.
    device : str, optional
        Torch device used for sequence-model training and prediction. If omitted,
        CUDA is used when available, otherwise CPU.
    seed : int, default=2024
        Random seed used for numpy, Python random, and torch.
    action_space : Sequence[int], default=(0, 1)
        Discrete action space used by the environment and tabular Q-learning.
    reward_fn : callable, optional
        Custom reward function. If None, the default reward is
        `-predicted_outcomes[reward_outcome_idx]`.
    action_constraint_fn : callable, optional
        Function returning valid actions at each state. If None, all actions in
        action_space are allowed.
    episode_start_fn : callable, optional
        Stored in the environment config for reference; the actual per-patient start
        indices are usually determined when creating the dataset.
    transition_fn : callable, optional
        Custom transition rule for updating the next covariate row.
    terminal_fn : callable, optional
        Custom terminal condition. If None, no custom early termination rule is used.
    """
    def __init__(
        self,
        dataset: TrajectoryDataset,
        device: Optional[str] = None,
        seed: int = 2024,
        action_space: Sequence[int] = (0, 1),
        reward_fn: Optional[RewardFn] = None,
        action_constraint_fn: Optional[ActionConstraintFn] = None,
        episode_start_fn: Optional[EpisodeStartFn] = None,
        transition_fn: Optional[TransitionFn] = None,
        terminal_fn: Optional[TerminalFn] = None,
    ) -> None:
        self.dataset = dataset
        self.seed = int(seed)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.rnn_model: Optional[RNNModel] = None
        self.loss_history: List[float] = []
        self.q_learner: Optional[TabularQLearner] = None

        self.env_config = EnvironmentConfig(
            action_col=self.dataset.action_col,
            action_col_idx=self.dataset.feature_col_index[self.dataset.action_col],
            rl_state_cols=self.dataset.rl_state_cols,
            rl_state_maps=self.dataset.rl_state_maps,
            outcome_indices=self.dataset.outcome_col_index,
            reward_outcome_col=self.dataset.reward_outcome_col,
            reward_outcome_idx=self.dataset.outcome_col_index[self.dataset.reward_outcome_col],
            feature_col_index=self.dataset.feature_col_index,
            action_space=tuple(action_space),
            cumulative_action_col=self.dataset.cumulative_action_col,
            reward_fn=reward_fn,
            action_constraint_fn=action_constraint_fn,
            episode_start_fn=episode_start_fn,
            transition_fn=transition_fn,
            terminal_fn=terminal_fn,
        )

    def fit_sequence_model(
        self,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        epochs: int = 2000,
        lr: float = 1e-4,
        batch_size: int = 32,
        verbose_every: int = 100,
    ) -> Dict[str, Any]:
        """Train the sequence model on the prepared padded trajectory arrays.

        Returns:
        Dict[str, Any]
            Summary of the fitted model configuration and final training loss.
        """
        dataset = SequenceDataset(
            self.dataset.covariates_rnn,
            self.dataset.outcomes_rnn,
            self.dataset.seq_length,
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = RNNModel(
            input_size=self.dataset.covariates_rnn.shape[2],
            output_size=self.dataset.outcomes_rnn.shape[2],
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.loss_history = []
        model.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            for x_batch, y_batch, seq_mask_y_batch in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                seq_mask_y_batch = seq_mask_y_batch.to(self.device).bool()

                logits = model(x_batch)
                logits_masked = logits[seq_mask_y_batch]
                targets_masked = y_batch[seq_mask_y_batch]

                loss = nn.functional.binary_cross_entropy(
                    torch.sigmoid(logits_masked),
                    targets_masked,
                    reduction="mean",
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.item())
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            self.loss_history.append(avg_loss)

            if verbose_every and ((epoch + 1) % verbose_every == 0 or epoch == 0 or epoch + 1 == epochs):
                print(f"[fit_sequence_model] epoch {epoch + 1:4d}/{epochs} | loss={avg_loss:.6f}")

        self.rnn_model = model
        return {
            "final_loss": self.loss_history[-1],
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "epochs": epochs,
            "lr": lr,
        }

    def load_sequence_model(
        self,
        model_path: str,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        """Load pretrained sequence-model weights into an RNNModel instance."""
        model = RNNModel(
            input_size=self.dataset.covariates_rnn.shape[2],
            output_size=self.dataset.outcomes_rnn.shape[2],
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        ).to(self.device)
        state = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state)
        model.eval()
        self.rnn_model = model

    def build_env(self, patient_index: int, action_history: Optional[np.ndarray] = None) -> GenericTrajectoryEnv:
        """Construct a GenericTrajectoryEnv for one patient trajectory."""
        if self.rnn_model is None:
            raise ValueError("Train or load the sequence model first.")
        return GenericTrajectoryEnv(
            rnn_model=self.rnn_model,
            patient=self.dataset.patients[patient_index],
            config=self.env_config,
            device=self.device,
            action_history=action_history,
        )
    
    def save_q_table(self, q_table_path: str) -> None:
        """Save the currently learned Q-table to a .npy file."""
        if self.q_learner is None:
            raise ValueError("No trained Q-learning agent found. Train or load a Q-table first.")
        np.save(q_table_path, self.q_learner.q_table)

    def load_q_table(self, q_table_path: str) -> None:
        """Load a previously saved Q-table into a TabularQLearner instance."""
        state_levels = tuple(len(self.dataset.rl_state_maps[c]) for c in self.dataset.rl_state_cols)

        q_table = np.load(q_table_path)

        expected_shape = state_levels + (len(self.env_config.action_space),)
        if q_table.shape != expected_shape:
            raise ValueError(
                f"Loaded Q-table has shape {q_table.shape}, but expected {expected_shape} "
                f"based on the current dataset RL state levels and action space."
            )

        agent = TabularQLearner(
            state_levels=state_levels,
            action_space=self.env_config.action_space,
            seed=self.seed,
        )
        agent.q_table = q_table.astype(np.float32, copy=False)
        self.q_learner = agent

    def _resolve_policy_action(
        self,
        policy: Union[str, PolicyFn],
        env: GenericTrajectoryEnv,
        patient: TrajectoryPatientArtifacts,
        state: State,
        t: int,
    ) -> int:
        valid_actions = env.get_valid_actions()

        if callable(policy):
            return int(policy(state, {"env": env, "patient": patient, "t": t, "valid_actions": valid_actions}))

        if policy == "observed":
            return int(patient.observed_actions[t])
        if policy == "none":
            return 0
        if policy == "all":
            return max(valid_actions)
        if policy == "learned":
            if self.q_learner is None:
                raise ValueError("Train Q-learning first.")
            return int(self.q_learner.select_action(state, valid_actions=valid_actions, greedy_only=True))

        raise ValueError("Unsupported policy.")

    def simulate(
        self,
        n: Optional[int] = None,
        policy: Union[str, PolicyFn] = "observed",
    ) -> pd.DataFrame:
        """Simulate trajectories under a specified policy and return a row-level DataFrame.

        Parameters:
        n : int, optional
            Number of patients to simulate. If omitted, all patients are used.
        policy : {"observed", "none", "all", "learned"} or callable
            Policy specification. A callable should accept `(state, context)` and return
            an action.
        """
        n_patients = len(self.dataset.patients) if n is None else min(int(n), len(self.dataset.patients))
        out_rows = []

        for patient_index in range(n_patients):
            patient = self.dataset.patients[patient_index]
            env = self.build_env(patient_index)
            state = tuple(env.tq_state)

            for t in range(patient.history_start_idx, patient.rnn_inputs.shape[0]):
                action = self._resolve_policy_action(policy, env, patient, state, t)
                _, next_state, reward, done = env.step(action)

                row = {
                    "patient_id": patient.patient_id,
                    "t": t,
                    "action": int(action),
                    "reward": float(reward),
                }
                for j, col in enumerate(self.dataset.rl_state_cols):
                    row[f"state_{col}"] = int(next_state[j])

                if env.last_predicted_risk is not None:
                    for outcome_name, idx in self.dataset.outcome_col_index.items():
                        row[f"pred_{outcome_name}"] = float(env.last_predicted_risk[idx])

                out_rows.append(row)
                state = tuple(next_state)

                if done:
                    break

        return pd.DataFrame(out_rows)

    def fit_tabular_q_learning(
        self,
        repeats_train_eval: int = 30,
        gamma: float = 0.99,
        learning_rate: float = 0.01,
        learning_rate_decay: float = 0.998,
        min_learning_rate: float = 1e-5,
        epsilon: float = 0.5,
        epsilon_decay: float = 0.99,
        decay_every: int = 5000,
        save_q_table_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Train tabular Q-learning using the simulation environment.

        Notes:
        Training alternates between one training pass and one greedy evaluation pass,
        repeated `repeats_train_eval` times. The returned `epoch_reward_list` stores
        rewards only for the evaluation passes.
        """
        state_levels = tuple(len(self.dataset.rl_state_maps[c]) for c in self.dataset.rl_state_cols)

        agent = TabularQLearner(
            state_levels=state_levels,
            action_space=self.env_config.action_space,
            gamma=gamma,
            learning_rate=learning_rate,
            learning_rate_decay=learning_rate_decay,
            min_learning_rate=min_learning_rate,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            decay_every=decay_every,
            seed=self.seed,
        )

        epochs_train_eval = np.tile(np.repeat([False, True], [1, 1]), repeats_train_eval)
        epoch_reward_list = np.full((len(epochs_train_eval), len(self.dataset.patients)), np.nan, dtype=np.float32)

        for epoch, is_train in enumerate(epochs_train_eval):
            sample_idx_array = np.random.choice(len(self.dataset.patients), size=len(self.dataset.patients))

            for i, sample_idx in enumerate(sample_idx_array):
                patient = self.dataset.patients[int(sample_idx)]
                env = self.build_env(int(sample_idx))
                state = tuple(env.tq_state)
                episodic_reward = 0.0
                horizon = max(1, patient.rnn_inputs.shape[0] - patient.history_start_idx)

                for t in range(patient.history_start_idx, patient.rnn_inputs.shape[0]):
                    if t == patient.history_start_idx:
                        action = int(patient.observed_actions[t])
                    else:
                        valid_actions = env.get_valid_actions()
                        action = agent.select_action(
                            state,
                            valid_actions=valid_actions,
                            greedy_only=(not is_train),
                        )

                    _, next_state, reward, done = env.step(action)
                    episodic_reward += reward / horizon

                    if is_train and t != patient.history_start_idx:
                        agent.update(state, action, reward, next_state)

                    state = tuple(next_state)
                    if done:
                        break

                if not is_train:
                    epoch_reward_list[epoch, i] = episodic_reward

        self.q_learner = agent
        if save_q_table_path is not None:
            np.save(save_q_table_path, agent.q_table)
        return {"q_table": agent.q_table.copy(), "epoch_reward_list": epoch_reward_list}

    def evaluate_policy(
        self,
        policy: Union[str, PolicyFn],
        epochs: int = 5,
    ) -> np.ndarray:
        """Evaluate a policy over multiple resampled patient-order passes.

        Parameters:
        policy : {"observed", "none", "all", "learned"} or callable
            Policy to evaluate.
        epochs : int, default=5
            Number of evaluation passes.

        Returns:
        np.ndarray
            Array of shape (epochs, n_patients) containing episodic rewards.
        """
        out = np.full((epochs, len(self.dataset.patients)), np.nan, dtype=np.float32)

        for epoch in range(epochs):
            sample_idx_array = np.random.choice(len(self.dataset.patients), size=len(self.dataset.patients))

            for i, sample_idx in enumerate(sample_idx_array):
                patient = self.dataset.patients[int(sample_idx)]
                env = self.build_env(int(sample_idx))
                state = tuple(env.tq_state)
                episodic_reward = 0.0
                horizon = max(1, patient.rnn_inputs.shape[0] - patient.history_start_idx)

                for t in range(patient.history_start_idx, patient.rnn_inputs.shape[0]):
                    action = self._resolve_policy_action(policy, env, patient, state, t)
                    _, next_state, reward, done = env.step(action)
                    episodic_reward += reward / horizon
                    state = tuple(next_state)
                    if done:
                        break

                out[epoch, i] = episodic_reward

        return out

