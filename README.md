# Digital twin policy learning

This repository provides a simplified and user-friendly implementation of a trajectory-based digital twin policy learning framework, illustrated using a facsimile dataset modeled after the COVID-19 booster policy application in the paper.

---

## Repository Structure

- `digital_twin_policy_learning.py`  
  Core implementation of the framework.  
  This module defines the main classes and functions, including:
  - Data preprocessing into trajectory format
  - RNN-based environment simulator
  - Tabular Q-learning algorithm
  - Policy evaluation methods

- `example.ipynb`  
  End-to-end example demonstrating how to use the framework.  
  This notebook:
  - Loads the sample dataset
  - Builds the learning environment
  - Trains or loads the RNN model
  - Runs Q-learning
  - Evaluates multiple policies

- `facsimile_data.csv`  
  A synthetic dataset with the same structure as the original EHR data used in the paper.  
  This dataset is included as the primary example data for demonstrating the generic interface.

- `create_model_ready_data.ipynb`  
  Preprocessing notebook for the facsimile example data.  
  This notebook:
  - Loads `facsimile_data.csv`
  - Harmonizes the column names used by the example workflow
  - Creates derived variables such as `month_index`, `age_cat`, and `months_since_vax_cat`
  - Standardizes age and generates dummy-variable columns for the RNN covariates
  - Outputs `facsimile_model_ready_data.csv` for use in the example pipeline
---

# Main Classes

All core components are defined in `digital_twin_policy_learning.py`.

## 1. `TrajectoryDataset`

This class handles **data preprocessing and representation**.

It converts long-format trajectory data (one row per patient per time step) into:

- padded arrays for RNN training
- per-patient trajectory objects
- mappings from raw variables to discrete RL states

Key functionality:

- `from_long_format(...)`  
  Main entry point. Takes a dataframe and builds the full dataset structure.

- `summary()`  
  Returns dataset statistics (number of patients, sequence length, input/output size, state levels).

This class is responsible for bridging raw data and the learning framework.

---

## 2. `RNNModel`

This is the **sequence model used as the environment simulator**.

It is an LSTM-based neural network that:

- takes patient history as input
- predicts next-step outcomes (e.g., infection risk)

Key methods:

- `forward(x)` → raw logits  
- `predict_proba(x)` → probabilities via sigmoid

After training, this model acts as a **digital twin**, generating simulated outcomes for new action sequences.

---

## 3. `TabularQLearner`

This class implements **tabular Q-learning**.

It maintains a Q-table over discrete states and actions and updates it iteratively.

Key functionality:

- `select_action(state, valid_actions, greedy_only=False)`  
  Chooses an action using epsilon-greedy exploration.

- `update(cur_state, cur_action, reward, next_state)`  
  Applies the standard Q-learning update rule.

The Q-table learned here defines the final policy.

---

## 4. `GenericTrajectoryEnv`

This class defines the **simulation environment** used by the RL agent.

It combines:

- a trained RNN model
- a single patient trajectory
- user-defined rules (reward, transition, constraints)

Key method:

- `step(action)`

This method:

1. updates the trajectory with the chosen action  
2. uses the RNN to predict next-step outcomes  
3. computes the reward  
4. updates the state  
5. determines whether the episode ends  

It returns:

- next state
- reward
- termination flag

This class is the core interaction layer between the RNN and Q-learning.

---

## 5. `MicrosimQLearner`

This is the **main user-facing interface**.

It orchestrates the entire workflow:

- sequence model training/loading
- environment construction
- Q-learning training
- policy evaluation

Key methods:

- `fit_sequence_model(...)`  
  Trains the RNN on trajectory data.

- `load_sequence_model(...)`  
  Loads pretrained RNN weights.

- `fit_tabular_q_learning(...)`  
  Runs Q-learning using the simulated environment.

- `evaluate_policy(policy, epochs)`  
  Evaluates a policy (learned or predefined).

- `simulate(...)`  
  Generates simulated trajectories under a policy.

This class is the primary entry point for users.

---

# How `example.ipynb` Uses the Framework

The example notebook demonstrates a full pipeline:

1. Load `facsimile_data.csv`
2. Construct a `TrajectoryDataset`
3. Initialize `MicrosimQLearner`
4. Train or load the RNN model
5. Run Q-learning
6. Evaluate policies:
   - learned policy
   - observed policy
   - always-treat
   - never-treat

---

# Note on `create_model_ready_data.ipynb`

The notebook `create_model_ready_data.ipynb` is provided to create a model-ready version of the facsimile dataset. In particular, it generates the discrete RL state variables and the processed RNN input columns used by `example.ipynb`. The resulting file, `facsimile_model_ready_data.csv`, is the recommended input for the example pipeline.
