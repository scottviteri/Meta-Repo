uter-Policy Experiments

A repository design and implementation guide for testing the behavioral differences between **argmax** and **softmax** outer policies that act to maximize **empowerment**.

---

## 1. Purpose

This repo is for one question:

> How much of “empowerment behavior” comes from the empowerment objective itself, and how much comes from the **outer control rule** used to act on empowerment estimates?

The core experimental move is to **hold the empowerment computation fixed** and vary **only** the outer policy used to select actions.

Concretely, the repo should compare at least:

- **Greedy / argmax outer control**
- **Softmax / Boltzmann outer control**
- optionally **epsilon-greedy** as a control

The main comparison should isolate the following distinction:

- **Inner optimization:** the definition of empowerment at a state
- **Outer optimization:** the behavior policy used to choose actions given empowerment-derived action scores

The paper `Complex behavior from intrinsic motivation to occupy future action-state path space` is a useful motivation because it compares MOP’s stochastic policy against an empowerment controller that is explicitly greedy over successor-state empowerment. This repo is meant to make that comparison more symmetric and controlled.

---

## 2. Scientific framing

### 2.1 Inner object: empowerment

For an environment state `s` and horizon `n`, define empowerment as

```math
E_n(s) = \max_{q(a_{0:n-1} \mid s)} I(A_{0:n-1}; S_n \mid s)
```

This is a property of the local action-to-future-state channel induced by the dynamics.

### 2.2 Outer object: action selection using empowerment

Given empowerment estimates for successor states, define an empowerment-derived action score

```math
Q_emp(s, a) = \sum_{s'} P(s' \mid s, a) E_n(s')
```

In a deterministic environment:

```math
Q_emp(s, a) = E_n(f(s, a))
```

Then compare different outer policies:

#### Greedy / argmax

```math
\pi_{arg}(a \mid s) = \frac{\mathbf{1}[a \in \arg\max_b Q_emp(s,b)]}{|\arg\max_b Q_emp(s,b)|}
```

#### Softmax / Boltzmann

```math
\pi_\tau(a \mid s) = \frac{\exp(Q_emp(s,a)/\tau)}{\sum_b \exp(Q_emp(s,b)/\tau)}
```

#### Optional epsilon-greedy control

```math
\pi_\epsilon(a \mid s) =
\begin{cases}
1-\epsilon + \epsilon/|A(s)| & a \in \arg\max_b Q_emp(s,b) \\
\epsilon/|A(s)| & \text{otherwise}
\end{cases}
```

This makes the central experimental variable explicit:

> **Same inner empowerment estimate, different outer control law.**

---

## 3. Core hypotheses

The repo should be organized around a few testable hypotheses.

### H1. Greedy empowerment is behaviorally narrow

Argmax outer control will often collapse into stereotyped loops, local maxima, or repeated visits to a small set of high-empowerment states.

### H2. Softmax outer control improves basin escape

At moderate temperatures, softmax will more often cross temporary low-empowerment bottlenecks and reach better long-run empowerment basins.

### H3. Softmax trades off peak empowerment for diversity

Softmax may sometimes lower the mean instantaneous empowerment visited while increasing:

- state visitation entropy
- trajectory diversity
- number of distinct states visited
- strategy diversity across episodes

### H4. Very high temperature hurts control quality

As temperature increases, the policy becomes overly diffuse and performance degrades on survival or long-run empowerment metrics.

### H5. Some previously reported “empowerment behavior” is really outer-policy behavior

Behavioral conclusions about empowerment may be substantially altered by the outer policy used to realize it.

---

## 4. Recommended repository layout

```text
empowerment-policy-study/
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── envs/
│   │   ├── corridor.yaml
│   │   ├── four_rooms_energy.yaml
│   │   ├── noisy_room.yaml
│   │   └── cartpole_discrete.yaml
│   ├── algs/
│   │   ├── greedy.yaml
│   │   ├── softmax.yaml
│   │   ├── epsilon_greedy.yaml
│   │   └── softmax_sweep.yaml
│   └── experiments/
│       ├── exact_tabular_baselines.yaml
│       ├── corridor_temperature_sweep.yaml
│       ├── four_rooms_reproduction.yaml
│       └── cartpole_behavior.yaml
├── src/
│   ├── __init__.py
│   ├── envs/
│   │   ├── base.py
│   │   ├── corridor.py
│   │   ├── four_rooms_energy.py
│   │   ├── noisy_room.py
│   │   └── cartpole_discrete.py
│   ├── empowerment/
│   │   ├── channel.py
│   │   ├── exact.py
│   │   ├── blahut_arimoto.py
│   │   ├── nstep_model.py
│   │   └── score.py
│   ├── policies/
│   │   ├── base.py
│   │   ├── greedy.py
│   │   ├── softmax.py
│   │   └── epsilon_greedy.py
│   ├── rollout/
│   │   ├── simulate.py
│   │   ├── logging.py
│   │   └── seeds.py
│   ├── metrics/
│   │   ├── visitation.py
│   │   ├── trajectories.py
│   │   ├── empowerment_metrics.py
│   │   └── survival.py
│   ├── analysis/
│   │   ├── aggregate.py
│   │   ├── plots.py
│   │   └── statistical_tests.py
│   └── utils/
│       ├── math.py
│       ├── config.py
│       └── serialization.py
├── scripts/
│   ├── run_experiment.py
│   ├── compute_empowerment_table.py
│   ├── rollout_policy.py
│   ├── sweep_temperature.py
│   └── make_figures.py
├── notebooks/
│   ├── 01_corridor_debug.ipynb
│   ├── 02_four_rooms_reproduction.ipynb
│   ├── 03_temperature_phase_diagram.ipynb
│   └── 04_cartpole_behavior.ipynb
├── tests/
│   ├── test_blahut_arimoto.py
│   ├── test_empowerment_exact.py
│   ├── test_softmax_policy.py
│   ├── test_metrics.py
│   └── test_env_dynamics.py
└── results/
    ├── raw/
    ├── processed/
    └── figures/
```

---

## 5. Minimal implementation plan

### Phase 1. Exact tabular experiments

Implement the simplest possible exact version first.

Requirements:

- finite state spaces
- finite action spaces
- known transition matrices
- exact n-step empowerment computation
- no function approximation

This phase should answer the scientific question cleanly before any neural approximations are introduced.

### Phase 2. Approximate / scalable experiments

Only after the tabular experiments are working:

- add variational or sampled empowerment estimation
- add continuous or discretized continuous environments
- optionally add learning of dynamics or amortized empowerment estimators

The repo should make Phase 1 the default path for reproduction.

---

## 6. Environment suite

The environments should be chosen to make the difference between argmax and softmax visible.

### 6.1 Corridor-to-better-basin

This should be the first environment.

#### Design goal

Create a small deterministic gridworld with:

- one nearby region with **moderately high empowerment**
- one farther region with **higher empowerment**
- a narrow corridor of temporarily lower-empowerment states between them

#### Why it matters

This is the cleanest test of whether softmax helps escape local empowerment maxima.

#### Expected result

- argmax gets stuck in the nearby basin
- moderate-temperature softmax reaches the farther high-empowerment hub more often

### 6.2 Four-rooms with energy / food

This should reproduce the qualitative setup used in the MOP paper, but with a symmetric empowerment comparison.

#### Design goal

A gridworld with:

- finite energy
- food locations that replenish energy
- absorbing death states at zero energy
- doorways and rooms creating spatial bottlenecks

#### Why it matters

This environment introduces survival pressure, and it is close to the setup in which greedy empowerment was reported to alternate between food and local room-center states.

#### Expected result

- argmax exhibits repetitive loops
- softmax exhibits broader exploration and more path diversity
- both may retain survival-seeking behavior under energy constraints

### 6.3 Noisy room

Add one room where transitions are stochastic.

#### Why it matters

This tests how soft outer control interacts with states that are noisy versus controllable.

#### Expected result

Depending on empowerment horizon and score construction:

- argmax may avoid noise if it harms controllability
- softmax may spend more time probing it
- behavior may reveal whether outer softness is enough to induce noisy-region attraction even with unchanged empowerment definition

### 6.4 Discretized cartpole

A small discrete-state approximation is enough initially.

#### Why it matters

The unstable upright region is often locally high in empowerment. This is a good test of whether softmax changes the “hover near the unstable summit” behavior into something broader.

#### Expected result

- argmax stays near high-empowerment unstable regions with narrow repertoire
- softmax may broaden trajectory occupation around those regions

---

## 7. Core abstractions

### 7.1 Environment interface

Each environment should expose:

```python
class DiscreteEnv:
    states: list[int]
    actions: dict[int, list[int]]

    def transition_prob(self, s: int, a: int) -> dict[int, float]:
        ...

    def is_absorbing(self, s: int) -> bool:
        ...
```

Optional helpers:

- `successors(s, a)`
- `render_state(s)`
- `state_features(s)`
- `state_label(s)`

### 7.2 Empowerment estimator interface

```python
class EmpowermentEstimator:
    def empowerment(self, s: int, horizon: int) -> float:
        ...
```

### 7.3 Outer policy interface

```python
class OuterPolicy:
    def action_probs(self, s: int) -> dict[int, float]:
        ...

    def sample_action(self, s: int, rng) -> int:
        ...
```

---

## 8. Inner empowerment implementation

### 8.1 Exact tabular empowerment

For each state `s` and horizon `n`:

1. enumerate all length-`n` action sequences available from `s`
2. compute the induced channel
   ```math
   p(s_n \mid s, a_{0:n-1})
   ```
3. run Blahut–Arimoto to compute channel capacity
4. return `E_n(s)`

This should live in:

- `src/empowerment/nstep_model.py`
- `src/empowerment/blahut_arimoto.py`
- `src/empowerment/exact.py`

### 8.2 Practical note on state-dependent action sets

The implementation should allow state-dependent action availability. This is especially important in gridworlds with walls or absorbing states.

### 8.3 Practical note on horizon choice

The horizon `n` should be configurable per environment. Good defaults:

- corridor: `n = 3 to 6`
- four rooms: `n = 5`
- cartpole discrete: `n = 2 or 3`

The repo should support horizon sweeps because some behavioral conclusions may depend strongly on `n`.

---

## 9. Outer policy implementation

### 9.1 Shared action score

All outer policies should consume the same score function:

```python
def empowerment_action_score(env, emp_table, s, a):
    probs = env.transition_prob(s, a)
    return sum(p * emp_table[s_next] for s_next, p in probs.items())
```

This is the most important invariance in the repo.

### 9.2 Greedy policy

```python
class GreedyEmpowermentPolicy(OuterPolicy):
    def action_probs(self, s):
        scores = ...
        max_score = max(scores.values())
        best = [a for a, v in scores.items() if v == max_score]
        return {a: 1/len(best) if a in best else 0.0 for a in scores}
```

### 9.3 Softmax policy

```python
class SoftmaxEmpowermentPolicy(OuterPolicy):
    def __init__(self, temperature: float):
        self.temperature = temperature

    def action_probs(self, s):
        scores = ...
        # subtract max for numerical stability
        m = max(scores.values())
        weights = {a: math.exp((v - m) / self.temperature) for a, v in scores.items()}
        z = sum(weights.values())
        return {a: w / z for a, w in weights.items()}
```

### 9.4 Important normalization option

Softmax temperature is not comparable across environments if score scales differ too much.

Include at least one optional normalization mode:

- no normalization
- subtract mean and divide by within-state std
- subtract max only

Recommended API:

```python
SoftmaxEmpowermentPolicy(temperature=0.5, normalize="zscore")
```

### 9.5 Epsilon-greedy control

This is helpful as a baseline to distinguish:

- “stochasticity helps”
- from “Boltzmann-shaped stochasticity helps”

---

## 10. Experiment matrix

The repo should explicitly define the experiment matrix.

### 10.1 Main factors

- environment
- empowerment horizon `n`
- outer policy type
- temperature `tau` for softmax
- epsilon for epsilon-greedy
- random seed
- rollout length

### 10.2 Minimum matrix

| Experiment | Env | Empowerment | Outer policy |
|---|---|---:|---|
| E1 | corridor | exact, fixed `n` | argmax vs softmax sweep |
| E2 | four rooms energy | exact, fixed `n` | argmax vs softmax sweep |
| E3 | noisy room | exact, fixed `n` | argmax vs softmax sweep |
| E4 | cartpole discrete | exact or approximate | argmax vs softmax sweep |
| E5 | any one env | exact, fixed `n` | argmax vs epsilon-greedy vs softmax |

### 10.3 Temperature sweep

A good default sweep:

```text
tau in {0.01, 0.03, 0.1, 0.3, 1.0, 3.0}
```

Interpretation:

- `tau -> 0` approximates greedy
- intermediate `tau` tests structured exploration
- large `tau` approximates near-uniform behavior

---

## 11. Metrics

The repo should not rely on just one scalar. It should separate **control quality**, **diversity**, and **survival/task** metrics.

### 11.1 Control-quality metrics

- mean empowerment of visited states
- max empowerment reached in episode
- average action score `Q_emp(s,a)` under chosen actions
- time-to-first-hit of high-empowerment region
- fraction of episodes reaching high-empowerment hub

### 11.2 Diversity metrics

- state visitation entropy
- action entropy of realized policy
- number of distinct states visited
- trajectory entropy over length-`T` windows
- edit-distance or Hamming diversity between trajectories
- occupancy heatmap spread
- spectral / autocorrelation score for loopiness

### 11.3 Survival / robustness metrics

- survival time
- fraction of time before absorption
- number of recoveries from low-energy states
- food hits / recharge frequency

### 11.4 Regime-specific metrics

#### Corridor env
- fraction reaching far basin
- mean dwell time in local basin vs far basin

#### Four rooms
- room occupancy entropy
- doorway crossing count
- cycle-period histogram

#### Cartpole
- angle-position occupation area
- time near unstable equilibrium
- phase-space coverage

---

## 12. Plots that should exist in the repo

These plots should be treated as first-class outputs.

### 12.1 Temperature phase diagram

For each environment, plot the following against softmax temperature:

- mean visited empowerment
- state visitation entropy
- distinct states visited
- survival time
- probability of reaching best basin

This is probably the single most important figure family.

### 12.2 Occupancy heatmaps

For gridworlds:

- state visitation heatmap for argmax
- state visitation heatmap for low / medium / high temperature softmax

### 12.3 Trajectory overlays

Overlay representative trajectories by policy type.

### 12.4 Basin transition plots

For corridor-like environments:

- probability of entering each basin over time
- cumulative fraction reaching far basin

### 12.5 Loopiness plot

Quantify repetitive cycling under argmax by:

- state autocorrelation
- recurrence plot
- dominant period estimate

---

## 13. Example experiment specifications

### 13.1 Experiment A: local maximum escape in corridor

#### Objective

Test whether softmax escapes local empowerment maxima better than argmax.

#### Setup

- deterministic gridworld
- one start state
- left action reaches local basin with moderate empowerment
- right action leads through narrow corridor to globally better basin

#### Independent variables

- policy type: argmax vs softmax
- temperature: sweep
- empowerment horizon: 3, 4, 5, 6

#### Dependent variables

- probability of reaching far basin
- mean dwell time in far basin
- state visitation entropy
- mean empowerment after burn-in

#### Expected qualitative result

Argmax gets trapped. Moderate-temperature softmax escapes more often.

### 13.2 Experiment B: four rooms with energy

#### Objective

Compare repetitive local empowerment control against broader exploration under energy constraints.

#### Setup

- four rooms
- food in corners
- energy decreases by 1 per step
- zero energy is absorbing
- exact empowerment with configurable horizon

#### Independent variables

- policy type
- temperature
- empowerment horizon
- food gain

#### Dependent variables

- survival time
- number of room transitions
- visitation entropy over locations
- fraction of time near food vs room centers
- cycle-period statistic

#### Expected qualitative result

Greedy empowerment exhibits stereotyped loops. Softmax broadens strategy repertoire while retaining some survival structure.

### 13.3 Experiment C: noisy room

#### Objective

Test whether outer softness alone induces increased occupancy of noisy regions, even with the same empowerment estimate.

#### Setup

- one region with stochastic transitions
- rest deterministic

#### Dependent variables

- fraction of time in noisy room
- survival time
- mean empowerment visited
- diversity metrics

#### Interpretation

This helps separate “empowerment proper” from “exploratory action selection.”

---

## 14. Suggested implementation order

### Milestone 1

- implement `DiscreteEnv`
- implement corridor environment
- implement exact `E_n(s)`
- implement greedy and softmax outer policies
- run 100-seed corridor experiment

### Milestone 2

- add visitation, trajectory, and basin metrics
- add temperature sweep plots
- add tests for softmax and empowerment calculations

### Milestone 3

- implement four-rooms-with-energy
- replicate greedy empowerment loopiness
- compare to softmax and epsilon-greedy

### Milestone 4

- implement noisy room variant
- produce phase diagrams over temperature and horizon

### Milestone 5

- add discretized cartpole
- compare qualitative trajectory repertoires

### Milestone 6

- optional scalable approximation methods

---

## 15. Reproducibility requirements

The repo should make reproduction easy.

### Required practices

- all experiments config-driven
- fixed random seeds logged
- raw trajectories saved to disk
- empowerment tables cached
- exact version of environment parameters logged
- figure scripts deterministic and idempotent

### Save per run

- seed
- environment config
- policy config
- empowerment horizon
- empowerment table checksum
- full metric dict
- optional raw trajectory states/actions

---

## 16. Tests

### 16.1 Mathematical tests

- Blahut–Arimoto converges on simple channels with known capacity
- empowerment of absorbing states is zero or minimal as expected
- deterministic one-action states have zero empowerment

### 16.2 Policy tests

- greedy returns tie-uniform distribution over maxima
- softmax probabilities sum to 1
- softmax approaches greedy as `tau -> 0`
- softmax approaches uniform as `tau -> infinity`

### 16.3 Environment tests

- transition kernels sum to 1
- absorbing states self-loop correctly
- corridor geometry matches specification

### 16.4 Metric tests

- visitation entropy decreases for concentrated occupancy
- distinct-states counter is exact
- survival metric matches trajectory termination

---

## 17. README outline

The root `README.md` should be short and implementation-focused.

Recommended sections:

1. What question the repo answers
2. Inner vs outer optimization distinction
3. Quickstart install
4. Minimal corridor reproduction command
5. Main figures produced by the repo
6. Repository structure
7. References

A good minimal quickstart could look like:

```bash
pip install -e .
python scripts/run_experiment.py --config configs/experiments/exact_tabular_baselines.yaml
python scripts/make_figures.py --input results/raw --output results/figures
```

---

## 18. Example pseudocode

### 18.1 Precompute empowerment table

```python
def compute_empowerment_table(env, horizon):
    table = {}
    for s in env.states:
        table[s] = exact_empowerment(env, s, horizon)
    return table
```

### 18.2 Build outer policy

```python
def build_policy(policy_name, env, emp_table, **kwargs):
    if policy_name == "greedy":
        return GreedyEmpowermentPolicy(env, emp_table)
    if policy_name == "softmax":
        return SoftmaxEmpowermentPolicy(env, emp_table, temperature=kwargs["temperature"])
    if policy_name == "epsilon_greedy":
        return EpsilonGreedyEmpowermentPolicy(env, emp_table, epsilon=kwargs["epsilon"])
    raise ValueError(policy_name)
```

### 18.3 Rollout loop

```python
def rollout(env, policy, horizon_T, seed):
    rng = np.random.default_rng(seed)
    s = env.reset(seed=seed)
    traj = []

    for t in range(horizon_T):
        probs = policy.action_probs(s)
        a = sample_from_dict(probs, rng)
        s_next = sample_transition(env.transition_prob(s, a), rng)
        traj.append((s, a, s_next, probs))
        s = s_next
        if env.is_absorbing(s):
            break

    return traj
```

---

## 19. Common pitfalls

### Pitfall 1. Changing both inner and outer policy at once

Do not compare:

- greedy one-step empowerment
- against soft Bellman RL with empowerment reward

as if only the action sampler changed. That changes the objective too.

### Pitfall 2. Ignoring score-scale dependence of softmax

Temperature only makes sense relative to score scale.

### Pitfall 3. Measuring only mean empowerment visited

This can miss the actual behavioral story, which may be about diversity and escape from local attractors.

### Pitfall 4. Using only learned approximations first

Approximation noise can look like stochastic exploration. Start exact.

### Pitfall 5. Failing to separate stochastic dynamics from stochastic policy

Log both. Otherwise one may incorrectly attribute environment noise to policy diversity.

---

## 20. Stretch goals

Once the basic repo is working, good extensions include:

- horizon-adaptive softmax policies
- state-dependent temperature schedules
- KL-regularized outer control around greedy empowerment
- learned world models for empowerment in larger domains
- variational empowerment estimators
- comparison against MOP-style path entropy objectives
- continuous-action empowerment approximations

---

## 21. Deliverables

The repo should aim to produce the following deliverables.

### Minimal publishable set

- one exact corridor experiment showing local-basin escape difference
- one four-room energy experiment showing loopiness vs diversity
- temperature sweep figure
- table of metrics across policies
- trajectories / heatmaps for visual behavior comparison

### Stronger set

- noisy-room experiment
- discretized cartpole experiment
- epsilon-greedy control baseline
- ablation over empowerment horizon

---

## 22. Suggested file contents by module

### `src/empowerment/blahut_arimoto.py`

Should contain:

- channel normalization checks
- BA iterations
- convergence criterion
- optional warm start

### `src/empowerment/nstep_model.py`

Should contain:

- enumeration of action sequences
- rollout of transition distributions
- compression into channel matrix form

### `src/policies/softmax.py`

Should contain:

- temperature handling
- normalization mode
- numerically stable exponentiation

### `src/metrics/trajectories.py`

Should contain:

- distinct state count
- trajectory n-gram frequencies
- loopiness / recurrence statistics

### `src/analysis/plots.py`

Should contain:

- heatmaps
- metric-vs-temperature plots
- representative trajectory panels

---

## 23. One canonical figure to target

If only one figure is built first, it should be this:

### Corridor phase diagram

**x-axis:** softmax temperature
**y-axis:**
- probability of reaching far basin
- state visitation entropy
- mean empowerment after burn-in

And include the argmax point on the same chart.

This will probably make the entire point of the repo legible in one glance.

---

## 24. Reference implementation conventions

Recommended defaults:

- Python 3.11+
- NumPy for tabular core
- SciPy only if needed
- Matplotlib for plots
- PyYAML for configs
- pandas for result aggregation
- no heavy framework dependence in Phase 1

For the exact tabular phase, avoid unnecessary ML dependencies.

---

## 25. Closing summary

The essential design principle of this repo is:

> **Fix empowerment. Vary only the outer policy. Measure behavior broadly.**

That gives a clean answer to the scientific question.

If the repo is built this way, it should be able to distinguish at least three different claims that are often blurred together:

1. what empowerment values states for
2. how an agent acts when chasing empowerment
3. which behavioral properties come from stochastic outer control rather than from empowerment itself

---

## 26. References

- Klyubin, Polani, Nehaniv. *Empowerment: A universal agent-centric measure of control.*
- Jung, Polani, Stone. *Empowerment for continuous agent-environment systems.*
- Mohamed, Rezende. *Variational information maximisation for intrinsically motivated reinforcement learning.*
- Ramírez-Ruiz et al. *Complex behavior from intrinsic motivation to occupy future action-state path space.*
- Blahut. *Computation of channel capacity and rate-distortion functions.*i(venv) [sv /home/scottviteri/Downloads]$ cat ~/Downloads/empowerment_repo_implementation_guide.md
# Empowerment Outer-Policy Experiments

A repository design and implementation guide for testing the behavioral differences between **argmax** and **softmax** outer policies that act to maximize **empowerment**.

---

## 1. Purpose

This repo is for one question:

> How much of “empowerment behavior” comes from the empowerment objective itself, and how much comes from the **outer control rule** used to act on empowerment estimates?

The core experimental move is to **hold the empowerment computation fixed** and vary **only** the outer policy used to select actions.

Concretely, the repo should compare at least:

- **Greedy / argmax outer control**
- **Softmax / Boltzmann outer control**
- optionally **epsilon-greedy** as a control

The main comparison should isolate the following distinction:

- **Inner optimization:** the definition of empowerment at a state
- **Outer optimization:** the behavior policy used to choose actions given empowerment-derived action scores

The paper `Complex behavior from intrinsic motivation to occupy future action-state path space` is a useful motivation because it compares MOP’s stochastic policy against an empowerment controller that is explicitly greedy over successor-state empowerment. This repo is meant to make that comparison more symmetric and controlled.

---

## 2. Scientific framing

### 2.1 Inner object: empowerment

For an environment state `s` and horizon `n`, define empowerment as

```math
E_n(s) = \max_{q(a_{0:n-1} \mid s)} I(A_{0:n-1}; S_n \mid s)
```

This is a property of the local action-to-future-state channel induced by the dynamics.

### 2.2 Outer object: action selection using empowerment

Given empowerment estimates for successor states, define an empowerment-derived action score

```math
Q_emp(s, a) = \sum_{s'} P(s' \mid s, a) E_n(s')
```

In a deterministic environment:

```math
Q_emp(s, a) = E_n(f(s, a))
```

Then compare different outer policies:

#### Greedy / argmax

```math
\pi_{arg}(a \mid s) = \frac{\mathbf{1}[a \in \arg\max_b Q_emp(s,b)]}{|\arg\max_b Q_emp(s,b)|}
```

#### Softmax / Boltzmann

```math
\pi_\tau(a \mid s) = \frac{\exp(Q_emp(s,a)/\tau)}{\sum_b \exp(Q_emp(s,b)/\tau)}
```

#### Optional epsilon-greedy control

```math
\pi_\epsilon(a \mid s) =
\begin{cases}
1-\epsilon + \epsilon/|A(s)| & a \in \arg\max_b Q_emp(s,b) \\
\epsilon/|A(s)| & \text{otherwise}
\end{cases}
```

This makes the central experimental variable explicit:

> **Same inner empowerment estimate, different outer control law.**

---

## 3. Core hypotheses

The repo should be organized around a few testable hypotheses.

### H1. Greedy empowerment is behaviorally narrow

Argmax outer control will often collapse into stereotyped loops, local maxima, or repeated visits to a small set of high-empowerment states.

### H2. Softmax outer control improves basin escape

At moderate temperatures, softmax will more often cross temporary low-empowerment bottlenecks and reach better long-run empowerment basins.

### H3. Softmax trades off peak empowerment for diversity

Softmax may sometimes lower the mean instantaneous empowerment visited while increasing:

- state visitation entropy
- trajectory diversity
- number of distinct states visited
- strategy diversity across episodes

### H4. Very high temperature hurts control quality

As temperature increases, the policy becomes overly diffuse and performance degrades on survival or long-run empowerment metrics.

### H5. Some previously reported “empowerment behavior” is really outer-policy behavior

Behavioral conclusions about empowerment may be substantially altered by the outer policy used to realize it.

---

## 4. Recommended repository layout

```text
empowerment-policy-study/
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── envs/
│   │   ├── corridor.yaml
│   │   ├── four_rooms_energy.yaml
│   │   ├── noisy_room.yaml
│   │   └── cartpole_discrete.yaml
│   ├── algs/
│   │   ├── greedy.yaml
│   │   ├── softmax.yaml
│   │   ├── epsilon_greedy.yaml
│   │   └── softmax_sweep.yaml
│   └── experiments/
│       ├── exact_tabular_baselines.yaml
│       ├── corridor_temperature_sweep.yaml
│       ├── four_rooms_reproduction.yaml
│       └── cartpole_behavior.yaml
├── src/
│   ├── __init__.py
│   ├── envs/
│   │   ├── base.py
│   │   ├── corridor.py
│   │   ├── four_rooms_energy.py
│   │   ├── noisy_room.py
│   │   └── cartpole_discrete.py
│   ├── empowerment/
│   │   ├── channel.py
│   │   ├── exact.py
│   │   ├── blahut_arimoto.py
│   │   ├── nstep_model.py
│   │   └── score.py
│   ├── policies/
│   │   ├── base.py
│   │   ├── greedy.py
│   │   ├── softmax.py
│   │   └── epsilon_greedy.py
│   ├── rollout/
│   │   ├── simulate.py
│   │   ├── logging.py
│   │   └── seeds.py
│   ├── metrics/
│   │   ├── visitation.py
│   │   ├── trajectories.py
│   │   ├── empowerment_metrics.py
│   │   └── survival.py
│   ├── analysis/
│   │   ├── aggregate.py
│   │   ├── plots.py
│   │   └── statistical_tests.py
│   └── utils/
│       ├── math.py
│       ├── config.py
│       └── serialization.py
├── scripts/
│   ├── run_experiment.py
│   ├── compute_empowerment_table.py
│   ├── rollout_policy.py
│   ├── sweep_temperature.py
│   └── make_figures.py
├── notebooks/
│   ├── 01_corridor_debug.ipynb
│   ├── 02_four_rooms_reproduction.ipynb
│   ├── 03_temperature_phase_diagram.ipynb
│   └── 04_cartpole_behavior.ipynb
├── tests/
│   ├── test_blahut_arimoto.py
│   ├── test_empowerment_exact.py
│   ├── test_softmax_policy.py
│   ├── test_metrics.py
│   └── test_env_dynamics.py
└── results/
    ├── raw/
    ├── processed/
    └── figures/
```

---

## 5. Minimal implementation plan

### Phase 1. Exact tabular experiments

Implement the simplest possible exact version first.

Requirements:

- finite state spaces
- finite action spaces
- known transition matrices
- exact n-step empowerment computation
- no function approximation

This phase should answer the scientific question cleanly before any neural approximations are introduced.

### Phase 2. Approximate / scalable experiments

Only after the tabular experiments are working:

- add variational or sampled empowerment estimation
- add continuous or discretized continuous environments
- optionally add learning of dynamics or amortized empowerment estimators

The repo should make Phase 1 the default path for reproduction.

---

## 6. Environment suite

The environments should be chosen to make the difference between argmax and softmax visible.

### 6.1 Corridor-to-better-basin

This should be the first environment.

#### Design goal

Create a small deterministic gridworld with:

- one nearby region with **moderately high empowerment**
- one farther region with **higher empowerment**
- a narrow corridor of temporarily lower-empowerment states between them

#### Why it matters

This is the cleanest test of whether softmax helps escape local empowerment maxima.

#### Expected result

- argmax gets stuck in the nearby basin
- moderate-temperature softmax reaches the farther high-empowerment hub more often

### 6.2 Four-rooms with energy / food

This should reproduce the qualitative setup used in the MOP paper, but with a symmetric empowerment comparison.

#### Design goal

A gridworld with:

- finite energy
- food locations that replenish energy
- absorbing death states at zero energy
- doorways and rooms creating spatial bottlenecks

#### Why it matters

This environment introduces survival pressure, and it is close to the setup in which greedy empowerment was reported to alternate between food and local room-center states.

#### Expected result

- argmax exhibits repetitive loops
- softmax exhibits broader exploration and more path diversity
- both may retain survival-seeking behavior under energy constraints

### 6.3 Noisy room

Add one room where transitions are stochastic.

#### Why it matters

This tests how soft outer control interacts with states that are noisy versus controllable.

#### Expected result

Depending on empowerment horizon and score construction:

- argmax may avoid noise if it harms controllability
- softmax may spend more time probing it
- behavior may reveal whether outer softness is enough to induce noisy-region attraction even with unchanged empowerment definition

### 6.4 Discretized cartpole

A small discrete-state approximation is enough initially.

#### Why it matters

The unstable upright region is often locally high in empowerment. This is a good test of whether softmax changes the “hover near the unstable summit” behavior into something broader.

#### Expected result

- argmax stays near high-empowerment unstable regions with narrow repertoire
- softmax may broaden trajectory occupation around those regions

---

## 7. Core abstractions

### 7.1 Environment interface

Each environment should expose:

```python
class DiscreteEnv:
    states: list[int]
    actions: dict[int, list[int]]

    def transition_prob(self, s: int, a: int) -> dict[int, float]:
        ...

    def is_absorbing(self, s: int) -> bool:
        ...
```

Optional helpers:

- `successors(s, a)`
- `render_state(s)`
- `state_features(s)`
- `state_label(s)`

### 7.2 Empowerment estimator interface

```python
class EmpowermentEstimator:
    def empowerment(self, s: int, horizon: int) -> float:
        ...
```

### 7.3 Outer policy interface

```python
class OuterPolicy:
    def action_probs(self, s: int) -> dict[int, float]:
        ...

    def sample_action(self, s: int, rng) -> int:
        ...
```

---

## 8. Inner empowerment implementation

### 8.1 Exact tabular empowerment

For each state `s` and horizon `n`:

1. enumerate all length-`n` action sequences available from `s`
2. compute the induced channel
   ```math
   p(s_n \mid s, a_{0:n-1})
   ```
3. run Blahut–Arimoto to compute channel capacity
4. return `E_n(s)`

This should live in:

- `src/empowerment/nstep_model.py`
- `src/empowerment/blahut_arimoto.py`
- `src/empowerment/exact.py`

### 8.2 Practical note on state-dependent action sets

The implementation should allow state-dependent action availability. This is especially important in gridworlds with walls or absorbing states.

### 8.3 Practical note on horizon choice

The horizon `n` should be configurable per environment. Good defaults:

- corridor: `n = 3 to 6`
- four rooms: `n = 5`
- cartpole discrete: `n = 2 or 3`

The repo should support horizon sweeps because some behavioral conclusions may depend strongly on `n`.

---

## 9. Outer policy implementation

### 9.1 Shared action score

All outer policies should consume the same score function:

```python
def empowerment_action_score(env, emp_table, s, a):
    probs = env.transition_prob(s, a)
    return sum(p * emp_table[s_next] for s_next, p in probs.items())
```

This is the most important invariance in the repo.

### 9.2 Greedy policy

```python
class GreedyEmpowermentPolicy(OuterPolicy):
    def action_probs(self, s):
        scores = ...
        max_score = max(scores.values())
        best = [a for a, v in scores.items() if v == max_score]
        return {a: 1/len(best) if a in best else 0.0 for a in scores}
```

### 9.3 Softmax policy

```python
class SoftmaxEmpowermentPolicy(OuterPolicy):
    def __init__(self, temperature: float):
        self.temperature = temperature

    def action_probs(self, s):
        scores = ...
        # subtract max for numerical stability
        m = max(scores.values())
        weights = {a: math.exp((v - m) / self.temperature) for a, v in scores.items()}
        z = sum(weights.values())
        return {a: w / z for a, w in weights.items()}
```

### 9.4 Important normalization option

Softmax temperature is not comparable across environments if score scales differ too much.

Include at least one optional normalization mode:

- no normalization
- subtract mean and divide by within-state std
- subtract max only

Recommended API:

```python
SoftmaxEmpowermentPolicy(temperature=0.5, normalize="zscore")
```

### 9.5 Epsilon-greedy control

This is helpful as a baseline to distinguish:

- “stochasticity helps”
- from “Boltzmann-shaped stochasticity helps”

---

## 10. Experiment matrix

The repo should explicitly define the experiment matrix.

### 10.1 Main factors

- environment
- empowerment horizon `n`
- outer policy type
- temperature `tau` for softmax
- epsilon for epsilon-greedy
- random seed
- rollout length

### 10.2 Minimum matrix

| Experiment | Env | Empowerment | Outer policy |
|---|---|---:|---|
| E1 | corridor | exact, fixed `n` | argmax vs softmax sweep |
| E2 | four rooms energy | exact, fixed `n` | argmax vs softmax sweep |
| E3 | noisy room | exact, fixed `n` | argmax vs softmax sweep |
| E4 | cartpole discrete | exact or approximate | argmax vs softmax sweep |
| E5 | any one env | exact, fixed `n` | argmax vs epsilon-greedy vs softmax |

### 10.3 Temperature sweep

A good default sweep:

```text
tau in {0.01, 0.03, 0.1, 0.3, 1.0, 3.0}
```

Interpretation:

- `tau -> 0` approximates greedy
- intermediate `tau` tests structured exploration
- large `tau` approximates near-uniform behavior

---

## 11. Metrics

The repo should not rely on just one scalar. It should separate **control quality**, **diversity**, and **survival/task** metrics.

### 11.1 Control-quality metrics

- mean empowerment of visited states
- max empowerment reached in episode
- average action score `Q_emp(s,a)` under chosen actions
- time-to-first-hit of high-empowerment region
- fraction of episodes reaching high-empowerment hub

### 11.2 Diversity metrics

- state visitation entropy
- action entropy of realized policy
- number of distinct states visited
- trajectory entropy over length-`T` windows
- edit-distance or Hamming diversity between trajectories
- occupancy heatmap spread
- spectral / autocorrelation score for loopiness

### 11.3 Survival / robustness metrics

- survival time
- fraction of time before absorption
- number of recoveries from low-energy states
- food hits / recharge frequency

### 11.4 Regime-specific metrics

#### Corridor env
- fraction reaching far basin
- mean dwell time in local basin vs far basin

#### Four rooms
- room occupancy entropy
- doorway crossing count
- cycle-period histogram

#### Cartpole
- angle-position occupation area
- time near unstable equilibrium
- phase-space coverage

---

## 12. Plots that should exist in the repo

These plots should be treated as first-class outputs.

### 12.1 Temperature phase diagram

For each environment, plot the following against softmax temperature:

- mean visited empowerment
- state visitation entropy
- distinct states visited
- survival time
- probability of reaching best basin

This is probably the single most important figure family.

### 12.2 Occupancy heatmaps

For gridworlds:

- state visitation heatmap for argmax
- state visitation heatmap for low / medium / high temperature softmax

### 12.3 Trajectory overlays

Overlay representative trajectories by policy type.

### 12.4 Basin transition plots

For corridor-like environments:

- probability of entering each basin over time
- cumulative fraction reaching far basin

### 12.5 Loopiness plot

Quantify repetitive cycling under argmax by:

- state autocorrelation
- recurrence plot
- dominant period estimate

---

## 13. Example experiment specifications

### 13.1 Experiment A: local maximum escape in corridor

#### Objective

Test whether softmax escapes local empowerment maxima better than argmax.

#### Setup

- deterministic gridworld
- one start state
- left action reaches local basin with moderate empowerment
- right action leads through narrow corridor to globally better basin

#### Independent variables

- policy type: argmax vs softmax
- temperature: sweep
- empowerment horizon: 3, 4, 5, 6

#### Dependent variables

- probability of reaching far basin
- mean dwell time in far basin
- state visitation entropy
- mean empowerment after burn-in

#### Expected qualitative result

Argmax gets trapped. Moderate-temperature softmax escapes more often.

### 13.2 Experiment B: four rooms with energy

#### Objective

Compare repetitive local empowerment control against broader exploration under energy constraints.

#### Setup

- four rooms
- food in corners
- energy decreases by 1 per step
- zero energy is absorbing
- exact empowerment with configurable horizon

#### Independent variables

- policy type
- temperature
- empowerment horizon
- food gain

#### Dependent variables

- survival time
- number of room transitions
- visitation entropy over locations
- fraction of time near food vs room centers
- cycle-period statistic

#### Expected qualitative result

Greedy empowerment exhibits stereotyped loops. Softmax broadens strategy repertoire while retaining some survival structure.

### 13.3 Experiment C: noisy room

#### Objective

Test whether outer softness alone induces increased occupancy of noisy regions, even with the same empowerment estimate.

#### Setup

- one region with stochastic transitions
- rest deterministic

#### Dependent variables

- fraction of time in noisy room
- survival time
- mean empowerment visited
- diversity metrics

#### Interpretation

This helps separate “empowerment proper” from “exploratory action selection.”

---

## 14. Suggested implementation order

### Milestone 1

- implement `DiscreteEnv`
- implement corridor environment
- implement exact `E_n(s)`
- implement greedy and softmax outer policies
- run 100-seed corridor experiment

### Milestone 2

- add visitation, trajectory, and basin metrics
- add temperature sweep plots
- add tests for softmax and empowerment calculations

### Milestone 3

- implement four-rooms-with-energy
- replicate greedy empowerment loopiness
- compare to softmax and epsilon-greedy

### Milestone 4

- implement noisy room variant
- produce phase diagrams over temperature and horizon

### Milestone 5

- add discretized cartpole
- compare qualitative trajectory repertoires

### Milestone 6

- optional scalable approximation methods

---

## 15. Reproducibility requirements

The repo should make reproduction easy.

### Required practices

- all experiments config-driven
- fixed random seeds logged
- raw trajectories saved to disk
- empowerment tables cached
- exact version of environment parameters logged
- figure scripts deterministic and idempotent

### Save per run

- seed
- environment config
- policy config
- empowerment horizon
- empowerment table checksum
- full metric dict
- optional raw trajectory states/actions

---

## 16. Tests

### 16.1 Mathematical tests

- Blahut–Arimoto converges on simple channels with known capacity
- empowerment of absorbing states is zero or minimal as expected
- deterministic one-action states have zero empowerment

### 16.2 Policy tests

- greedy returns tie-uniform distribution over maxima
- softmax probabilities sum to 1
- softmax approaches greedy as `tau -> 0`
- softmax approaches uniform as `tau -> infinity`

### 16.3 Environment tests

- transition kernels sum to 1
- absorbing states self-loop correctly
- corridor geometry matches specification

### 16.4 Metric tests

- visitation entropy decreases for concentrated occupancy
- distinct-states counter is exact
- survival metric matches trajectory termination

---

## 17. README outline

The root `README.md` should be short and implementation-focused.

Recommended sections:

1. What question the repo answers
2. Inner vs outer optimization distinction
3. Quickstart install
4. Minimal corridor reproduction command
5. Main figures produced by the repo
6. Repository structure
7. References

A good minimal quickstart could look like:

```bash
pip install -e .
python scripts/run_experiment.py --config configs/experiments/exact_tabular_baselines.yaml
python scripts/make_figures.py --input results/raw --output results/figures
```

---

## 18. Example pseudocode

### 18.1 Precompute empowerment table

```python
def compute_empowerment_table(env, horizon):
    table = {}
    for s in env.states:
        table[s] = exact_empowerment(env, s, horizon)
    return table
```

### 18.2 Build outer policy

```python
def build_policy(policy_name, env, emp_table, **kwargs):
    if policy_name == "greedy":
        return GreedyEmpowermentPolicy(env, emp_table)
    if policy_name == "softmax":
        return SoftmaxEmpowermentPolicy(env, emp_table, temperature=kwargs["temperature"])
    if policy_name == "epsilon_greedy":
        return EpsilonGreedyEmpowermentPolicy(env, emp_table, epsilon=kwargs["epsilon"])
    raise ValueError(policy_name)
```

### 18.3 Rollout loop

```python
def rollout(env, policy, horizon_T, seed):
    rng = np.random.default_rng(seed)
    s = env.reset(seed=seed)
    traj = []

    for t in range(horizon_T):
        probs = policy.action_probs(s)
        a = sample_from_dict(probs, rng)
        s_next = sample_transition(env.transition_prob(s, a), rng)
        traj.append((s, a, s_next, probs))
        s = s_next
        if env.is_absorbing(s):
            break

    return traj
```

---

## 19. Common pitfalls

### Pitfall 1. Changing both inner and outer policy at once

Do not compare:

- greedy one-step empowerment
- against soft Bellman RL with empowerment reward

as if only the action sampler changed. That changes the objective too.

### Pitfall 2. Ignoring score-scale dependence of softmax

Temperature only makes sense relative to score scale.

### Pitfall 3. Measuring only mean empowerment visited

This can miss the actual behavioral story, which may be about diversity and escape from local attractors.

### Pitfall 4. Using only learned approximations first

Approximation noise can look like stochastic exploration. Start exact.

### Pitfall 5. Failing to separate stochastic dynamics from stochastic policy

Log both. Otherwise one may incorrectly attribute environment noise to policy diversity.

---

## 20. Stretch goals

Once the basic repo is working, good extensions include:

- horizon-adaptive softmax policies
- state-dependent temperature schedules
- KL-regularized outer control around greedy empowerment
- learned world models for empowerment in larger domains
- variational empowerment estimators
- comparison against MOP-style path entropy objectives
- continuous-action empowerment approximations

---

## 21. Deliverables

The repo should aim to produce the following deliverables.

### Minimal publishable set

- one exact corridor experiment showing local-basin escape difference
- one four-room energy experiment showing loopiness vs diversity
- temperature sweep figure
- table of metrics across policies
- trajectories / heatmaps for visual behavior comparison

### Stronger set

- noisy-room experiment
- discretized cartpole experiment
- epsilon-greedy control baseline
- ablation over empowerment horizon

---

## 22. Suggested file contents by module

### `src/empowerment/blahut_arimoto.py`

Should contain:

- channel normalization checks
- BA iterations
- convergence criterion
- optional warm start

### `src/empowerment/nstep_model.py`

Should contain:

- enumeration of action sequences
- rollout of transition distributions
- compression into channel matrix form

### `src/policies/softmax.py`

Should contain:

- temperature handling
- normalization mode
- numerically stable exponentiation

### `src/metrics/trajectories.py`

Should contain:

- distinct state count
- trajectory n-gram frequencies
- loopiness / recurrence statistics

### `src/analysis/plots.py`

Should contain:

- heatmaps
- metric-vs-temperature plots
- representative trajectory panels

---

## 23. One canonical figure to target

If only one figure is built first, it should be this:

### Corridor phase diagram

**x-axis:** softmax temperature
**y-axis:**
- probability of reaching far basin
- state visitation entropy
- mean empowerment after burn-in

And include the argmax point on the same chart.

This will probably make the entire point of the repo legible in one glance.

---

## 24. Reference implementation conventions

Recommended defaults:

- Python 3.11+
- NumPy for tabular core
- SciPy only if needed
- Matplotlib for plots
- PyYAML for configs
- pandas for result aggregation
- no heavy framework dependence in Phase 1

For the exact tabular phase, avoid unnecessary ML dependencies.

---

## 25. Closing summary

The essential design principle of this repo is:

> **Fix empowerment. Vary only the outer policy. Measure behavior broadly.**

That gives a clean answer to the scientific question.

If the repo is built this way, it should be able to distinguish at least three different claims that are often blurred together:

1. what empowerment values states for
2. how an agent acts when chasing empowerment
3. which behavioral properties come from stochastic outer control rather than from empowerment itself

---

## 26. References

- Klyubin, Polani, Nehaniv. *Empowerment: A universal agent-centric measure of control.*
- Jung, Polani, Stone. *Empowerment for continuous agent-environment systems.*
- Mohamed, Rezende. *Variational information maximisation for intrinsically motivated reinforcement learning.*
- Ramírez-Ruiz et al. *Complex behavior from intrinsic motivation to occupy future action-state path space.*
- Blahut. *Computation of channel capacity and rate-distortion functions.*

