# Predicting Neuronal Excitability from Membrane Properties

A linear regression model that predicts **rheobase** — the minimum current needed to
make a neuron fire an action potential — from passive membrane properties, using
2,333 patch-clamped neurons from the [Allen Cell Types Database](https://celltypes.brain-map.org/).

## The question

Ohm's law (V = IR) says a cell with high input resistance should need less current
to reach firing threshold, because the same current produces a larger voltage
change. Does that hold across thousands of real neurons?

## Result

Yes, and with a scaling exponent close to the theoretical prediction:

```
rheobase ≈ 10^4.09 × resistance^(-0.91)
```

Ohm's law predicts an exponent of **-1**. The measured value is **-0.91**. The gap
and the residual scatter come from what input resistance does not capture:
differences in threshold voltage, capacitance and active channel composition
between cells.

| Model | Features | RMSE (log10 units) |
|---|---|---|
| Baseline | input resistance | 0.256 |
| Two-feature | input resistance + membrane time constant | **0.195** (24% better) |

## What the analysis does

1. **Load** the Allen electrophysiology feature table via `allensdk` (2,333 cells, 56 measurements each).
2. **Explore** the distributions, and confirm the rheobase labels sit on 10 pA measurement steps.
3. **Correlate** each feature against rheobase, then log-transform. Correlation with
   resistance strengthens from **-0.48 to -0.64**, doubling explained variance from
   23% to 41%, because the underlying relationship is `1/x` rather than linear.
4. **Implement RMSprop from scratch in NumPy** — recording the weight, gradient,
   optimizer memory and step size at every one of 235 updates — to see how the
   optimizer behaves rather than only its final answer.
5. **Train the Keras model** and validate it against the closed-form least-squares
   solution (`np.polyfit`). The two agree to 3 decimal places.
6. **Benchmark hyperparameters** against the theoretical minimum RMSE.

## Hyperparameter findings

All runs used one feature, 100 epochs, with a theoretical floor of RMSE 0.256:

| Setting | RMSE | Outcome |
|---|---|---|
| lr=0.01, batch=50 | 0.258 | Converged |
| lr=0.01, batch=5 | 0.259 | Converged by epoch ~3, 10x slower per epoch |
| lr=0.01, batch=500 | 0.287 | Undertrained: only 5 updates per epoch |
| lr=0.0001 | 0.463 | Undertrained: the slope never left its starting sign |
| lr=1.0 | 1.653 | Diverged: steps overshoot the loss valley every time |

Feature scaling matters: resting potential (around -71 mV) alongside log resistance
(around 2.3) trains poorly until it is standardised to a z-score, because the
mismatched scales produce mismatched gradients.

## Running it

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

Open `rheobase.ipynb` and select the `.venv` kernel. The first run downloads the
Allen dataset into `cell_types/`, which takes about a minute.

## Status

Under active development.

## Stack

Python, TensorFlow/Keras, pandas, NumPy, Plotly, AllenSDK, Jupyter
