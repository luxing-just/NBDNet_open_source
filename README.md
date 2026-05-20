# NBD-Net

NBD-Net is a structural model for lithium-ion battery capacity and RUL prediction. This folder is kept as an architecture-oriented release: it shows the model composition and the main paper utilities, but it does not include datasets, checkpoints, or a full runnable experiment pipeline.

## Repository Structure

```text
NBDNet_open_source/
|-- RUL_NBDNet.py                         model structure
|-- requirements.txt                      pinned environment versions
|-- examples/
|   `-- plot_batteryC1_error_style.py     figure style reference
|-- paper_tools/                          paper utility scripts
|-- scripts/                              convenience commands
|-- data/
|   `-- README.md                         dataset layout note
|-- LICENSE
`-- README.md
```

## Model Structure

NBD-Net takes a health-indicator sequence `X` with shape `[B, L, C]` and predicts a future capacity/RUL-related value with shape `[B, pred_len]`.

```text
Input sequence
    |
Linear embedding
    |
RevIN
    |
    |------------------------------|
    v                              v
LocalStabilizer              GlobalTrendEncoder
    |                              |
local sequence h_local       global sequence h_global
    |                              |
    |                              v
    |                          SDMC memory slots
    |                              |
    |------------------------------|
                   |
                  LGBI
                   |
            Forecast head
                   |
               Prediction
```

The core modules are:

- `RevIN`: reversible instance normalization for sequence-level distribution shift.
- `LocalStabilizer`: local causal mean filtering plus causal depthwise convolution blocks, designed to stabilize short-range nonstationary fluctuations.
- `GlobalTrendEncoder`: GRU-based global path that preserves long-range degradation trend information.
- `SDMC`: Structured Degradation Memory Compression, using learnable trend, transition, and fluctuation queries to compress global features into semantic memory slots.
- `LGBI`: Local-Global Bridging Interaction, a bidirectional cross-attention block between local features and global memory slots.
- `Forecast head`: flatten-and-project prediction head for the final `pred_len` output.

## Core File

[RUL_NBDNet.py](RUL_NBDNet.py) contains only the structural model definitions:

- `RevIN`
- `CausalDWConv1d`
- `CausalConvBlock`
- `LocalStabilizer`
- `GlobalTrendEncoder`
- `SDMC`
- `LGBI`
- `NBDNet`

It intentionally omits data loading, training loops, checkpoints, and experiment-running code.

## Environment

The environment versions used during development are pinned in `requirements.txt`. They are provided for reference and compatibility, not as a promise that this folder is a one-command reproduction package.

## Data

Datasets are not included in this architecture release. The expected layout is documented in [data/README.md](data/README.md).

## License

This project is released under the MIT License. See `LICENSE`.
