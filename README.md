# ZB202 Prediction

Short-term **CO₂ forecasting and ventilation scenario simulation** for the ZB202 room. The project uses four hours of observed history to forecast CO₂ concentrations at five sensor locations over the next hour, and compares ventilation strategies under explicit assumptions.

The default forecasting model is a **joint CO₂ TCN**, averaging predictions from three fixed seeds. It has been evaluated on real data and tested for robustness using external datasets. Limitations remain around sensor offsets, spikes, and outages. The response to actual damper changes has not been calibrated, and the project is not connected to device controls.

## Directory

```text
.
├── docs/                   Usage, architecture, assumptions, and evaluation protocols
├── reports/                Dated progress reports
├── visualizations/         Figures, plotting scripts, and metrics
├── src/zb202/              Forecasting, physical models, and simulation logic
├── scripts/                Command-line entry points
├── configs/                Model parameters and simulation scenarios
├── data/                   Raw, processed, and synthetic data
├── influxdb/               Data export and verification tools
├── requirements/           Forecasting environment dependencies
└── tests/                  Data, forecasting, and physical model checks
```

## Documentation

| Category | Links |
| --- | --- |
| Progress reports | [2026-10-20 report](<reports/2026-10-20 report.md>) |
| Visualizations | [CO₂ figures and reading guide](visualizations/co2_20261020/README.md) |
| Usage and reproduction | [Installation, data, forecasting, simulation, and checks](docs/usage.md) |
| Evaluation | [Validation protocol](docs/forecasting-validation.md) · [Ablation protocol](docs/forecasting-ablation.md) · [Robustness validation](docs/forecasting-robustness.md) |
| Modeling | [Architecture](docs/architecture.md) · [Parameters and assumptions](docs/assumptions.md) · [Method proposal](docs/method-proposal.md) |
| Data | [Data overview](data/README.md) · [Processed data](data/processed/zb202_iot/README.md) · [Synthetic data](data/synthetic/README.md) |
| Project management | [Project plan](PROJECT_PLAN.md) · [Handoff](HANDOFF.md) |

Generated model weights, processed data, and simulation run directories are excluded from Git by default. See the usage documentation for required local artifacts and reproduction conditions. Linked documentation is primarily in Chinese.
