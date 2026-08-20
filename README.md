# Turbofan RUL — remaining useful life to a maintenance decision

Predict **when a turbofan engine will need maintenance** from multivariate
sensor telemetry, and turn that prediction into an actionable, cost-aware
decision. Built on NASA **C-MAPSS** and **N-CMAPSS**.

The deliverable is a decision with confidence — *"engine 12 needs maintenance
in ~27 cycles"* — not an RMSE. Evaluation is deliberately **cost-asymmetric**,
scored on the NASA prognostics metric, which punishes a late warning far harder
than an early one.

## Results

| | RMSE on N-CMAPSS |
|---|---|
| Tuned GRU baseline | 10.2 |
| Cross-dataset self-supervised pretrained encoder | **7.1** |

**Generalisation was tested by leaving a whole fleet out**, not by shuffling
rows. Across 9 fleets the model held on 6 and failed on 2 with unseen fault
modes. A Mahalanobis distance gate flagged exactly those 2 (r = 0.89 against
held-out error), and fine-tuning on a handful of target engines recovered them,
cutting RMSE from 32 to 13.

That gate is the part worth reading: a model that knows which fleets it should
not be trusted on is more deployable than one that is slightly more accurate on
average.

## Testing

```bash
pip install -r requirements.txt
pytest -q
```

Covers dataset loading for both C-MAPSS variants, sequence windowing, feature
construction, the SSL pretraining path, drift detection, uncertainty and the
evaluation metrics.

## Layout

| Path | What is in it |
|---|---|
| `src/pdm/` | data loading, features, models, evaluation, drift, explainability |
| `tests/` | unit tests over each stage |
| `configs/` | experiment configuration |
| `scripts/` | environment checks and benchmarking |
| `docs/` | dataset survey and design notes |
| `ROADMAP.md` | plan of record |

Datasets and trained weights are not committed — see `docs/` for how to fetch
C-MAPSS and N-CMAPSS.
