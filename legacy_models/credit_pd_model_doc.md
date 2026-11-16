# Model Documentation: Credit Card PD Model (v1.2, last updated March 2022)

> SYNTHETIC SAMPLE DOCUMENT. Written for the Blue demo with deliberately planted discrepancies
> to mimic documentation that has drifted away from the code. It is not a real bank document.

## Model summary
- Algorithm: Gradient boosted trees (200 trees, depth 3)
- Target: default
- Number of predictors: 12
- Regularisation: L2 penalty
- Decision cut-off: 0.50
- Protected attributes used as predictors: SEX, AGE

## Purpose
Estimate the probability that a credit card account defaults within the next billing cycle.

## Governance
Annual review by the Model Risk Committee. Next review due March 2023.
