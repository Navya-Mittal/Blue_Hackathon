# Data

`credit_default.csv` is a 24,000-row sample of the public **Default of Credit Card Clients** dataset
(credit-card holders in Taiwan, April to September 2005). It contains no personal identifiers.

> Yeh, I-C. and Lien, C-H. (2009). *The comparisons of data mining techniques for the predictive accuracy of
> probability of default of credit card clients.* Expert Systems with Applications, 36(2), 2473-2480.
> Hosted by the UCI Machine Learning Repository.

This copy comes from the pycaret example-data distribution. **Check the licence and terms on the UCI dataset
page before redistributing the file**; if that is a concern, delete it and place a copy of the UCI data at the
same path with the same column names (the first row of the CSV shows them) and the target column named `default`.
If you replace the data, regenerate the splits and the R reference outputs: `make fixtures`.

`splits/` holds the train / test / edge-case files the pipeline derived from it (seed 42, 70/30 stratified
split). `splits/MANIFEST.json` records their SHA-256 hashes; the R reference outputs in `../fixtures/` are only
accepted if those hashes still match, so parity is always measured on the same rows.

The edge-case file (`splits/edge.csv`, 15 rows with ids from 900001) is synthetic and is built from the training
data by `blue/data.py::edge_cases`: extreme values, undocumented categories, negative bills and missing values.

The sample has no time dimension, so there is no out-of-time test set. The ongoing-monitoring batches are a
synthetic stress scenario, not real data drift.
