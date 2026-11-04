# E-commerce Transaction Clustering

### A Comparative Study of PCA and Autoencoder-Based Feature Representation for K-Means Clustering

An academic group project exploring whether more complex feature representations produce more useful clusters of e-commerce purchases.

## Research question

Does applying PCA or autoencoders before K-Means improve cluster quality and interpretability compared with a prepared-feature baseline?

The unit of analysis is a **transaction**, rather than an individual customer. Potential applications include exploring purchase patterns and informing hypotheses for marketing or retail operations; operational impact was not measured.

## Data and workflow

The supplied source CSV contains **536,350 transaction-line records**. The saved notebook modelling outputs contain **19,788 aggregated transactions and 13 input features** after preparation. These counts refer to different processing stages.

The workflow covers:

1. Inspecting missing values, data types, distributions, and unusual transactions.
2. Filtering non-positive prices and quantities and aggregating purchase lines by transaction identifier.
3. Engineering six numerical purchase features and seven weekday indicators.
4. Standardizing numerical features.
5. Comparing K-Means, PCA-based representations, and shallow and deep autoencoder experiments.
6. Examining silhouette scores, runtime, cluster sizes, and profiles in the original business features.

## Main finding

**Higher silhouette scores did not necessarily produce more useful business segments.**

The submitted report lists the following results. They are historical reported results, not independently reproduced benchmarks.

| Experiment | Clusters | Reported silhouette score |
|---|---:|---:|
| K-Means baseline | 3 | 0.914 |
| PCA + K-Means baseline | 3 | 0.865 |
| Tuned PCA + K-Means | 3 | 0.921 |
| Autoencoder + K-Means baseline | 3 | 0.352 |
| Tuned shallow autoencoder + K-Means | 14 | 0.658 |
| Tuned deep autoencoder + K-Means | 3 | 0.897 |

These scores are measured in different feature spaces, so they are not directly interchangeable measures of business value. Cluster interpretability and stability need separate assessment.

The stored baseline cluster sizes are 19,739, 42, and 7: a high score can reflect separation of a few unusual observations from the majority, rather than a balanced, actionable segmentation. The report concludes that additional representation complexity offered limited practical benefit in this experiment.

## Technical skills demonstrated

- Python, pandas, and NumPy for cleaning and transaction-level feature engineering.
- scikit-learn for scaling, K-Means, PCA, and clustering evaluation.
- TensorFlow/Keras for neural representation experiments.
- KerasTuner for reconstruction-based hyperparameter searches.
- Matplotlib and Seaborn for exploratory analysis and experiment visualization.

## Known limitations and follow-up work

The original coursework is retained as a historical experiment. The following points require attention before treating the comparison as a validated benchmark:

- The baseline autoencoder branch passes reconstructed outputs to K-Means through the full autoencoder's `predict` method. It should use the encoder's bottleneck output for a latent-space comparison.
- That baseline uses a sigmoid output while numerical inputs are standardized and may fall outside 0–1. Output activation and input scaling need to be aligned.
- Reported scores and saved outputs should be reconciled in one fresh, documented run.
- Cluster stability across seeds and preprocessing choices, sensitivity to outliers, and segment usefulness should supplement silhouette-based selection.
- Reconstruction validation does not independently establish downstream clustering generalization.

These are proposed improvements, not completed fixes or new results.

## Running the analysis

Open the project notebook and inspect its input path before running. Supply the original sales transaction CSV and install the imported dependencies in a compatible Python environment. The complete notebook has not been rerun as part of this documentation update; exact package versions and a tested setup remain to be recorded.

Before redistributing the source data, verify the original dataset's license and provide its source URL. No data redistribution license is granted by this README.

## Authors and context

**Nilas Myler, Miaomiao Yuan, and Ya-Wen (Olivia) Liang**  
Machine Learning & Deep Learning, Copenhagen Business School, May 2025.

This is a group project; individual task ownership is not assigned in this overview.
