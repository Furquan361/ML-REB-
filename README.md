# MLRweb — ML Workbench — Regression Platform

Developed by **Dr. Furqan Ahmad**.

## Models

The platform retains the original 15 regression models and adds:
- ANFIS
- PSO-ANN
- GA-ANN
- GWO-ANN
- MPA-ANN — Marine Predators Algorithm optimized ANN
- OOA-ANN — Osprey Optimization Algorithm optimized ANN
- CGF-ANN — Fletcher–Reeves Conjugate Gradient optimized ANN
- ANFIS-MPA — MPA optimized ANFIS
- ANFIS-OOA — OOA optimized ANFIS
- ANFIS-ELM — ANFIS residual correction using Extreme Learning Machine

The previous MPA/OOA hidden-neuron 2–30 sweep has been removed. MPA-ANN and OOA-ANN now use the selected single ANN hidden-neuron value (default Nh=10).

Excel `.xlsx` upload is supported with sheet-name selection. The last column is the target and all preceding columns are predictors.

Results include dedicated **Training Result** and **Testing Result** sheets, observed-vs-predicted plots, convergence/iteration data, and a downloadable Excel workbook. CGF-ANN uses the Fletcher–Reeves nonlinear conjugate-gradient method and its RMSE/MAE iteration history is included in the results workbook.

## ANN-MPA manuscript convergence analysis
The final app includes a dedicated ANN-MPA convergence module for exactly 8 input predictors and the 8-17-1 and 8-26-1 architectures. It runs population sizes 5, 10, ..., 50 at 500 and 1000 iterations and records the actual best MSE at every MPA iteration. The app generates publication-style 600-DPI PNG convergence figures with an adjustable zoom inset and exports the complete iteration-by-iteration MSE/RMSE history, final train/test metrics, and actual/predicted values to Excel. No synthetic convergence values are generated.
