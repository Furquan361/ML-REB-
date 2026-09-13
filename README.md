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
