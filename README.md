# MLRweb — ML Workbench

**Developed by Dr. Furquan Ahmad**

A GitHub-ready Streamlit regression platform using Excel workbooks instead of CSV files.

## Main workflow

1. Upload `.xlsx`
2. Select Excel Sheet Name
3. Auto Split or Manual Split
4. Select preprocessing
5. Select regression models
6. Set hyperparameters
7. Run Training
8. View Results & Analysis
9. Export Excel results
10. Generate Colab code

## Excel format

For Auto Split:

- One `.xlsx` workbook
- Select the required sheet, e.g. `T`
- Last column = target
- All preceding columns = predictors

For Manual Split:

- `train.xlsx` + selected training sheet
- `test.xlsx` + selected testing sheet
- Last column in each = target

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the Streamlit URL shown in the terminal.

## GitHub

Suggested repository structure:

MLRweb/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
