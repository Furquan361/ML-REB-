# MLRweb — GitHub Pages Frontend

Developed by Dr. Furquan Ahmad.

This repository provides a GitHub Pages frontend for the MLRweb Streamlit ML application.

## Architecture

GitHub Pages:
- hosts the public `github.io/MLRweb/` address
- provides the branded frontend

Streamlit Community Cloud:
- runs the Python ML backend
- handles Excel upload
- reads Excel sheet names
- trains the regression models
- displays Results & Analysis
- provides the Code Generator

## Current backend

The frontend currently embeds:

https://jvvkvgduxqwv5h3mea3e2c.streamlit.app/?embed=true

If the Streamlit app URL changes, edit the `src` value in `index.html`.

## GitHub Pages deployment

1. Create a GitHub repository named `MLRweb`.
2. Upload `index.html` and `README.md`.
3. Open repository Settings.
4. Select Pages.
5. Under Build and deployment choose:
   - Source: Deploy from a branch
   - Branch: main
   - Folder: / (root)
6. Save.
7. GitHub will provide a URL similar to:

https://YOUR-GITHUB-USERNAME.github.io/MLRweb/

## Important

The Python ML computation is still performed by Streamlit Community Cloud. GitHub Pages itself does not execute Python/scikit-learn/XGBoost.
