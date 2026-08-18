
import streamlit as st
import pandas as pd
import numpy as np
import time
import io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
    BaggingRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False


# ============================================================
# PAGE
# ============================================================
st.set_page_config(
    page_title="MLRweb — ML Workbench",
    page_icon="ML",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# STYLE
# ============================================================
st.markdown("""
<style>
.stApp {
    background: #f7f9fc;
}

.topbar {
    background: #123f63;
    color: white;
    padding: 14px 20px;
    border-radius: 6px;
    margin-bottom: 12px;
}

.brand {
    font-size: 24px;
    font-weight: 800;
}

.developer {
    font-size: 12px;
    margin-top: 4px;
    opacity: 0.95;
}

.section-title {
    color: #173f60;
    font-weight: 800;
    border-left: 4px solid #173f60;
    padding-left: 8px;
    margin-top: 8px;
    margin-bottom: 10px;
}

.model-card {
    border: 1px solid #dce4ec;
    border-radius: 8px;
    padding: 8px 10px;
    margin-bottom: 6px;
    background: white;
}

.ready {
    display: inline-block;
    background: #e7f6ec;
    color: #15753a;
    border: 1px solid #b9e2c5;
    padding: 5px 12px;
    border-radius: 18px;
    font-weight: 700;
}
</style>

<div class="topbar">
    <div class="brand">MLRweb &nbsp; ML Workbench — Regression Platform</div>
    <div class="developer">
        Developed by <b>Dr. Furquan Ahmad</b> · GitHub Edition
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# MODEL NAMES
# ============================================================
MODEL_NAMES = {
    "LR": "Linear Regression",
    "Ridge": "Ridge Regression",
    "Lasso": "Lasso Regression",
    "EN": "Elastic Net",
    "DTR": "Decision Tree",
    "ABR": "AdaBoost",
    "GBR": "Gradient Boosting",
    "HGBR": "Hist GradBoost",
    "RFR": "Random Forest",
    "ETR": "Extra Trees",
    "BGR": "Bagging",
    "SVR": "Support Vector (SVR)",
    "KNN": "K-Nearest Neighbors",
    "XGB": "XGBoost",
    "MLP": "Neural Network (MLP)"
}


# ============================================================
# SESSION STATE
# ============================================================
if "results" not in st.session_state:
    st.session_state.results = None

if "trained_models" not in st.session_state:
    st.session_state.trained_models = {}

if "predictions" not in st.session_state:
    st.session_state.predictions = {}

if "dataset" not in st.session_state:
    st.session_state.dataset = None

if "sheet_name" not in st.session_state:
    st.session_state.sheet_name = None


# ============================================================
# FUNCTIONS
# ============================================================
def get_excel_sheets(uploaded_file):
    return pd.ExcelFile(uploaded_file).sheet_names


def read_excel_sheet(uploaded_file, sheet_name):
    return pd.read_excel(uploaded_file, sheet_name=sheet_name)


def make_models(p):
    models = {
        "LR": LinearRegression(),

        "Ridge": Ridge(
            alpha=p["Ridge"]
        ),

        "Lasso": Lasso(
            alpha=p["Lasso"][0],
            max_iter=p["Lasso"][1],
            random_state=1
        ),

        "EN": ElasticNet(
            alpha=p["EN"][0],
            l1_ratio=p["EN"][1],
            max_iter=p["EN"][2],
            random_state=1
        ),

        "DTR": DecisionTreeRegressor(
            max_depth=p["DTR"][0],
            min_samples_leaf=p["DTR"][1],
            min_samples_split=p["DTR"][2],
            random_state=1
        ),

        "ABR": AdaBoostRegressor(
            n_estimators=p["ABR"][0],
            learning_rate=p["ABR"][1],
            loss=p["ABR"][2],
            random_state=1
        ),

        "GBR": GradientBoostingRegressor(
            n_estimators=p["GBR"][0],
            learning_rate=p["GBR"][1],
            max_depth=p["GBR"][2],
            subsample=p["GBR"][3],
            random_state=1
        ),

        "HGBR": HistGradientBoostingRegressor(
            learning_rate=p["HGBR"][0],
            max_depth=p["HGBR"][1],
            max_iter=p["HGBR"][2],
            min_samples_leaf=p["HGBR"][3],
            random_state=1
        ),

        "RFR": RandomForestRegressor(
            n_estimators=p["RFR"][0],
            max_depth=p["RFR"][1],
            min_samples_leaf=p["RFR"][2],
            random_state=1,
            n_jobs=-1
        ),

        "ETR": ExtraTreesRegressor(
            n_estimators=p["ETR"][0],
            max_depth=p["ETR"][1],
            min_samples_leaf=p["ETR"][2],
            random_state=1,
            n_jobs=-1
        ),

        "BGR": BaggingRegressor(
            n_estimators=p["BGR"][0],
            max_samples=p["BGR"][1],
            max_features=p["BGR"][2],
            random_state=1,
            n_jobs=-1
        ),

        "SVR": SVR(
            kernel=p["SVR"][0],
            C=p["SVR"][1],
            epsilon=p["SVR"][2],
            degree=p["SVR"][3]
        ),

        "KNN": KNeighborsRegressor(
            n_neighbors=p["KNN"][0],
            weights=p["KNN"][1],
            p=p["KNN"][2]
        ),

        "MLP": MLPRegressor(
            hidden_layer_sizes=(p["MLP"][0], p["MLP"][1]),
            activation=p["MLP"][2],
            alpha=p["MLP"][3],
            max_iter=p["MLP"][4],
            random_state=1
        )
    }

    if XGB_AVAILABLE:
        models["XGB"] = xgb.XGBRegressor(
            n_estimators=p["XGB"][0],
            max_depth=p["XGB"][1],
            learning_rate=p["XGB"][2],
            alpha=p["XGB"][3],
            subsample=p["XGB"][4],
            colsample_bytree=p["XGB"][5],
            random_state=1,
            verbosity=0,
            objective="reg:squarederror"
        )

    return models


# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3 = st.tabs([
    "1  WORKBENCH",
    "2  RESULTS & ANALYSIS",
    "3  CODE GENERATOR"
])


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:

    st.markdown("### SPLIT MODE")

    split_mode = st.radio(
        "Split Mode",
        ["Auto Split", "Manual Split"]
    )

    if split_mode == "Auto Split":

        st.caption(
            "One Excel workbook — platform splits automatically"
        )

        test_percent = st.number_input(
            "Test %",
            min_value=5,
            max_value=50,
            value=20,
            step=5
        )

        seed = st.number_input(
            "Seed",
            min_value=0,
            max_value=999999,
            value=1
        )

        shuffle = st.selectbox(
            "Shuffle",
            ["Yes", "No"]
        )

    else:

        st.caption(
            "Separate training.xlsx + testing.xlsx"
        )

        train_file = st.file_uploader(
            "Training Excel",
            type=["xlsx"],
            key="train_excel"
        )

        train_sheet = None

        if train_file is not None:
            train_sheets = get_excel_sheets(train_file)

            train_sheet = st.selectbox(
                "Training Sheet Name",
                train_sheets,
                index=train_sheets.index("T")
                if "T" in train_sheets else 0
            )

        test_file = st.file_uploader(
            "Testing Excel",
            type=["xlsx"],
            key="test_excel"
        )

        test_sheet = None

        if test_file is not None:
            test_sheets = get_excel_sheets(test_file)

            test_sheet = st.selectbox(
                "Testing Sheet Name",
                test_sheets,
                index=test_sheets.index("T")
                if "T" in test_sheets else 0
            )

        seed = st.number_input(
            "Seed",
            min_value=0,
            max_value=999999,
            value=1,
            key="manual_seed"
        )

        shuffle = st.selectbox(
            "Shuffle",
            ["Yes", "No"],
            key="manual_shuffle"
        )

    st.markdown("### UPLOAD DATASET")

    if split_mode == "Auto Split":

        uploaded_file = st.file_uploader(
            "Excel File (.xlsx)",
            type=["xlsx"]
        )

        sheet_name = None

        if uploaded_file is not None:

            sheet_names = get_excel_sheets(
                uploaded_file
            )

            default_index = (
                sheet_names.index("T")
                if "T" in sheet_names
                else 0
            )

            sheet_name = st.selectbox(
                "Excel Sheet Name",
                sheet_names,
                index=default_index
            )

            st.session_state.sheet_name = sheet_name

    st.markdown("### PREPROCESSING")

    scaler_name = st.selectbox(
        "Scaler",
        [
            "StandardScaler",
            "MinMaxScaler",
            "RobustScaler",
            "None"
        ]
    )

    st.caption(
        "Recommended for SVR, KNN and MLP"
    )


# ============================================================
# WORKBENCH
# ============================================================
with tab1:

    left, right = st.columns(
        [0.36, 0.64]
    )

    # ========================================================
    # MODEL PANEL
    # ========================================================
    with left:

        st.markdown(
            '<div class="section-title">'
            'MODELS & HYPERPARAMETERS'
            '</div>',
            unsafe_allow_html=True
        )

        select_all = st.checkbox(
            "✓ ALL",
            value=True
        )

        selected_models = []

        def model_switch(code):
            enabled = st.checkbox(
                f"{code}  ·  {MODEL_NAMES[code]}",
                value=select_all,
                key=f"enable_{code}"
            )

            if enabled:
                selected_models.append(code)

        model_switch("LR")

        ridge_alpha = st.number_input(
            "Ridge alpha",
            0.0001, 100000.0, 1.0
        )
        model_switch("Ridge")

        lasso_alpha = st.number_input(
            "Lasso alpha",
            0.000001, 100.0, 0.001,
            format="%.6f"
        )
        lasso_iter = st.number_input(
            "Lasso max_iter",
            100, 100000, 1000, 100
        )
        model_switch("Lasso")

        en_alpha = st.number_input(
            "EN alpha",
            0.000001, 100.0, 0.001,
            format="%.6f"
        )
        en_l1 = st.number_input(
            "EN l1_ratio",
            0.0, 1.0, 0.5
        )
        en_iter = st.number_input(
            "EN max_iter",
            100, 100000, 1000, 100
        )
        model_switch("EN")

        tree_depth = st.number_input(
            "DTR max_depth",
            1, 100, 4
        )
        tree_leaf = st.number_input(
            "DTR min_leaf",
            1, 100, 2
        )
        tree_split = st.number_input(
            "DTR min_split",
            2, 100, 2
        )
        model_switch("DTR")

        ada_n = st.number_input(
            "ABR n_est",
            5, 2000, 45
        )
        ada_lr = st.number_input(
            "ABR lr",
            0.0001, 2.0, 0.35
        )
        ada_loss = st.selectbox(
            "ABR loss",
            ["linear", "square", "exponential"]
        )
        model_switch("ABR")

        gb_n = st.number_input(
            "GBR n_est",
            5, 2000, 45
        )
        gb_lr = st.number_input(
            "GBR lr",
            0.0001, 2.0, 0.35
        )
        gb_depth = st.number_input(
            "GBR max_depth",
            1, 100, 4
        )
        gb_subsample = st.number_input(
            "GBR subsample",
            0.1, 1.0, 1.0
        )
        model_switch("GBR")

        hgb_lr = st.number_input(
            "HGBR lr",
            0.0001, 2.0, 0.1
        )
        hgb_depth = st.number_input(
            "HGBR max_depth",
            1, 100, 4
        )
        hgb_iter = st.number_input(
            "HGBR max_iter",
            10, 2000, 100
        )
        hgb_leaf = st.number_input(
            "HGBR min_leaf",
            1, 500, 20
        )
        model_switch("HGBR")

        rf_n = st.number_input(
            "RFR n_est",
            5, 2000, 45
        )
        rf_depth = st.number_input(
            "RFR max_depth",
            1, 100, 4
        )
        rf_leaf = st.number_input(
            "RFR min_leaf",
            1, 100, 1
        )
        model_switch("RFR")

        et_n = st.number_input(
            "ETR n_est",
            5, 2000, 45
        )
        et_depth = st.number_input(
            "ETR max_depth",
            1, 100, 4
        )
        et_leaf = st.number_input(
            "ETR min_leaf",
            1, 100, 1
        )
        model_switch("ETR")

        bag_n = st.number_input(
            "BGR n_est",
            5, 2000, 45
        )
        bag_samples = st.number_input(
            "BGR max_samp",
            0.1, 1.0, 1.0
        )
        bag_features = st.number_input(
            "BGR max_feat",
            0.1, 1.0, 1.0
        )
        model_switch("BGR")

        svr_kernel = st.selectbox(
            "SVR kernel",
            ["rbf", "linear", "poly", "sigmoid"]
        )
        svr_c = st.number_input(
            "SVR C",
            0.0001, 10000000.0, 10000.0
        )
        svr_epsilon = st.number_input(
            "SVR epsilon",
            0.0, 1000.0, 10.0
        )
        svr_degree = st.number_input(
            "SVR degree",
            1, 10, 4
        )
        model_switch("SVR")

        knn_k = st.number_input(
            "KNN k",
            1, 100, 5
        )
        knn_weights = st.selectbox(
            "KNN weights",
            ["uniform", "distance"]
        )
        knn_p = st.number_input(
            "KNN p",
            1, 5, 2
        )
        model_switch("KNN")

        if XGB_AVAILABLE:

            xgb_n = st.number_input(
                "XGB n_est",
                5, 2000, 45
            )
            xgb_depth = st.number_input(
                "XGB max_depth",
                1, 100, 4
            )
            xgb_lr = st.number_input(
                "XGB lr",
                0.0001, 2.0, 0.35
            )
            xgb_alpha = st.number_input(
                "XGB alpha",
                0.0, 1000.0, 0.2
            )
            xgb_subsample = st.number_input(
                "XGB subsample",
                0.1, 1.0, 1.0
            )
            xgb_colsample = st.number_input(
                "XGB col_bt",
                0.1, 1.0, 1.0
            )
            model_switch("XGB")

        mlp_l1 = st.number_input(
            "MLP layer1",
            1, 1000, 100
        )
        mlp_l2 = st.number_input(
            "MLP layer2",
            1, 1000, 50
        )
        mlp_activation = st.selectbox(
            "MLP activation",
            ["relu", "tanh", "logistic"]
        )
        mlp_alpha = st.number_input(
            "MLP alpha",
            0.0000001, 10.0, 0.0001,
            format="%.7f"
        )
        mlp_iter = st.number_input(
            "MLP max_iter",
            100, 20000, 300, 100
        )
        model_switch("MLP")

    # ========================================================
    # MAIN PANEL
    # ========================================================
    with right:

        st.markdown(
            '<div class="section-title">'
            'UPLOAD / CONFIGURATION'
            '</div>',
            unsafe_allow_html=True
        )

        if split_mode == "Auto Split":

            if uploaded_file is None:

                st.info(
                    "Upload your Excel workbook from the left. "
                    "The available sheet names will appear automatically."
                )

            else:

                preview = read_excel_sheet(
                    uploaded_file,
                    sheet_name
                )

                st.write(
                    f"**Excel Sheet:** `{sheet_name}`"
                )

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Samples",
                    len(preview)
                )

                c2.metric(
                    "Columns",
                    len(preview.columns)
                )

                c3.metric(
                    "Target",
                    str(preview.columns[-1])
                )

                st.dataframe(
                    preview.head(10),
                    use_container_width=True,
                    height=250
                )

                st.caption(
                    "The last column is automatically treated as the target."
                )

        else:

            st.info(
                "Manual Split: upload separate training and testing Excel files."
            )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Models Enabled",
            len(selected_models)
        )

        c2.metric(
            "Split",
            split_mode
        )

        c3.metric(
            "Scaler",
            scaler_name
        )

        c4.markdown(
            '<div class="ready">● READY</div>',
            unsafe_allow_html=True
        )

        run_training = st.button(
            "▶ RUN TRAINING",
            type="primary",
            use_container_width=True
        )

        if run_training:

            if split_mode == "Auto Split":

                if uploaded_file is None:
                    st.error(
                        "Please upload an Excel (.xlsx) file."
                    )
                    st.stop()

                df = read_excel_sheet(
                    uploaded_file,
                    sheet_name
                ).dropna()

                X = df.iloc[:, :-1].apply(
                    pd.to_numeric,
                    errors="coerce"
                ).values

                y = pd.to_numeric(
                    df.iloc[:, -1],
                    errors="coerce"
                ).values

                valid = (
                    np.isfinite(X).all(axis=1)
                    & np.isfinite(y)
                )

                X = X[valid]
                y = y[valid]

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=test_percent / 100,
                    random_state=int(seed),
                    shuffle=shuffle == "Yes"
                )

            else:

                if train_file is None or test_file is None:
                    st.error(
                        "Upload both training and testing Excel files."
                    )
                    st.stop()

                train_df = read_excel_sheet(
                    train_file,
                    train_sheet
                ).dropna()

                test_df = read_excel_sheet(
                    test_file,
                    test_sheet
                ).dropna()

                X_train = train_df.iloc[:, :-1].apply(
                    pd.to_numeric,
                    errors="coerce"
                ).values

                y_train = pd.to_numeric(
                    train_df.iloc[:, -1],
                    errors="coerce"
                ).values

                X_test = test_df.iloc[:, :-1].apply(
                    pd.to_numeric,
                    errors="coerce"
                ).values

                y_test = pd.to_numeric(
                    test_df.iloc[:, -1],
                    errors="coerce"
                ).values

            if scaler_name == "StandardScaler":
                scaler = StandardScaler()
            elif scaler_name == "MinMaxScaler":
                scaler = MinMaxScaler()
            elif scaler_name == "RobustScaler":
                scaler = RobustScaler()
            else:
                scaler = None

            if scaler is not None:

                X_train = scaler.fit_transform(
                    X_train
                )

                X_test = scaler.transform(
                    X_test
                )

            params = {
                "Ridge": ridge_alpha,
                "Lasso": (
                    lasso_alpha,
                    lasso_iter
                ),
                "EN": (
                    en_alpha,
                    en_l1,
                    en_iter
                ),
                "DTR": (
                    tree_depth,
                    tree_leaf,
                    tree_split
                ),
                "ABR": (
                    ada_n,
                    ada_lr,
                    ada_loss
                ),
                "GBR": (
                    gb_n,
                    gb_lr,
                    gb_depth,
                    gb_subsample
                ),
                "HGBR": (
                    hgb_lr,
                    hgb_depth,
                    hgb_iter,
                    hgb_leaf
                ),
                "RFR": (
                    rf_n,
                    rf_depth,
                    rf_leaf
                ),
                "ETR": (
                    et_n,
                    et_depth,
                    et_leaf
                ),
                "BGR": (
                    bag_n,
                    bag_samples,
                    bag_features
                ),
                "SVR": (
                    svr_kernel,
                    svr_c,
                    svr_epsilon,
                    svr_degree
                ),
                "KNN": (
                    knn_k,
                    knn_weights,
                    knn_p
                ),
                "MLP": (
                    mlp_l1,
                    mlp_l2,
                    mlp_activation,
                    mlp_alpha,
                    mlp_iter
                )
            }

            if XGB_AVAILABLE:
                params["XGB"] = (
                    xgb_n,
                    xgb_depth,
                    xgb_lr,
                    xgb_alpha,
                    xgb_subsample,
                    xgb_colsample
                )

            all_models = make_models(params)

            models = {
                key: all_models[key]
                for key in selected_models
            }

            results = []
            fitted = {}
            prediction_data = {}

            progress = st.progress(0)
            status = st.empty()

            for i, (name, model) in enumerate(
                models.items(),
                start=1
            ):

                status.info(
                    f"Training {name} — {MODEL_NAMES[name]}"
                )

                start_time = time.time()

                try:

                    model.fit(
                        X_train,
                        y_train
                    )

                    pred_train = model.predict(
                        X_train
                    )

                    pred_test = model.predict(
                        X_test
                    )

                    elapsed = time.time() - start_time

                    results.append({
                        "Model": name,
                        "R2_Train": r2_score(
                            y_train,
                            pred_train
                        ),
                        "R2_Test": r2_score(
                            y_test,
                            pred_test
                        ),
                        "RMSE_Test": np.sqrt(
                            mean_squared_error(
                                y_test,
                                pred_test
                            )
                        ),
                        "MAE_Test": mean_absolute_error(
                            y_test,
                            pred_test
                        ),
                        "Time_s": elapsed
                    })

                    fitted[name] = model

                    prediction_data[name] = pd.DataFrame({
                        "Observed": y_test,
                        "Predicted": pred_test,
                        "Residual": y_test - pred_test
                    })

                except Exception as error:

                    results.append({
                        "Model": name,
                        "R2_Train": np.nan,
                        "R2_Test": np.nan,
                        "RMSE_Test": np.nan,
                        "MAE_Test": np.nan,
                        "Time_s": time.time() - start_time,
                        "Error": str(error)
                    })

                progress.progress(
                    i / len(models)
                )

            results_df = pd.DataFrame(
                results
            ).sort_values(
                "R2_Test",
                ascending=False,
                na_position="last"
            ).reset_index(drop=True)

            st.session_state.results = results_df
            st.session_state.trained_models = fitted
            st.session_state.predictions = prediction_data

            status.success(
                "Training completed successfully."
            )

            st.dataframe(
                results_df,
                use_container_width=True
            )


# ============================================================
# RESULTS & ANALYSIS
# ============================================================
with tab2:

    st.markdown(
        '<div class="section-title">'
        'RESULTS & ANALYSIS'
        '</div>',
        unsafe_allow_html=True
    )

    results = st.session_state.results

    if results is None:

        st.info(
            "Run Training from the Workbench tab first."
        )

    else:

        valid = results.dropna(
            subset=["R2_Test"]
        )

        if len(valid) > 0:

            best = valid.iloc[0]

            a, b, c, d = st.columns(4)

            a.metric(
                "Best Model",
                best["Model"]
            )

            b.metric(
                "R² Test",
                f'{best["R2_Test"]:.4f}'
            )

            c.metric(
                "RMSE Test",
                f'{best["RMSE_Test"]:.4f}'
            )

            d.metric(
                "MAE Test",
                f'{best["MAE_Test"]:.4f}'
            )

        st.subheader(
            "Model Comparison"
        )

        st.dataframe(
            results.style.format({
                "R2_Train": "{:.4f}",
                "R2_Test": "{:.4f}",
                "RMSE_Test": "{:.4f}",
                "MAE_Test": "{:.4f}",
                "Time_s": "{:.3f}"
            }),
            use_container_width=True
        )

        excel_buffer = io.BytesIO()

        with pd.ExcelWriter(
            excel_buffer,
            engine="openpyxl"
        ) as writer:

            results.to_excel(
                writer,
                index=False,
                sheet_name="Results"
            )

            for model_name, prediction_df in (
                st.session_state.predictions.items()
            ):

                prediction_df.to_excel(
                    writer,
                    index=False,
                    sheet_name=f"{model_name}_Pred"[:31]
                )

        st.download_button(
            "⬇ DOWNLOAD MODEL_RESULTS.XLSX",
            excel_buffer.getvalue(),
            "MLRweb_Model_Results.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.subheader(
            "Observed vs Predicted"
        )

        selected_model = st.selectbox(
            "Select Model",
            list(
                st.session_state.trained_models.keys()
            )
        )

        prediction_df = st.session_state.predictions[
            selected_model
        ]

        st.line_chart(
            prediction_df[
                ["Observed", "Predicted"]
            ]
        )

        st.download_button(
            "⬇ DOWNLOAD PREDICTIONS",
            prediction_df.to_csv(index=False),
            f"{selected_model}_Predictions.csv",
            "text/csv"
        )


# ============================================================
# CODE GENERATOR
# ============================================================
with tab3:

    st.markdown(
        '<div class="section-title">'
        'CODE GENERATOR'
        '</div>',
        unsafe_allow_html=True
    )

    if st.session_state.results is None:

        st.info(
            "Run Training first to generate the configuration code."
        )

    else:

        st.code(
            """# MLRweb — Auto Generated Code
# Developed by Dr. Furquan Ahmad

from google.colab import files
import pandas as pd

uploaded = files.upload()
file_name = list(uploaded.keys())[0]

# Change this to your Excel sheet
SHEET_NAME_INPUT = "T"

df = pd.read_excel(
    file_name,
    sheet_name=SHEET_NAME_INPUT
)

print("Uploaded file:", file_name)
print("Data loaded from sheet:", SHEET_NAME_INPUT)
print("Number of samples:", len(df))

# Last column = target
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
""",
            language="python"
        )


st.divider()

st.caption(
    "MLRweb — ML Workbench | Developed by Dr. Furquan Ahmad | "
    "Excel Sheet Name Edition"
)
