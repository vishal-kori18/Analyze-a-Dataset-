from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


APP_DIR = Path(__file__).resolve().parent
SAMPLE_DATA = APP_DIR / "data" / "wine_classification.csv"


st.set_page_config(
    page_title="DataLens | Dataset Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
    .stApp { background: #f5f7fb; }
    [data-testid="stSidebar"] { background: #111827; }
    [data-testid="stSidebar"] * { color: #eef2ff; }
    [data-testid="stSidebar"] [data-baseweb="select"] * { color: #111827; }
    .hero { padding: 1.35rem 1.6rem; border-radius: 18px; margin: 0.3rem 0 1.2rem;
            color: white; background: linear-gradient(115deg, #172554, #2563eb 60%, #38bdf8); }
    .hero h1 { margin: 0 0 .3rem; font-size: 2rem; }
    .hero p { margin: 0; opacity: .88; }
    [data-testid="stMetric"] { background: #fff; padding: 1rem; border-radius: 14px;
        border: 1px solid #e5eaf3; box-shadow: 0 2px 10px rgba(20,40,80,.04); }
    .section-note { color: #64748b; font-size: .92rem; }
    div[data-testid="stDataFrame"] { border-radius: 12px; }
</style>
"""


def load_sample_data() -> pd.DataFrame:
    """Load the bundled, real Wine recognition dataset from the UCI repository."""
    return pd.read_csv(SAMPLE_DATA)


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    return frame.select_dtypes(include=np.number).columns.tolist()


def categorical_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in numeric_columns(frame)]


def outlier_counts(frame: pd.DataFrame, multiplier: float = 1.5) -> pd.Series:
    """Return IQR outlier counts for each numeric column (nulls are ignored)."""
    counts: dict[str, int] = {}
    for column in numeric_columns(frame):
        values = frame[column].dropna()
        if values.empty:
            counts[column] = 0
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        spread = q3 - q1
        if pd.isna(spread) or spread == 0:
            counts[column] = 0
        else:
            counts[column] = int(((values < q1 - multiplier * spread) | (values > q3 + multiplier * spread)).sum())
    return pd.Series(counts, dtype="int64", name="Outliers")


def clean_dataset(
    frame: pd.DataFrame,
    missing_action: str = "Leave as-is",
    remove_duplicates: bool = False,
    outlier_action: str = "Keep",
    multiplier: float = 1.5,
) -> pd.DataFrame:
    """Apply user-selected cleaning rules and return a new DataFrame."""
    cleaned = frame.copy()
    if missing_action == "Drop rows with missing values":
        cleaned = cleaned.dropna()
    elif missing_action == "Fill numeric with median and categories with mode":
        for column in cleaned.columns:
            if cleaned[column].isna().any():
                if pd.api.types.is_numeric_dtype(cleaned[column]):
                    median = cleaned[column].median()
                    cleaned[column] = cleaned[column].fillna(median)
                else:
                    modes = cleaned[column].mode(dropna=True)
                    if not modes.empty:
                        cleaned[column] = cleaned[column].fillna(modes.iloc[0])
    if remove_duplicates:
        cleaned = cleaned.drop_duplicates()
    numeric = numeric_columns(cleaned)
    if outlier_action in {"Cap values at IQR limits", "Remove rows with IQR outliers"} and numeric:
        outlier_mask = pd.Series(False, index=cleaned.index)
        for column in numeric:
            values = cleaned[column].dropna()
            if values.empty:
                continue
            q1, q3 = values.quantile([0.25, 0.75])
            spread = q3 - q1
            if pd.isna(spread) or spread == 0:
                continue
            lower, upper = q1 - multiplier * spread, q3 + multiplier * spread
            outside = (cleaned[column] < lower) | (cleaned[column] > upper)
            outlier_mask |= outside.fillna(False)
            if outlier_action == "Cap values at IQR limits":
                cleaned[column] = cleaned[column].clip(lower, upper)
        if outlier_action == "Remove rows with IQR outliers":
            cleaned = cleaned.loc[~outlier_mask]
    return cleaned.reset_index(drop=True)


def make_quality_table(frame: pd.DataFrame) -> pd.DataFrame:
    missing = frame.isna().sum()
    return pd.DataFrame(
        {
            "Column": frame.columns,
            "Data type": [str(dtype) for dtype in frame.dtypes],
            "Missing": missing.values,
            "Missing %": (missing.values / max(len(frame), 1) * 100).round(1),
            "Unique values": [frame[column].nunique(dropna=True) for column in frame.columns],
        }
    )


def automatic_insights(frame: pd.DataFrame) -> list[str]:
    """Generate clear, data-grounded observations without assuming domain context."""
    insights: list[str] = []
    if frame.empty:
        return ["The current dataset has no rows. Adjust the cleaning settings to continue."]

    missing_total = int(frame.isna().sum().sum())
    duplicate_total = int(frame.duplicated().sum())
    if missing_total:
        insights.append(f"There are **{missing_total:,} missing cells** across the dataset; check the Data quality page before modeling.")
    else:
        insights.append("The dataset has **no missing cells** in its current cleaned view.")
    if duplicate_total:
        insights.append(f"The current view contains **{duplicate_total:,} duplicate rows**.")
    else:
        insights.append("No duplicate rows remain in the current view.")

    numeric = numeric_columns(frame)
    if numeric:
        spread = frame[numeric].std(numeric_only=True).dropna()
        if not spread.empty:
            varied = spread.sort_values(ascending=False).index[0]
            insights.append(f"**{varied}** has the largest standard deviation among numeric columns ({spread[ varied ]:,.3g}), so its values vary the most on their original scale.")
        corr = frame[numeric].corr().abs()
        if len(numeric) > 1:
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().sort_values(ascending=False)
            if not upper.empty and float(upper.iloc[0]) >= 0.5:
                (left, right), value = upper.index[0], float(upper.iloc[0])
                signed = float(frame[[left, right]].corr().iloc[0, 1])
                direction = "positive" if signed > 0 else "negative"
                insights.append(f"**{left}** and **{right}** show the strongest numeric relationship (Pearson r = {signed:.2f}, a {direction} association). Correlation does not establish causation.")
            elif not upper.empty:
                insights.append(f"No pair of numeric columns has an absolute Pearson correlation of 0.50 or higher; the strongest pair is **{upper.index[0][0]}** and **{upper.index[0][1]}** (|r| = {upper.iloc[0]:.2f}).")
        top_mean = frame[numeric].mean(numeric_only=True).dropna()
        if not top_mean.empty:
            col = top_mean.abs().sort_values(ascending=False).index[0]
            insights.append(f"The average of **{col}** is {top_mean[col]:,.3g}. Compare scales carefully before ranking unlike measurements.")

    date_columns: list[tuple[str, pd.Series]] = []
    for column in frame.columns:
        if "date" in str(column).lower() or "time" in str(column).lower():
            parsed = pd.to_datetime(frame[column], errors="coerce")
            if parsed.notna().mean() >= 0.6:
                date_columns.append((column, parsed))
    if date_columns and numeric:
        date_name, dates = date_columns[0]
        valid = pd.DataFrame({"date": dates, **{name: frame[name] for name in numeric}}).dropna(subset=["date"])
        if len(valid) >= 4:
            valid = valid.sort_values("date")
            width = max(1, len(valid) // 4)
            for column in numeric[:3]:
                early = valid[column].iloc[:width].mean()
                late = valid[column].iloc[-width:].mean()
                if pd.notna(early) and pd.notna(late) and early != 0:
                    change = (late - early) / abs(early) * 100
                    direction = "increased" if change > 0 else "decreased"
                    insights.append(f"Across the earliest and latest date quartiles in **{date_name}**, mean **{column}** {direction} by about {abs(change):.1f}% (descriptive comparison).")
    if not insights:
        insights.append("This dataset is ready to explore. Select columns in Visualize to uncover distributions and relationships.")
    return insights


def make_report(frame: pd.DataFrame, source_name: str) -> str:
    quality = make_quality_table(frame)
    numeric = numeric_columns(frame)
    stats = frame[numeric].describe().round(3).to_markdown() if numeric else "No numeric columns available."
    insight_lines = "\n".join(f"- {item.replace('**', '')}" for item in automatic_insights(frame))
    return (
        "# DataLens Analysis Report\n\n"
        f"- **Dataset:** {source_name}\n- **Rows:** {len(frame):,}\n- **Columns:** {len(frame.columns):,}\n"
        f"- **Duplicate rows:** {int(frame.duplicated().sum()):,}\n- **Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        "## Column quality\n\n" + quality.to_markdown(index=False) + "\n\n"
        "## Numeric descriptive statistics\n\n" + stats + "\n\n"
        "## Automatic insights\n\n" + insight_lines + "\n\n"
        "> Correlations and automatically generated observations are descriptive. They do not establish causation or replace domain expertise.\n"
    )


def model_pipeline(frame: pd.DataFrame, target: str, task: str) -> tuple[Pipeline, list[str], list[str]]:
    features = [column for column in frame.columns if column != target]
    numeric = [column for column in features if pd.api.types.is_numeric_dtype(frame[column])]
    categorical = [column for column in features if column not in numeric]
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True))])
        transformers.append(("numeric", numeric_pipeline, numeric))
    if categorical:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    estimator = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced_subsample", n_jobs=-1) if task == "classification" else RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    return pipeline, numeric, categorical


def render_seaborn_missing_chart(frame: pd.DataFrame) -> None:
    missing = frame.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        st.success("No missing values in the current dataset.")
        return
    fig, ax = plt.subplots(figsize=(9, max(2.8, min(5.5, 0.38 * len(missing)))))
    sns.barplot(x=missing.values, y=missing.index.astype(str), color="#3b82f6", ax=ax)
    ax.set_title("Missing values by column")
    ax.set_xlabel("Missing cells")
    ax.set_ylabel("")
    sns.despine(ax=ax)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def show_overview(frame: pd.DataFrame, raw: pd.DataFrame, source_name: str) -> None:
    st.subheader("Dataset overview")
    st.caption(f"Currently exploring **{source_name}**. The sample dataset is the UCI Wine recognition dataset bundled with scikit-learn.")
    a, b, c, d = st.columns(4)
    a.metric("Rows", f"{len(frame):,}", delta=f"{len(frame) - len(raw):+,} vs. original" if len(frame) != len(raw) else None)
    b.metric("Columns", f"{len(frame.columns):,}")
    c.metric("Missing cells", f"{int(frame.isna().sum().sum()):,}")
    d.metric("Duplicate rows", f"{int(frame.duplicated().sum()):,}")
    st.markdown("#### Data preview")
    st.dataframe(frame.head(12), width="stretch", hide_index=True)
    left, right = st.columns([1.05, 1])
    with left:
        st.markdown("#### Column profile")
        st.dataframe(make_quality_table(frame), width="stretch", hide_index=True)
    with right:
        st.markdown("#### Descriptive statistics")
        try:
            st.dataframe(frame.describe(include="all").transpose(), width="stretch")
        except (TypeError, ValueError) as exc:
            st.warning(f"Statistics could not be calculated for this dataset: {exc}")
    st.markdown("#### Data quality snapshot")
    render_seaborn_missing_chart(frame)


def show_quality(frame: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    st.subheader("Data quality & cleaning")
    st.markdown("Select rules below, then apply them to the working copy. Your original upload remains unchanged for this session.")
    st.dataframe(make_quality_table(frame), width="stretch", hide_index=True)
    duplicates = int(frame.duplicated().sum())
    counts = outlier_counts(frame)
    left, right = st.columns(2)
    with left:
        st.metric("Exact duplicate rows", f"{duplicates:,}")
        st.caption("Duplicate rows are complete records whose cell values match earlier rows exactly.")
    with right:
        total_outliers = int(counts.sum()) if not counts.empty else 0
        st.metric("Numeric IQR outlier cells", f"{total_outliers:,}")
        st.caption("A value is flagged when it falls more than the selected IQR multiplier beyond Q1 or Q3.")
    if not counts.empty:
        st.dataframe(counts.rename_axis("Column").reset_index(), width="stretch", hide_index=True)

    with st.form("cleaning_form"):
        missing_action = st.selectbox("Missing-value treatment", ["Leave as-is", "Drop rows with missing values", "Fill numeric with median and categories with mode"])
        remove_duplicates = st.checkbox("Remove duplicate rows", value=duplicates > 0)
        outlier_action = st.selectbox("Numeric outlier treatment", ["Keep", "Cap values at IQR limits", "Remove rows with IQR outliers"])
        multiplier = st.slider("IQR multiplier", min_value=1.0, max_value=3.0, value=1.5, step=0.25, help="1.5 is the common Tukey rule; higher values are less aggressive.")
        submitted = st.form_submit_button("Apply cleaning to working dataset", type="primary")
    updated = clean_dataset(frame, missing_action, remove_duplicates, outlier_action, multiplier) if submitted else frame
    return updated, submitted


def show_visualizations(frame: pd.DataFrame) -> None:
    st.subheader("Interactive visualizations")
    numeric = numeric_columns(frame)
    categorical = [column for column in frame.columns if column not in numeric]
    chart_type = st.selectbox("Chart type", ["Histogram", "Bar chart", "Scatter plot", "Box plot", "Correlation heatmap"])
    if chart_type == "Histogram":
        if not numeric:
            st.info("A histogram needs at least one numeric column.")
            return
        column = st.selectbox("Numeric column", numeric)
        bins = st.slider("Number of bins", 5, 80, 30)
        fig = px.histogram(frame, x=column, nbins=bins, marginal="box", color_discrete_sequence=["#2563eb"], title=f"Distribution of {column}")
        fig.update_layout(bargap=0.08, template="plotly_white")
        st.plotly_chart(fig, width="stretch")
    elif chart_type == "Bar chart":
        choices = categorical + numeric
        if not choices:
            st.info("There are no columns to chart.")
            return
        column = st.selectbox("Column", choices)
        if column in categorical or frame[column].nunique(dropna=True) <= 30:
            counts = frame[column].astype("string").fillna("(missing)").value_counts().head(30).rename_axis("Value").reset_index(name="Count")
            fig = px.bar(counts, x="Value", y="Count", color="Count", color_continuous_scale="Blues", title=f"Top value counts: {column}")
        else:
            fig = px.histogram(frame, x=column, nbins=25, title=f"Value distribution: {column}", color_discrete_sequence=["#0ea5e9"])
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, width="stretch")
    elif chart_type == "Scatter plot":
        if len(numeric) < 2:
            st.info("A scatter plot needs at least two numeric columns.")
            return
        left, right = st.columns(2)
        x_col = left.selectbox("X axis", numeric, index=0)
        y_col = right.selectbox("Y axis", numeric, index=1 if len(numeric) > 1 else 0)
        color_col = st.selectbox("Color by (optional)", ["None"] + categorical)
        fig = px.scatter(frame, x=x_col, y=y_col, color=None if color_col == "None" else color_col, opacity=0.78, trendline="ols" if len(frame) >= 3 else None, title=f"{y_col} vs. {x_col}")
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, width="stretch")
    elif chart_type == "Box plot":
        if not numeric:
            st.info("A box plot needs a numeric column.")
            return
        column = st.selectbox("Numeric column", numeric)
        group = st.selectbox("Group by (optional)", ["None"] + categorical)
        fig = px.box(frame, y=column, x=None if group == "None" else group, points="outliers", color=None if group == "None" else group, title=f"Spread of {column}")
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, width="stretch")
    else:
        if len(numeric) < 2:
            st.info("A correlation heatmap needs at least two numeric columns.")
            return
        selected = st.multiselect("Numeric columns", numeric, default=numeric[:min(12, len(numeric))], max_selections=30)
        if len(selected) < 2:
            st.info("Choose at least two numeric columns to draw the heatmap.")
            return
        corr = frame[selected].corr().round(3)
        fig = px.imshow(corr, text_auto=True, aspect="auto", zmin=-1, zmax=1, color_continuous_scale="RdBu_r", title="Pearson correlation matrix")
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, width="stretch")
        st.caption("Pearson correlation measures linear association between numeric features; it does not imply causation.")


def show_insights(frame: pd.DataFrame) -> None:
    st.subheader("Automatic insights")
    st.caption("Observations are generated from the current cleaned view and should be interpreted with domain context.")
    for item in automatic_insights(frame):
        st.markdown(f"- {item}")
    numeric = numeric_columns(frame)
    if len(numeric) >= 2:
        corr = frame[numeric].corr()
        values = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().rename("Pearson r").reset_index()
        values.columns = ["Feature A", "Feature B", "Pearson r"]
        values["|r|"] = values["Pearson r"].abs()
        values = values.sort_values("|r|", ascending=False).drop(columns="|r|").head(15)
        st.markdown("#### Strongest numeric correlations")
        st.dataframe(values, width="stretch", hide_index=True)
        st.caption("Correlation is not evidence of causation. Pearson r can miss nonlinear relationships and is sensitive to outliers.")


def show_machine_learning(frame: pd.DataFrame) -> None:
    st.subheader("Basic machine learning")
    st.caption("Train a quick Random Forest using a holdout split. Preprocessing (median imputation and one-hot encoding) happens inside the pipeline to reduce leakage.")
    if len(frame.columns) < 2:
        st.info("Machine learning needs at least one feature column and one target column.")
        return
    target = st.selectbox("Choose the target to predict", frame.columns)
    clean = frame.dropna(subset=[target]).copy()
    if clean.empty:
        st.warning("The selected target has no non-missing values.")
        return
    unique_count = clean[target].nunique()
    inferred = "classification" if (not pd.api.types.is_numeric_dtype(clean[target]) or unique_count <= 10) else "regression"
    task = st.radio("Prediction type", ["classification", "regression"], index=0 if inferred == "classification" else 1, horizontal=True)
    if task == "classification" and unique_count < 2:
        st.warning("Classification needs at least two distinct target classes.")
        return
    if task == "regression" and not pd.api.types.is_numeric_dtype(clean[target]):
        st.warning("Regression requires a numeric target. Select classification for this column.")
        return
    if len(clean) < 8:
        st.warning("Use at least 8 complete target rows for a meaningful train/test demonstration.")
        return
    if clean.drop(columns=[target]).shape[1] == 0:
        st.warning("No feature columns remain after selecting the target.")
        return
    if st.button("Train Random Forest", type="primary"):
        X = clean.drop(columns=[target])
        y = clean[target]
        stratify = None
        if task == "classification":
            class_counts = y.value_counts()
            test_count = max(1, int(np.ceil(0.25 * len(y))))
            if class_counts.min() >= 2 and test_count >= len(class_counts) and len(y) - test_count >= len(class_counts):
                stratify = y
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=stratify)
            model, _, _ = model_pipeline(X_train.assign(**{target: y_train}), target, task)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            if task == "classification":
                score = accuracy_score(y_test, predictions)
                st.metric("Holdout accuracy", f"{score:.1%}")
                labels = model.named_steps["model"].classes_
                matrix = confusion_matrix(y_test, predictions, labels=labels)
                fig = px.imshow(matrix, x=[str(x) for x in labels], y=[str(x) for x in labels], text_auto=True, labels={"x": "Predicted", "y": "Actual", "color": "Count"}, title="Confusion matrix")
                fig.update_layout(template="plotly_white")
                st.plotly_chart(fig, width="stretch")
            else:
                left, right = st.columns(2)
                left.metric("Mean absolute error", f"{mean_absolute_error(y_test, predictions):.4g}")
                score = r2_score(y_test, predictions) if len(y_test) >= 2 else np.nan
                right.metric("R² score", f"{score:.3f}" if np.isfinite(score) else "Not available")
                fig = px.scatter(x=y_test, y=predictions, labels={"x": "Actual", "y": "Predicted"}, title="Actual vs. predicted")
                min_value = min(float(np.min(y_test)), float(np.min(predictions)))
                max_value = max(float(np.max(y_test)), float(np.max(predictions)))
                fig.add_trace(go.Scatter(x=[min_value, max_value], y=[min_value, max_value], mode="lines", name="Perfect prediction", line={"dash": "dash", "color": "#64748b"}))
                fig.update_layout(template="plotly_white")
                st.plotly_chart(fig, width="stretch")
            st.caption(f"Model trained on {len(X_train):,} rows and evaluated on {len(X_test):,} holdout rows. Results depend on the selected target and random split.")
            importances = model.named_steps["model"].feature_importances_
            names = model.named_steps["preprocessor"].get_feature_names_out()
            importance = pd.DataFrame({"Feature": names, "Importance": importances}).sort_values("Importance", ascending=False).head(15)
            st.markdown("#### Most influential model features")
            st.plotly_chart(px.bar(importance.sort_values("Importance"), x="Importance", y="Feature", orientation="h", title="Random Forest feature importance").update_layout(template="plotly_white"), width="stretch")
            st.caption("Feature importance describes the fitted model, not causal influence. One-hot encoding expands categorical columns into indicators.")
        except (ValueError, TypeError, KeyError) as exc:
            st.error(f"The model could not be trained on this selection: {exc}. Try another target or clean the data first.")


def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown('<div class="hero"><h1>DataLens</h1><p>Explore, clean, visualize, and learn from your data — all in one workspace.</p></div>', unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("## DataLens")
        st.caption("AI & Machine Learning · Dataset Analyzer")
        source = st.radio("Dataset source", ["Built-in sample", "Upload a CSV"], index=0)
        uploaded = None
        if source == "Upload a CSV":
            uploaded = st.file_uploader("Choose a CSV file", type=["csv"], help="Files are processed in this Streamlit session and are not uploaded to an external service.")
        page = st.radio("Navigate", ["Overview", "Data quality", "Visualize", "Insights", "Machine learning", "Export"], label_visibility="visible")
        st.markdown("---")
        st.caption("Tip: begin with the Overview and Data quality pages, then explore charts or train a model.")

    if source == "Built-in sample":
        source_name = "Wine recognition sample"
        source_key = "sample-wine"
        try:
            current_raw = load_sample_data()
        except (FileNotFoundError, pd.errors.ParserError) as exc:
            st.error(f"The bundled sample dataset could not be loaded: {exc}")
            st.stop()
    else:
        if uploaded is None:
            st.info("Upload a CSV file in the sidebar to begin. You can also select **Built-in sample** to explore the included Wine dataset.")
            st.stop()
        source_name = uploaded.name
        upload_hash = hashlib.sha256(uploaded.getvalue()).hexdigest()
        source_key = f"upload:{uploaded.name}:{upload_hash}"
        try:
            current_raw = pd.read_csv(uploaded)
            if len(current_raw.columns) == 0:
                st.error("This CSV has no columns. Please upload a valid comma-separated dataset.")
                st.stop()
        except (pd.errors.ParserError, UnicodeDecodeError, ValueError, OSError) as exc:
            st.error(f"Could not read this CSV: {exc}. Check the delimiter, encoding, and file contents.")
            st.stop()

    if current_raw.empty:
        st.warning("This dataset contains no rows. Upload a CSV with at least one data row.")
        st.stop()
    if st.session_state.get("dataset_key") != source_key:
        st.session_state.dataset_key = source_key
        st.session_state.raw_data = current_raw.copy()
        st.session_state.clean_data = current_raw.copy()
    raw = st.session_state.raw_data
    frame = st.session_state.clean_data

    if page == "Data quality" and st.session_state.get("clean_notice"):
        st.success(st.session_state.pop("clean_notice"))

    if page == "Overview":
        show_overview(frame, raw, source_name)
    elif page == "Data quality":
        updated, submitted = show_quality(frame)
        if submitted:
            st.session_state.clean_data = updated
            st.session_state.clean_notice = f"Cleaning applied. Working dataset now has {len(updated):,} rows and {len(updated.columns):,} columns."
            st.rerun()
    elif page == "Visualize":
        show_visualizations(frame)
    elif page == "Insights":
        show_insights(frame)
    elif page == "Machine learning":
        show_machine_learning(frame)
    else:
        st.subheader("Export your work")
        st.markdown("Download the current cleaned working dataset and a concise Markdown report for your notes or submission.")
        csv_bytes = frame.to_csv(index=False).encode("utf-8-sig")
        report = make_report(frame, source_name)
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("Download cleaned CSV", data=csv_bytes, file_name="cleaned_dataset.csv", mime="text/csv", type="primary", width="stretch")
            st.caption(f"{len(frame):,} rows · {len(frame.columns):,} columns · UTF-8 with BOM for spreadsheet compatibility")
        with col2:
            st.download_button("Download analysis report (.md)", data=report, file_name="analysis_report.md", mime="text/markdown", width="stretch")
            st.caption("Includes dataset profile, missingness, statistics, duplicates, and automatically generated insights.")
        st.markdown("#### Report preview")
        st.markdown(report)


if __name__ == "__main__":
    main()
