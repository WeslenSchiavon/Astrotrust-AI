from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


ROOT_DIR = Path(__file__).resolve().parents[1]

CACHE_DIR = (
    ROOT_DIR
    / "data"
    / "processed"
    / "elasticc2_large"
    / "dashboard_cache"
)

POLICY_FILES = {
    "Novelty + rarity": "hybrid_test_ranking_novelty_rarity.csv",
    "Rarity only": "hybrid_test_ranking_rarity_only.csv",
    "Previous discovery": "hybrid_test_ranking_previous_discovery.csv",
    "Fixed discovery": "hybrid_test_ranking_fixed_discovery.csv",
}

BAND_MAP = {
    "0": "u",
    "1": "g",
    "2": "r",
    "3": "i",
    "4": "z",
    "5": "Y",
    0: "u",
    1: "g",
    2: "r",
    3: "i",
    4: "z",
    5: "Y",
}


st.set_page_config(
    page_title="AstroTrust-AI Dashboard",
    page_icon="🔭",
    layout="wide",
)


@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


@st.cache_data
def load_lightcurves(path):
    return pd.read_parquet(path)


def normalize_band(value):
    if value in BAND_MAP:
        return BAND_MAP[value]

    value_str = str(value).strip()

    if value_str in BAND_MAP:
        return BAND_MAP[value_str]

    if value_str.lower() == "y":
        return "Y"

    return value_str


def load_class_names():
    path = CACHE_DIR / "full_class_counts.csv"

    if not path.exists():
        return {}

    df = load_csv(path)

    if "label" not in df.columns or "class_name" not in df.columns:
        return {}

    return dict(zip(df["label"].astype(int), df["class_name"].astype(str)))


def add_class_names(df, class_names):
    out = df.copy()

    if "true_label" in out.columns:
        out["true_class_name"] = out["true_label"].map(lambda x: class_names.get(int(x), str(x)))

    if "predicted_label" in out.columns:
        out["predicted_class_name"] = out["predicted_label"].map(lambda x: class_names.get(int(x), str(x)))

    return out


def metric_card(label, value, fmt="{:.4f}"):
    if value is None or pd.isna(value):
        st.metric(label, "—")
    elif isinstance(value, (float, np.floating)):
        st.metric(label, fmt.format(value))
    else:
        st.metric(label, value)


def plot_model_performance(perf):
    plot_df = perf.copy()
    plot_df = plot_df.sort_values("macro_f1", ascending=True)

    fig = px.bar(
        plot_df,
        x="macro_f1",
        y="model",
        orientation="h",
        color="family",
        hover_data=["accuracy", "balanced_accuracy", "weighted_f1", "dataset"],
        title="Model comparison by Macro-F1",
    )

    fig.update_layout(height=420, yaxis_title="", xaxis_title="Macro-F1")
    st.plotly_chart(fig, use_container_width=True)


def plot_light_curve(lc_object):
    if lc_object.empty:
        st.warning("No light-curve data found for this object.")
        return

    lc = lc_object.copy()

    if "band" in lc.columns:
        lc["band_display"] = lc["band"].map(normalize_band)
    else:
        lc["band_display"] = "unknown"

    lc = lc.sort_values("mjd")

    error_col = None
    for candidate in ["flux_err", "fluxerr", "flux_error"]:
        if candidate in lc.columns:
            error_col = candidate
            break

    fig = go.Figure()

    for band, group in lc.groupby("band_display"):
        if error_col is not None:
            fig.add_trace(
                go.Scatter(
                    x=group["mjd"],
                    y=group["flux"],
                    error_y=dict(
                        type="data",
                        array=group[error_col],
                        visible=True,
                    ),
                    mode="markers+lines",
                    name=str(band),
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=group["mjd"],
                    y=group["flux"],
                    mode="markers+lines",
                    name=str(band),
                )
            )

    fig.update_layout(
        title="Multiband light curve",
        xaxis_title="MJD",
        yaxis_title="Flux",
        height=480,
        legend_title="Band",
    )

    st.plotly_chart(fig, use_container_width=True)


def main():
    st.title("🔭 AstroTrust-AI Dashboard")
    st.caption(
        "Hybrid temporal-tabular classification, calibrated uncertainty, novelty, rarity, "
        "and follow-up prioritization for astronomical alerts."
    )

    if not CACHE_DIR.exists():
        st.error(
            f"Dashboard cache not found: {CACHE_DIR}\n\n"
            "Run first: python .\\app\\prepare_dashboard_data.py"
        )
        return

    class_names = load_class_names()

    perf_path = CACHE_DIR / "final_model_performance_summary.csv"
    cal_path = CACHE_DIR / "final_hybrid_calibration_summary.csv"
    policy_summary_path = CACHE_DIR / "final_followup_policy_summary.csv"
    selected_policy_path = CACHE_DIR / "final_selected_followup_policies.csv"
    lc_path = CACHE_DIR / "dashboard_test_lightcurves.parquet"

    perf = load_csv(perf_path) if perf_path.exists() else pd.DataFrame()
    cal = load_csv(cal_path) if cal_path.exists() else pd.DataFrame()
    policy_summary = load_csv(policy_summary_path) if policy_summary_path.exists() else pd.DataFrame()
    selected_policy = load_csv(selected_policy_path) if selected_policy_path.exists() else pd.DataFrame()

    st.sidebar.header("Controls")

    policy_label = st.sidebar.selectbox(
        "Follow-up policy",
        list(POLICY_FILES.keys()),
        index=0,
    )

    ranking_path = CACHE_DIR / POLICY_FILES[policy_label]

    if ranking_path.exists():
        ranking = load_csv(ranking_path)
    else:
        fallback_path = CACHE_DIR / "dashboard_scoring_table.csv"
        ranking = load_csv(fallback_path)
        if "priority_score" not in ranking.columns:
            ranking["priority_score"] = (
                0.5 * ranking.get("novelty_score", 0.0)
                + 0.5 * ranking.get("rarity_score", 0.0)
            )
            ranking = ranking.sort_values("priority_score", ascending=False).reset_index(drop=True)
            ranking["priority_rank"] = np.arange(1, len(ranking) + 1)

    ranking = add_class_names(ranking, class_names)

    st.header("1. Final AI summary")

    if not perf.empty:
        best = perf.sort_values("macro_f1", ascending=False).iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Best Accuracy", best["accuracy"])
        with c2:
            metric_card("Best Macro-F1", best["macro_f1"])
        with c3:
            metric_card("Best Balanced Acc.", best["balanced_accuracy"])
        with c4:
            st.metric("Best model", str(best["model"]))

        plot_model_performance(perf)
    else:
        st.info("Performance summary not found.")

    st.header("2. Calibration and follow-up policy")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Calibration")
        if not cal.empty:
            display_cal = cal.copy()
            st.dataframe(display_cal, use_container_width=True, hide_index=True)

            fig = px.bar(
                display_cal,
                x="model",
                y="ece",
                title="Expected Calibration Error",
                hover_data=["accuracy", "macro_f1", "brier_score", "mean_confidence"],
            )
            fig.update_layout(height=360, xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Calibration summary not found.")

    with c2:
        st.subheader("Follow-up policies")
        if not selected_policy.empty:
            st.dataframe(selected_policy, use_container_width=True, hide_index=True)

        if not policy_summary.empty:
            cols = [
                "configuration",
                "w_uncertainty",
                "w_novelty",
                "w_rarity",
                "weighted_rare_enrichment",
                "weighted_uncertainty",
                "weighted_novelty",
            ]
            cols = [c for c in cols if c in policy_summary.columns]
            st.dataframe(policy_summary[cols], use_container_width=True, hide_index=True)

            fig = px.bar(
                policy_summary.sort_values("weighted_rare_enrichment", ascending=True),
                x="weighted_rare_enrichment",
                y="configuration",
                orientation="h",
                title="Rare-class enrichment by policy",
                hover_data=[
                    c for c in ["weighted_uncertainty", "weighted_novelty"] if c in policy_summary.columns
                ],
            )
            fig.update_layout(height=360, yaxis_title="", xaxis_title="Weighted rare enrichment")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Follow-up policy summary not found.")

    st.header("3. Candidate ranking")

    st.sidebar.subheader("Ranking filters")

    top_n = st.sidebar.slider(
        "Top N candidates",
        min_value=50,
        max_value=min(5000, len(ranking)),
        value=500,
        step=50,
    )

    filtered = ranking.copy()

    if "predicted_class_name" in filtered.columns:
        predicted_classes = ["All"] + sorted(filtered["predicted_class_name"].dropna().unique().tolist())
        selected_pred_class = st.sidebar.selectbox("Predicted class", predicted_classes)

        if selected_pred_class != "All":
            filtered = filtered[filtered["predicted_class_name"] == selected_pred_class]

    correctness = st.sidebar.selectbox(
        "Prediction status",
        ["All", "Correct only", "Errors only"],
    )

    if correctness == "Correct only" and "correct" in filtered.columns:
        filtered = filtered[filtered["correct"] == True]
    elif correctness == "Errors only" and "correct" in filtered.columns:
        filtered = filtered[filtered["correct"] == False]

    if "true_is_rare" in filtered.columns:
        rare_filter = st.sidebar.selectbox(
            "True rarity",
            ["All", "Rare only", "Non-rare only"],
        )

        if rare_filter == "Rare only":
            filtered = filtered[filtered["true_is_rare"] == True]
        elif rare_filter == "Non-rare only":
            filtered = filtered[filtered["true_is_rare"] == False]

    for col, label in [
        ("confidence", "Confidence"),
        ("uncertainty_score", "Uncertainty"),
        ("novelty_score", "Novelty"),
        ("rarity_score", "Rarity"),
    ]:
        if col in filtered.columns:
            min_v = float(filtered[col].min())
            max_v = float(filtered[col].max())

            if min_v < max_v:
                lo, hi = st.sidebar.slider(
                    label,
                    min_value=min_v,
                    max_value=max_v,
                    value=(min_v, max_v),
                )
                filtered = filtered[(filtered[col] >= lo) & (filtered[col] <= hi)]

    filtered = filtered.sort_values("priority_score", ascending=False).head(top_n)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Displayed candidates", len(filtered))
    with c2:
        if "true_is_rare" in filtered.columns and len(filtered) > 0:
            st.metric("Rare true rate", f"{filtered['true_is_rare'].mean():.3f}")
    with c3:
        if "correct" in filtered.columns and len(filtered) > 0:
            st.metric("Accuracy in selection", f"{filtered['correct'].mean():.3f}")
    with c4:
        if "priority_score" in filtered.columns and len(filtered) > 0:
            st.metric("Mean priority", f"{filtered['priority_score'].mean():.3f}")

    display_cols = [
        "priority_rank",
        "object_id",
        "true_label",
        "true_class_name",
        "predicted_label",
        "predicted_class_name",
        "correct",
        "confidence",
        "uncertainty_score",
        "novelty_score",
        "rarity_score",
        "priority_score",
        "true_is_rare",
        "predicted_is_rare",
    ]

    display_cols = [c for c in display_cols if c in filtered.columns]

    st.dataframe(
        filtered[display_cols],
        use_container_width=True,
        hide_index=True,
    )

    st.header("4. Candidate inspection")

    if filtered.empty:
        st.warning("No candidates match the current filters.")
        return

    object_options = filtered["object_id"].astype(int).tolist()

    selected_object = st.selectbox(
        "Select object_id",
        object_options,
        index=0,
    )

    selected_row = filtered[filtered["object_id"].astype(int) == int(selected_object)].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Object ID", int(selected_row["object_id"]))
    with c2:
        st.metric("Predicted", selected_row.get("predicted_class_name", selected_row.get("predicted_label", "—")))
    with c3:
        st.metric("True", selected_row.get("true_class_name", selected_row.get("true_label", "—")))
    with c4:
        st.metric("Correct", str(bool(selected_row.get("correct", False))))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Confidence", selected_row.get("confidence"))
    with c2:
        metric_card("Uncertainty", selected_row.get("uncertainty_score"))
    with c3:
        metric_card("Novelty", selected_row.get("novelty_score"))
    with c4:
        metric_card("Rarity", selected_row.get("rarity_score"))

    if lc_path.exists():
        lc = load_lightcurves(lc_path)
        lc_object = lc[lc["object_id"].astype(int) == int(selected_object)]
        plot_light_curve(lc_object)
    else:
        st.warning(
            f"Light-curve cache not found: {lc_path}. "
            "Run python .\\app\\prepare_dashboard_data.py first."
        )


if __name__ == "__main__":
    main()