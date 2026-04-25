from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[2]

LIGHTCURVES_PATH = ROOT_DIR / "data" / "processed" / "elasticc2" / "lightcurves_1000obj.parquet"
RANKING_PATH = ROOT_DIR / "results" / "followup_prioritization_extra_trees" / "followup_priority_ranking.csv"
ENRICHMENT_PATH = ROOT_DIR / "results" / "followup_prioritization_extra_trees" / "followup_enrichment_vs_random.csv"


st.set_page_config(
    page_title="AstroTrust-AI",
    page_icon="🔭",
    layout="wide",
)


@st.cache_data
def load_lightcurves() -> pd.DataFrame:
    df = pd.read_parquet(LIGHTCURVES_PATH)
    df["object_id"] = df["object_id"].astype(int)
    return df


@st.cache_data
def load_ranking() -> pd.DataFrame:
    df = pd.read_csv(RANKING_PATH)
    df["object_id"] = df["object_id"].astype(int)
    return df


@st.cache_data
def load_enrichment() -> pd.DataFrame | None:
    if ENRICHMENT_PATH.exists():
        return pd.read_csv(ENRICHMENT_PATH)
    return None


def plot_lightcurve(lightcurve: pd.DataFrame, object_id: int):
    fig, ax = plt.subplots(figsize=(10, 5))

    for band, band_df in lightcurve.groupby("band"):
        band_df = band_df.sort_values("mjd")

        ax.errorbar(
            band_df["mjd"],
            band_df["flux"],
            yerr=band_df["flux_err"],
            fmt="o",
            markersize=4,
            alpha=0.8,
            label=f"band {band}",
        )

    ax.set_title(f"Light curve - object {object_id}")
    ax.set_xlabel("MJD / time")
    ax.set_ylabel("Flux")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def format_bool(value: bool) -> str:
    return "Yes" if bool(value) else "No"


def main():
    st.title("AstroTrust-AI")
    st.caption(
        "Prototype dashboard for uncertainty-aware classification, novelty detection, "
        "and follow-up prioritization in astronomical alert streams."
    )

    if not LIGHTCURVES_PATH.exists():
        st.error(f"Light-curve file not found: {LIGHTCURVES_PATH}")
        st.stop()

    if not RANKING_PATH.exists():
        st.error(f"Ranking file not found: {RANKING_PATH}")
        st.stop()

    lightcurves = load_lightcurves()
    ranking = load_ranking()
    enrichment = load_enrichment()

    st.sidebar.header("Controls")

    max_rank = int(ranking["priority_rank"].max())
    top_n = st.sidebar.slider(
        "Show top N candidates",
        min_value=5,
        max_value=max_rank,
        value=min(50, max_rank),
        step=5,
    )

    priority_filter = st.sidebar.multiselect(
        "Priority category",
        options=sorted(ranking["priority_category"].unique()),
        default=sorted(ranking["priority_category"].unique()),
    )

    only_errors = st.sidebar.checkbox("Show only incorrect predictions", value=False)
    only_rare = st.sidebar.checkbox("Show only true rare classes", value=False)

    filtered = ranking.copy()

    filtered = filtered[filtered["priority_category"].isin(priority_filter)]

    if only_errors:
        filtered = filtered[filtered["correct"] == False]

    if only_rare and "true_is_rare" in filtered.columns:
        filtered = filtered[filtered["true_is_rare"] == True]

    filtered = filtered.sort_values("priority_score", ascending=False).head(top_n)

    st.subheader("Global Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Objects in ranking", len(ranking))
    col2.metric("Mean priority", f"{ranking['priority_score'].mean():.3f}")
    col3.metric("Mean uncertainty", f"{ranking['uncertainty_score'].mean():.3f}")
    col4.metric("Mean novelty", f"{ranking['novelty_score'].mean():.3f}")

    if enrichment is not None:
        st.subheader("Follow-up Enrichment vs Random Selection")

        selected_cols = [
            "budget_fraction",
            "n_selected",
            "top_rare_true_rate",
            "random_rare_true_rate_mean",
            "rare_enrichment",
            "top_mean_uncertainty",
            "random_mean_uncertainty",
            "top_mean_novelty",
            "random_mean_novelty",
        ]

        selected_cols = [c for c in selected_cols if c in enrichment.columns]

        st.dataframe(
            enrichment[selected_cols],
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Prioritized Follow-up Candidates")

    display_cols = [
        "priority_rank",
        "object_id",
        "true_label",
        "predicted_label",
        "correct",
        "confidence",
        "uncertainty_score",
        "novelty_score",
        "rarity_score",
        "priority_score",
        "priority_category",
    ]

    display_cols = [c for c in display_cols if c in filtered.columns]

    st.dataframe(
        filtered[display_cols],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Inspect Candidate")

    if len(filtered) == 0:
        st.warning("No objects match the current filters.")
        return

    object_options = filtered["object_id"].tolist()

    selected_object = st.selectbox(
        "Select object",
        options=object_options,
        index=0,
    )

    selected_row = ranking[ranking["object_id"] == selected_object].iloc[0]
    selected_lightcurve = lightcurves[lightcurves["object_id"] == selected_object]

    left, right = st.columns([1, 2])

    with left:
        st.markdown("### Candidate information")

        st.write(f"**Object ID:** {int(selected_row['object_id'])}")
        st.write(f"**Priority rank:** {int(selected_row['priority_rank'])}")
        st.write(f"**True label:** {selected_row['true_label']}")
        st.write(f"**Predicted label:** {selected_row['predicted_label']}")
        st.write(f"**Correct prediction:** {format_bool(selected_row['correct'])}")

        st.divider()

        st.metric("Confidence", f"{selected_row['confidence']:.3f}")
        st.metric("Uncertainty", f"{selected_row['uncertainty_score']:.3f}")
        st.metric("Novelty score", f"{selected_row['novelty_score']:.3f}")
        st.metric("Rarity score", f"{selected_row['rarity_score']:.3f}")
        st.metric("Priority score", f"{selected_row['priority_score']:.3f}")

        if "true_is_rare" in selected_row:
            st.write(f"**True rare class:** {format_bool(selected_row['true_is_rare'])}")

        if "predicted_is_rare" in selected_row:
            st.write(f"**Predicted rare class:** {format_bool(selected_row['predicted_is_rare'])}")

    with right:
        st.markdown("### Light curve")

        if len(selected_lightcurve) == 0:
            st.warning("No light-curve points found for this object.")
        else:
            fig = plot_lightcurve(selected_lightcurve, selected_object)
            st.pyplot(fig)

            st.markdown("### Photometric points")
            st.dataframe(
                selected_lightcurve.sort_values("mjd"),
                use_container_width=True,
                hide_index=True,
            )


if __name__ == "__main__":
    main()