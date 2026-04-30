from pathlib import Path
import json
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

try:
    from astrotrust_data_adapter import (
        read_and_normalize_from_uploaded,
        read_and_normalize_from_path,
        read_and_normalize_from_url,
        summarize_objects as adapter_summarize_objects,
    )
    HAS_DATA_ADAPTER = True
    DATA_ADAPTER_IMPORT_ERROR = None
except Exception as exc:
    HAS_DATA_ADAPTER = False
    DATA_ADAPTER_IMPORT_ERROR = exc

try:
    from snana_elasticc_utils import (
        load_snana_pair_from_uploads,
        load_snana_pair_from_paths,
    )
    HAS_SNANA_UTILS = True
    SNANA_UTILS_IMPORT_ERROR = None
except Exception as exc:
    HAS_SNANA_UTILS = False
    SNANA_UTILS_IMPORT_ERROR = exc

try:
    from astrotrust_inference import get_inference_engine
    HAS_ASTROTRUST_INFERENCE = True
    ASTROTRUST_INFERENCE_IMPORT_ERROR = None
except Exception as exc:
    HAS_ASTROTRUST_INFERENCE = False
    ASTROTRUST_INFERENCE_IMPORT_ERROR = exc

try:
    from astrotrust_feature_builder import build_v4_feature_row_auto
    HAS_FEATURE_BUILDER = True
    FEATURE_BUILDER_IMPORT_ERROR = None
except Exception as exc:
    HAS_FEATURE_BUILDER = False
    FEATURE_BUILDER_IMPORT_ERROR = exc


from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

try:
    from astropy.io import fits
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False


ROOT_DIR = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT_DIR / "data" / "processed" / "elasticc2_large" / "dashboard_cache"

POLICY_FILES = {
    "Novelty + rarity": "hybrid_test_ranking_novelty_rarity.csv",
    "Rarity only": "hybrid_test_ranking_rarity_only.csv",
    "Previous discovery": "hybrid_test_ranking_previous_discovery.csv",
    "Fixed discovery": "hybrid_test_ranking_fixed_discovery.csv",
}

BAND_MAP = {
    "0": "u", "1": "g", "2": "r", "3": "i", "4": "z", "5": "Y",
    0: "u", 1: "g", 2: "r", 3: "i", 4: "z", 5: "Y",
}

st.set_page_config(
    page_title="AstroTrust-AI",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
    .main {
        background: linear-gradient(135deg, #070B18 0%, #10172A 45%, #111827 100%);
        color: #E5E7EB;
    }

    [data-testid="stSidebar"] {
        background: #08111F;
        border-right: 1px solid rgba(148, 163, 184, 0.18);
    }

    [data-testid="stHeader"] {
        background: rgba(7, 11, 24, 0.72);
        backdrop-filter: blur(10px);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    .hero {
        padding: 1.6rem 1.8rem;
        border-radius: 28px;
        background: radial-gradient(circle at top left, rgba(56, 189, 248, 0.25), transparent 32%),
                    linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(30, 41, 59, 0.86));
        border: 1px solid rgba(148, 163, 184, 0.18);
        box-shadow: 0 24px 80px rgba(0, 0, 0, 0.30);
        margin-bottom: 1rem;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.2rem;
        letter-spacing: -0.04em;
        color: #F8FAFC;
    }

    .hero p {
        margin-top: 0.55rem;
        margin-bottom: 0;
        color: #CBD5E1;
        font-size: 1.02rem;
        line-height: 1.55;
    }

    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 1rem;
    }

    .chip {
        padding: 0.35rem 0.68rem;
        border-radius: 999px;
        background: rgba(14, 165, 233, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.25);
        color: #BAE6FD;
        font-size: 0.82rem;
    }

    .section-card {
        padding: 1.2rem;
        border-radius: 24px;
        background: rgba(15, 23, 42, 0.78);
        border: 1px solid rgba(148, 163, 184, 0.15);
        box-shadow: 0 18px 50px rgba(0, 0, 0, 0.18);
        margin-bottom: 1rem;
    }

    .small-caption {
        color: #94A3B8;
        font-size: 0.88rem;
        margin-top: -0.25rem;
        margin-bottom: 0.75rem;
    }

    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.84);
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 22px;
        padding: 1rem 1.1rem;
        box-shadow: 0 10px 35px rgba(0, 0, 0, 0.18);
    }

    div[data-testid="stMetric"] label {
        color: #CBD5E1 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #F8FAFC !important;
    }

    .stDataFrame {
        border-radius: 18px;
        overflow: hidden;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.35rem;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 999px;
        padding: 0.55rem 1.0rem;
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(148, 163, 184, 0.14);
        color: #CBD5E1;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.28), rgba(99, 102, 241, 0.22));
        color: #F8FAFC;
    }

    .status-good {
        color: #86EFAC;
        font-weight: 600;
    }

    .status-warn {
        color: #FDE68A;
        font-weight: 600;
    }

    .status-bad {
        color: #FCA5A5;
        font-weight: 600;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_lightcurves(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def normalize_band(value):
    if isinstance(value, bytes):
        value = value.decode(errors="ignore").strip()

    if value in BAND_MAP:
        return BAND_MAP[value]

    value_str = str(value).strip()

    if value_str.startswith("b'") and value_str.endswith("'"):
        value_str = value_str[2:-1]

    if value_str.startswith('b"') and value_str.endswith('"'):
        value_str = value_str[2:-1]

    aliases = {
        "zg": "g",
        "zr": "r",
        "zi": "i",
        "ztfg": "g",
        "ztfr": "r",
        "ztfi": "i",
        "ZTF_g": "g",
        "ZTF_r": "r",
        "ZTF_i": "i",
        "g_ZTF": "g",
        "r_ZTF": "r",
        "i_ZTF": "i",
        "sdssg": "g",
        "sdssr": "r",
        "sdssi": "i",
        "sdssz": "z",
        "u": "u",
        "g": "g",
        "r": "r",
        "i": "i",
        "z": "z",
        "y": "Y",
        "Y": "Y",
        "B": "B",
        "V": "V",
        "R": "r",
        "I": "i",
        "Rc": "r",
        "Ic": "i",
    }

    if value_str in aliases:
        return aliases[value_str]

    if value_str.lower() in aliases:
        return aliases[value_str.lower()]

    if value_str in BAND_MAP:
        return BAND_MAP[value_str]

    if value_str.lower() == "y":
        return "Y"

    return value_str


def safe_metric(label, value, fmt="{:.4f}"):
    if value is None or pd.isna(value):
        st.metric(label, "—")
    elif isinstance(value, (float, np.floating)):
        st.metric(label, fmt.format(float(value)))
    else:
        st.metric(label, value)


def load_class_names():
    path = CACHE_DIR / "full_class_counts.csv"
    if not path.exists():
        return {}
    df = load_csv(path)
    if {"label", "class_name"}.issubset(df.columns):
        return dict(zip(df["label"].astype(int), df["class_name"].astype(str)))
    return {}


def add_class_names(df: pd.DataFrame, class_names: dict) -> pd.DataFrame:
    out = df.copy()
    if "true_label" in out.columns:
        out["true_class_name"] = out["true_label"].map(lambda x: class_names.get(int(x), str(x)))
    if "predicted_label" in out.columns:
        out["predicted_class_name"] = out["predicted_label"].map(lambda x: class_names.get(int(x), str(x)))
    return out


def load_dashboard_assets():
    assets = {}
    filenames = {
        "performance": "final_model_performance_summary.csv",
        "calibration": "final_hybrid_calibration_summary.csv",
        "policy_summary": "final_followup_policy_summary.csv",
        "selected_policies": "final_selected_followup_policies.csv",
        "scoring": "dashboard_scoring_table.csv",
    }
    for key, filename in filenames.items():
        path = CACHE_DIR / filename
        assets[key] = load_csv(path) if path.exists() else pd.DataFrame()
    return assets


def load_ranking(policy_label: str) -> pd.DataFrame:
    ranking_path = CACHE_DIR / POLICY_FILES[policy_label]
    if ranking_path.exists():
        ranking = load_csv(ranking_path)
    else:
        fallback = CACHE_DIR / "dashboard_scoring_table.csv"
        ranking = load_csv(fallback) if fallback.exists() else pd.DataFrame()
        if not ranking.empty and "priority_score" not in ranking.columns:
            ranking["priority_score"] = 0.5 * ranking.get("novelty_score", 0.0) + 0.5 * ranking.get("rarity_score", 0.0)
            ranking = ranking.sort_values("priority_score", ascending=False).reset_index(drop=True)
            ranking["priority_rank"] = np.arange(1, len(ranking) + 1)
    return ranking


def plot_model_performance(perf: pd.DataFrame):
    if perf.empty or "macro_f1" not in perf.columns:
        st.info("Model performance summary is not available.")
        return

    plot_df = perf.sort_values("macro_f1", ascending=True)
    fig = px.bar(
        plot_df,
        x="macro_f1",
        y="model",
        orientation="h",
        color="family" if "family" in plot_df.columns else None,
        hover_data=[c for c in ["accuracy", "balanced_accuracy", "weighted_f1", "dataset"] if c in plot_df.columns],
        title="Model comparison by Macro-F1",
    )
    fig.update_layout(
        height=460,
        yaxis_title="",
        xaxis_title="Macro-F1",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E5E7EB"),
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_calibration(cal: pd.DataFrame):
    if cal.empty or "ece" not in cal.columns:
        st.info("Calibration summary is not available.")
        return

    fig = px.bar(
        cal,
        x="model",
        y="ece",
        hover_data=[c for c in ["accuracy", "macro_f1", "brier_score", "mean_confidence"] if c in cal.columns],
        title="Expected Calibration Error (lower is better)",
    )
    fig.update_layout(
        height=380,
        xaxis_title="",
        yaxis_title="ECE",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E5E7EB"),
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_policy_enrichment(policy_summary: pd.DataFrame):
    if policy_summary.empty or "weighted_rare_enrichment" not in policy_summary.columns:
        st.info("Follow-up policy summary is not available.")
        return

    fig = px.bar(
        policy_summary.sort_values("weighted_rare_enrichment", ascending=True),
        x="weighted_rare_enrichment",
        y="configuration",
        orientation="h",
        hover_data=[c for c in ["weighted_uncertainty", "weighted_novelty", "w_uncertainty", "w_novelty", "w_rarity"] if c in policy_summary.columns],
        title="Rare-class enrichment by follow-up policy",
    )
    fig.update_layout(
        height=380,
        yaxis_title="",
        xaxis_title="Weighted rare enrichment",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E5E7EB"),
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_light_curve(lc_object: pd.DataFrame):
    if lc_object.empty:
        st.warning("No light-curve data found for this object.")
        return

    lc = lc_object.copy()
    lc["band_display"] = lc["band"].map(normalize_band) if "band" in lc.columns else "unknown"
    lc = lc.sort_values("mjd") if "mjd" in lc.columns else lc

    error_col = next((c for c in ["flux_err", "fluxerr", "flux_error"] if c in lc.columns), None)

    fig = go.Figure()
    for band, group in lc.groupby("band_display"):
        if error_col:
            fig.add_trace(
                go.Scatter(
                    x=group["mjd"],
                    y=group["flux"],
                    error_y=dict(type="data", array=group[error_col], visible=True),
                    mode="markers+lines",
                    name=str(band),
                )
            )
        else:
            fig.add_trace(
                go.Scatter(x=group["mjd"], y=group["flux"], mode="markers+lines", name=str(band))
            )

    fig.update_layout(
        title="Multiband light curve",
        xaxis_title="MJD",
        yaxis_title="Flux",
        height=520,
        legend_title="Band",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.42)",
        font=dict(color="#E5E7EB"),
    )
    st.plotly_chart(fig, use_container_width=True)


def summarize_fits_hdus(hdul):
    rows = []
    for i, hdu in enumerate(hdul):
        data = hdu.data
        header = hdu.header
        shape = None if data is None else tuple(data.shape)
        rows.append({
            "index": i,
            "name": hdu.name,
            "type": type(hdu).__name__,
            "shape": str(shape),
            "naxis": header.get("NAXIS", None),
            "object": header.get("OBJECT", ""),
            "instrument": header.get("INSTRUME", ""),
            "bunit": header.get("BUNIT", ""),
        })
    return pd.DataFrame(rows)


def robust_image_scale(image):
    arr = np.asarray(image, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr, 0.0, 1.0
    vmin, vmax = np.nanpercentile(finite, [1, 99])
    if np.isclose(vmin, vmax):
        vmin, vmax = np.nanmin(finite), np.nanmax(finite)
    return arr, vmin, vmax



@st.cache_data(show_spinner=False)
def scan_local_files(folder_str: str, suffixes: tuple, recursive: bool, max_files: int) -> pd.DataFrame:
    folder = Path(folder_str).expanduser()

    if not folder.exists() or not folder.is_dir():
        return pd.DataFrame(columns=["path", "name", "folder", "size_mb", "modified"])

    files = []
    for suffix in suffixes:
        pattern = f"**/*{suffix}" if recursive else f"*{suffix}"
        files.extend(folder.glob(pattern))

    rows = []
    for p in files:
        try:
            stat = p.stat()
            rows.append({
                "path": str(p),
                "name": p.name,
                "folder": str(p.parent),
                "size_mb": stat.st_size / (1024 * 1024),
                "modified": stat.st_mtime,
            })
        except OSError:
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["path", "name", "folder", "size_mb", "modified"])

    df = df.sort_values("modified", ascending=False).head(max_files).reset_index(drop=True)
    return df


def native_file_picker_dialog(suffixes, initial_dir):
    """Open a native OS file picker and return the selected local path.

    This works when Streamlit is running on the same desktop machine as the user.
    It is not suitable for remote/cloud deployments because the dialog opens on the server.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        raise RuntimeError(f"tkinter is not available: {exc}")

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    filetypes = [
        ("Compatible files", " ".join([f"*{s}" for s in suffixes])),
        ("All files", "*.*"),
    ]

    selected = filedialog.askopenfilename(
        title="Select local file",
        initialdir=str(initial_dir),
        filetypes=filetypes,
    )

    root.destroy()

    if not selected:
        return None

    return Path(selected)


def interactive_local_file_picker(label: str, suffixes, default_dir: str, key_prefix: str):
    """User-friendly local file selection for large files.

    Browser apps cannot use st.file_uploader to return the full local path for security reasons.
    In local desktop mode, we can open a native OS file picker with tkinter and store the selected path.
    """
    default_dir_path = Path(default_dir).expanduser()

    selection_mode = st.radio(
        "Local selection mode",
        ["Native file picker", "Browse folder", "Paste full path"],
        horizontal=True,
        key=f"{key_prefix}_selection_mode",
    )

    session_key = f"{key_prefix}_selected_native_path"

    if selection_mode == "Native file picker":
        c1, c2 = st.columns([1, 3])
        with c1:
            open_clicked = st.button("Choose file…", key=f"{key_prefix}_open_native_dialog", use_container_width=True)
        with c2:
            current = st.session_state.get(session_key, "")
            st.text_input("Selected file", value=current, key=f"{key_prefix}_native_path_display", disabled=True)

        if open_clicked:
            try:
                selected = native_file_picker_dialog(suffixes=suffixes, initial_dir=default_dir_path)
                if selected is not None:
                    st.session_state[session_key] = str(selected)
                    st.rerun()
            except Exception as exc:
                st.error(f"Could not open native file picker: {exc}")
                st.info("Use 'Paste full path' as a fallback.")
                return None

        selected_path = st.session_state.get(session_key)
        if not selected_path:
            st.info("Click **Choose file…** to open the native file selection window.")
            return None

        return Path(selected_path)

    if selection_mode == "Paste full path":
        local_path = st.text_input(
            label,
            placeholder="C:/Users/wesle/Desktop/dados/example.fits",
            key=f"{key_prefix}_direct_path",
        )
        if not local_path:
            return None
        return Path(local_path.strip().strip('"')).expanduser()

    folder = st.text_input(
        "Folder to browse",
        value=default_dir,
        key=f"{key_prefix}_folder",
        help="Choose a folder that contains the files. Keep recursive search off for very large directories.",
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        recursive = st.checkbox("Search subfolders", value=False, key=f"{key_prefix}_recursive")
    with c2:
        max_files = st.number_input("Max files to list", min_value=10, max_value=5000, value=300, step=10, key=f"{key_prefix}_max_files")

    if not folder:
        return None

    folder_path = Path(folder).expanduser()
    if not folder_path.exists() or not folder_path.is_dir():
        st.error(f"Folder not found: {folder_path}")
        return None

    with st.spinner("Scanning folder..."):
        files_df = scan_local_files(str(folder_path), tuple(suffixes), bool(recursive), int(max_files))

    if files_df.empty:
        st.warning(f"No compatible files found in: {folder_path}")
        return None

    def fmt(path_str):
        row = files_df[files_df["path"] == path_str].iloc[0]
        return f"{row['name']}  —  {row['size_mb']:.1f} MB  —  {row['folder']}"

    selected = st.selectbox(
        "Select file",
        files_df["path"].tolist(),
        format_func=fmt,
        key=f"{key_prefix}_selected_file",
    )

    with st.expander("Files found", expanded=False):
        preview = files_df.copy()
        preview["size_mb"] = preview["size_mb"].map(lambda x: f"{x:.1f}")
        st.dataframe(preview[["name", "size_mb", "folder", "path"]], use_container_width=True, hide_index=True)

    return Path(selected)


def show_fits_viewer():
    st.markdown("### Advanced Data Inspector")
    st.caption(
        "Inspect generic astronomical FITS files, including images, tables, and datacubes. "
        "For multi-GB FITS files, prefer local path mode instead of browser upload."
    )

    if not HAS_ASTROPY:
        st.error("Astropy is not installed. Run: pip install astropy")
        return

    input_mode = st.radio(
        "Input mode",
        ["Browser upload", "Local file path"],
        horizontal=True,
        key="fits_input_mode",
    )

    tmp_path = None
    fits_path = None

    if input_mode == "Browser upload":
        uploaded = st.file_uploader("Upload a FITS file", type=["fits", "fit", "fts"], key="fits_upload")
        if uploaded is None:
            st.info("Upload a FITS file to inspect headers, images, tables, or datacube slices.")
            return

        with tempfile.NamedTemporaryFile(delete=False, suffix=".fits") as tmp:
            tmp.write(uploaded.read())
            tmp_path = Path(tmp.name)
            fits_path = tmp_path

    else:
        fits_path = interactive_local_file_picker(
            label="Local FITS path",
            suffixes=[".fits", ".fit", ".fts"],
            default_dir=str(Path.home()),
            key_prefix="fits_local",
        )

        if fits_path is None:
            st.info("Browse a folder or paste the full local path of a FITS file. This is recommended for multi-GB datacubes.")
            return

        if not fits_path.exists():
            st.error(f"File not found: {fits_path}")
            return

        if fits_path.suffix.lower() not in [".fits", ".fit", ".fts"]:
            st.warning("The file extension is not a standard FITS extension, but the viewer will still try to open it.")

    try:
        with fits.open(fits_path, memmap=True) as hdul:
            summary = summarize_fits_hdus(hdul)
            st.dataframe(summary, use_container_width=True, hide_index=True)

            hdu_index = st.selectbox(
                "Select HDU",
                summary["index"].tolist(),
                format_func=lambda i: f"{i}: {summary.loc[summary['index'] == i, 'name'].iloc[0]} | {summary.loc[summary['index'] == i, 'shape'].iloc[0]}",
            )

            hdu = hdul[int(hdu_index)]
            data = hdu.data
            header = hdu.header

            with st.expander("Header preview", expanded=False):
                header_rows = [{"keyword": k, "value": str(v)} for k, v in list(header.items())[:250]]
                st.dataframe(pd.DataFrame(header_rows), use_container_width=True, hide_index=True)

            if data is None:
                st.warning("Selected HDU has no data.")
                return

            if hasattr(data, "columns"):
                max_rows = st.slider("Rows to preview", 50, 5000, 500, step=50, key="fits_table_preview_rows")
                arr = np.array(data[:max_rows])
                arr = arr.byteswap().view(arr.dtype.newbyteorder())
                table_df = pd.DataFrame(arr)
                st.markdown("#### FITS table preview")
                st.dataframe(table_df, use_container_width=True)
                st.info(
                    "If this table contains columns like object_id, mjd, band, flux, and flux_err, "
                    "use the Upload Alert / Light Curve tab to validate it as a potential AstroTrust-AI input."
                )
                return

            arr = np.asarray(data)
            st.markdown(f"#### Data shape: `{arr.shape}`")

            if arr.ndim == 2:
                downsample = st.slider("Display downsample factor", 1, 20, 1, key="fits_2d_downsample")
                img_data = arr[::downsample, ::downsample]
                img, vmin, vmax = robust_image_scale(img_data)
                fig = px.imshow(img, zmin=vmin, zmax=vmax, color_continuous_scale="Viridis", title="FITS image")
                fig.update_layout(height=620, paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#E5E7EB"))
                st.plotly_chart(fig, use_container_width=True)

            elif arr.ndim == 3:
                axis = st.selectbox("Datacube slicing axis", [0, 1, 2], index=0)
                max_slice = arr.shape[axis] - 1
                slice_idx = st.slider("Slice index", 0, int(max_slice), int(max_slice // 2))
                downsample = st.slider("Display downsample factor", 1, 20, 2, key="fits_cube_downsample")

                if axis == 0:
                    image = arr[slice_idx, ::downsample, ::downsample]
                elif axis == 1:
                    image = arr[::downsample, slice_idx, ::downsample]
                else:
                    image = arr[::downsample, ::downsample, slice_idx]

                img, vmin, vmax = robust_image_scale(image)
                fig = px.imshow(
                    img,
                    zmin=vmin,
                    zmax=vmax,
                    color_continuous_scale="Viridis",
                    title=f"Datacube slice axis={axis}, index={slice_idx}",
                )
                fig.update_layout(height=620, paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#E5E7EB"))
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("#### Quick spectrum extraction")
                st.caption("Optimized for cubes where axis 0 is spectral/time-like and axes 1-2 are spatial.")

                if arr.shape[1] > 1 and arr.shape[2] > 1:
                    c1, c2 = st.columns(2)
                    with c1:
                        y_pix = st.number_input("Y pixel", min_value=0, max_value=int(arr.shape[1] - 1), value=int(arr.shape[1] // 2))
                    with c2:
                        x_pix = st.number_input("X pixel", min_value=0, max_value=int(arr.shape[2] - 1), value=int(arr.shape[2] // 2))

                    spectrum = arr[:, int(y_pix), int(x_pix)]
                    fig_spec = go.Figure(go.Scatter(y=spectrum, mode="lines"))
                    fig_spec.update_layout(
                        title="Extracted spectrum / cube profile",
                        xaxis_title="Slice",
                        yaxis_title="Value",
                        height=320,
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#E5E7EB"),
                    )
                    st.plotly_chart(fig_spec, use_container_width=True)

                st.warning(
                    "This looks like an image/datacube FITS. AstroTrust-AI can visualize it, but the current classifier expects "
                    "light-curve/alert-like data, not spectral datacubes."
                )

            else:
                st.warning("This FITS data has dimensionality not yet supported by the viewer.")

    except Exception as exc:
        st.error(f"Could not open FITS file: {exc}")

    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


def read_lightcurve_file_from_url(url: str):
    """Read a generic light-curve table directly from a public URL."""
    url = str(url).strip()

    if not url:
        raise ValueError("Empty URL.")

    lower_url = url.lower()

    if "format=csv" in lower_url or lower_url.endswith(".csv"):
        return pd.read_csv(url), "csv_url"

    if lower_url.endswith(".parquet"):
        return pd.read_parquet(url), "parquet_url"

    # Many astronomy APIs return CSV even when the URL does not end with .csv.
    # Try CSV as a practical default.
    try:
        return pd.read_csv(url), "csv_url"
    except Exception as exc:
        raise ValueError(
            "Could not read URL as CSV/Parquet. Make sure the URL points to a public table file "
            "or an API response with FORMAT=CSV."
        ) from exc

def read_uploaded_lightcurve_file(uploaded):
    """Read CSV, Parquet, or FITS table uploaded by the user."""
    suffix = Path(uploaded.name).suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(uploaded), "csv"

    if suffix == ".parquet":
        return pd.read_parquet(uploaded), "parquet"

    if suffix in [".fits", ".fit", ".fts"]:
        if not HAS_ASTROPY:
            raise RuntimeError("Astropy is required to read FITS files. Install with: pip install astropy")

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = Path(tmp.name)

        try:
            return read_lightcurve_file_from_path(tmp_path)
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    raise ValueError(f"Unsupported file type: {suffix}")


def read_lightcurve_file_from_path(path):
    """Read CSV, Parquet, or FITS table from a local path."""
    path = Path(str(path).strip().strip('"'))
    suffix = path.suffix.lower()

    if not path.exists():
        raise FileNotFoundError(path)

    if suffix == ".csv":
        return pd.read_csv(path), "csv_local"

    if suffix == ".parquet":
        return pd.read_parquet(path), "parquet_local"

    if suffix in [".fits", ".fit", ".fts"]:
        if not HAS_ASTROPY:
            raise RuntimeError("Astropy is required to read FITS files. Install with: pip install astropy")

        with fits.open(path, memmap=True) as hdul:
            table_hdus = []
            for idx, hdu in enumerate(hdul):
                if getattr(hdu, "data", None) is not None and hasattr(hdu.data, "columns"):
                    table_hdus.append((idx, hdu))

            if not table_hdus:
                raise ValueError(
                    "No FITS table HDU was found. Image/datacube FITS files can be inspected in the Advanced Data Inspector, "
                    "but they are not direct light-curve inputs."
                )

            selected_idx, selected_hdu = table_hdus[0]
            for idx, hdu in table_hdus:
                if str(hdu.name).upper() in ["PHOT", "PHOTOMETRY", "LIGHTCURVE", "LC"]:
                    selected_idx, selected_hdu = idx, hdu
                    break

            arr = np.array(selected_hdu.data)
            arr = arr.byteswap().view(arr.dtype.newbyteorder())
            table = pd.DataFrame(arr)

            return table, f"fits_table_hdu_{selected_idx}_{selected_hdu.name}_local"

    raise ValueError(f"Unsupported file type: {suffix}")



def infer_lightcurve_columns(df: pd.DataFrame):
    lower_map = {str(c).lower(): c for c in df.columns}

    candidates = {
        "object_id": [
            "object_id",
            "objectid",
            "objid",
            "diaobjectid",
            "diaobjectId",
            "snid",
            "id",
            "source_id",
            "oid",
            "name",
            "target_name",
        ],
        "mjd": [
            "mjd",
            "mjdobs",
            "time",
            "t",
            "jd",
            "hjd",
            "bjd",
            "date",
            "obsjd",
            "obsmjd",
        ],
        "band": [
            "band",
            "filter",
            "passband",
            "fid",
            "flt",
            "filtercode",
            "filter_code",
            "filtername",
            "bandname",
        ],
        "flux": [
            "flux",
            "fluxcal",
            "flx",
            "psflux",
            "forcediffimflux",
            "flux_jy",
            "fnu",
            "flux_density",
        ],
        "flux_err": [
            "flux_err",
            "fluxerr",
            "flux_error",
            "fluxcalerr",
            "psfluxerr",
            "forcediffimfluxunc",
            "flux_unc",
            "flux_uncertainty",
            "fluxerr_jy",
        ],
        "mag": [
            "mag",
            "magnitude",
            "magpsf",
            "psfmag",
            "mag_auto",
            "magap",
            "mag_calibrated",
            "magnitude_calibrated",
        ],
        "mag_err": [
            "magerr",
            "mag_err",
            "mag_error",
            "sigmapsf",
            "e_mag",
            "mag_unc",
            "magnitude_error",
            "uncertainty",
        ],
    }

    inferred = {}

    for target, names in candidates.items():
        inferred[target] = None

        for name in names:
            if str(name).lower() in lower_map:
                inferred[target] = lower_map[str(name).lower()]
                break

    return inferred


def _convert_magnitude_to_relative_flux(mag, mag_err=None):
    """Convert magnitude to relative flux.

    The zero point is set from the median magnitude in the uploaded table.
    This keeps the scale stable even when the real photometric zero point is unknown.
    """
    mag = pd.to_numeric(mag, errors="coerce")
    finite_mag = mag[np.isfinite(mag)]

    if len(finite_mag) == 0:
        flux = pd.Series(np.nan, index=mag.index)
        flux_err = pd.Series(np.nan, index=mag.index)
        return flux, flux_err

    mag0 = float(np.nanmedian(finite_mag))
    flux = 10.0 ** (-0.4 * (mag - mag0))

    if mag_err is None:
        flux_err = pd.Series(np.nan, index=mag.index)
    else:
        mag_err = pd.to_numeric(mag_err, errors="coerce")
        flux_err = flux * 0.4 * np.log(10.0) * mag_err

    return flux, flux_err


def normalize_uploaded_lightcurve(df: pd.DataFrame, inferred: dict):
    issues = []

    has_flux = inferred.get("flux") is not None
    has_mag = inferred.get("mag") is not None

    missing = []

    if inferred.get("mjd") is None:
        missing.append("mjd/time")

    if inferred.get("band") is None:
        missing.append("band/filter")

    if not has_flux and not has_mag:
        missing.append("flux or mag")

    if missing:
        issues.append(f"Missing required light-curve columns: {', '.join(missing)}")
        return pd.DataFrame(), issues

    out = pd.DataFrame(index=df.index)

    if inferred.get("object_id") is not None:
        out["object_id"] = df[inferred["object_id"]].astype(str)
    else:
        out["object_id"] = "single_object"
        issues.append("No object_id column found; treating the file as a single object.")

    out["mjd"] = pd.to_numeric(df[inferred["mjd"]], errors="coerce")
    out["band"] = df[inferred["band"]].map(normalize_band)

    if has_flux:
        out["flux"] = pd.to_numeric(df[inferred["flux"]], errors="coerce")

        if inferred.get("flux_err") is not None:
            out["flux_err"] = pd.to_numeric(df[inferred["flux_err"]], errors="coerce")
        else:
            out["flux_err"] = np.nan
            issues.append(
                "No flux uncertainty column found; plotting will work, but SNR/tensor construction will be incomplete."
            )

    else:
        mag = pd.to_numeric(df[inferred["mag"]], errors="coerce")

        if inferred.get("mag_err") is not None:
            mag_err = pd.to_numeric(df[inferred["mag_err"]], errors="coerce")
        else:
            mag_err = None

        flux, flux_err = _convert_magnitude_to_relative_flux(mag, mag_err)

        out["flux"] = flux
        out["flux_err"] = flux_err
        out["mag"] = mag

        if mag_err is not None:
            out["mag_err"] = mag_err

        issues.append(
            "Input used magnitude columns. Converted mag/magerr to relative flux/flux_err automatically."
        )

    if "flux_err" in out.columns:
        out["flux_err"] = pd.to_numeric(out["flux_err"], errors="coerce")
        out.loc[out["flux_err"] <= 0, "flux_err"] = np.nan

    # Preserve useful numeric context columns from real survey tables, e.g. IRSA/ZTF.
    # These can later become features such as ra_mean, dec_mean, catflags_mean, etc.
    used_columns = {
        inferred.get("object_id"),
        inferred.get("mjd"),
        inferred.get("band"),
        inferred.get("flux"),
        inferred.get("flux_err"),
        inferred.get("mag"),
        inferred.get("mag_err"),
    }
    used_columns = {c for c in used_columns if c is not None}

    for col in df.columns:
        if col in used_columns:
            continue

        col_lower = str(col).lower()

        # Keep common positional/context/quality columns.
        keep = (
            col_lower in [
                "ra",
                "dec",
                "catflags",
                "clrcoeff",
                "field",
                "ccdid",
                "qid",
                "airmass",
                "seeing",
                "fwhm",
                "chi",
                "sharp",
            ]
            or "ra" == col_lower
            or "dec" == col_lower
            or "flag" in col_lower
            or "quality" in col_lower
        )

        if keep:
            numeric_col = pd.to_numeric(df[col], errors="coerce")
            if numeric_col.notna().any():
                out[col_lower] = numeric_col

    before = len(out)
    out = out.dropna(subset=["mjd", "flux"]).reset_index(drop=True)
    dropped = before - len(out)

    if dropped > 0:
        issues.append(f"Dropped {dropped} rows with invalid mjd/flux values.")

    return out, issues



def plot_uploaded_lightcurve(lc: pd.DataFrame, title="Uploaded light curve"):
    if lc.empty:
        st.warning("No valid light-curve rows to plot.")
        return

    fig = go.Figure()
    lc = lc.sort_values("mjd")

    for band, group in lc.groupby("band"):
        if "flux_err" in group.columns and group["flux_err"].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=group["mjd"],
                    y=group["flux"],
                    error_y=dict(type="data", array=group["flux_err"], visible=True),
                    mode="markers+lines",
                    name=str(band),
                )
            )
        else:
            fig.add_trace(go.Scatter(x=group["mjd"], y=group["flux"], mode="markers+lines", name=str(band)))

    fig.update_layout(
        title=title,
        xaxis_title="MJD / time",
        yaxis_title="Flux",
        height=520,
        legend_title="Band",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.42)",
        font=dict(color="#E5E7EB"),
    )
    st.plotly_chart(fig, use_container_width=True)


def summarize_uploaded_objects(lc: pd.DataFrame) -> pd.DataFrame:
    """Rank objects by usefulness for prediction.

    Best object = more bands, then more observations, then longer time span.
    """
    rows = []

    for object_id, group in lc.groupby("object_id"):
        n_obs = len(group)
        n_bands = group["band"].nunique()

        if "mjd" in group.columns and len(group) > 1:
            time_span = float(group["mjd"].max() - group["mjd"].min())
        else:
            time_span = 0.0

        rows.append(
            {
                "object_id": str(object_id),
                "n_obs": int(n_obs),
                "n_bands": int(n_bands),
                "time_span": time_span,
            }
        )

    summary = pd.DataFrame(rows)

    if summary.empty:
        return summary

    summary = summary.sort_values(
        ["n_bands", "n_obs", "time_span"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return summary


def compute_prediction_reliability(
    lc_obj: pd.DataFrame,
    auto_feature_report=None,
    source_domain="generic",
    prediction_result=None,
):
    """Separate input quality from model-domain reliability.

    Input quality describes whether the uploaded light curve is technically useful.
    Model-domain reliability describes whether the trained model should be trusted
    for this source domain and feature coverage.
    """
    n_obs = len(lc_obj)
    n_bands = lc_obj["band"].nunique() if "band" in lc_obj.columns else 0

    if "mjd" in lc_obj.columns and len(lc_obj) > 1:
        time_span = float(lc_obj["mjd"].max() - lc_obj["mjd"].min())
    else:
        time_span = 0.0

    has_flux_err = (
        "flux_err" in lc_obj.columns
        and lc_obj["flux_err"].notna().any()
    )

    # ----------------------------
    # 1. Input quality
    # ----------------------------
    input_score = 3
    input_reasons = []

    if n_obs < 10:
        input_score = min(input_score, 1)
        input_reasons.append("Very few observations.")
    elif n_obs < 30:
        input_score = min(input_score, 2)
        input_reasons.append("Limited number of observations.")

    if n_bands < 1:
        input_score = min(input_score, 1)
        input_reasons.append("No valid photometric band detected.")
    elif n_bands == 1:
        input_score = min(input_score, 2)
        input_reasons.append("Single-band input; multiband coverage is preferable.")

    if time_span <= 0:
        input_score = min(input_score, 1)
        input_reasons.append("Invalid or zero time span.")

    if not has_flux_err:
        input_score = min(input_score, 2)
        input_reasons.append("No valid flux uncertainty available.")

    if input_score >= 3:
        input_level = "Good"
    elif input_score == 2:
        input_level = "Limited"
    else:
        input_level = "Low"

    if not input_reasons:
        input_reasons.append("Input has enough observations, valid time coverage, and usable photometric uncertainty.")

    # ----------------------------
    # 2. Feature completeness
    # ----------------------------
    feature_info = {
        "n_expected_features": None,
        "n_matched_features": None,
        "n_missing_filled_zero": None,
        "matched_fraction": None,
        "level": "Unknown",
        "reasons": [],
    }

    missing_frac = None

    if auto_feature_report is not None:
        n_expected = auto_feature_report.get("n_expected_features", 0)
        n_matched = auto_feature_report.get("n_matched_features", 0)
        n_missing = auto_feature_report.get("n_missing_filled_zero", 0)

        if n_expected > 0:
            matched_fraction = n_matched / n_expected
            missing_frac = n_missing / n_expected

            if matched_fraction >= 0.85:
                feature_level = "Good"
            elif matched_fraction >= 0.60:
                feature_level = "Limited"
            else:
                feature_level = "Low"

            feature_reasons = [
                f"{n_matched}/{n_expected} features were available or derived.",
                f"{n_missing}/{n_expected} features were filled with zero as model-compatible placeholders.",
            ]

            feature_info = {
                "n_expected_features": int(n_expected),
                "n_matched_features": int(n_matched),
                "n_missing_filled_zero": int(n_missing),
                "matched_fraction": float(matched_fraction),
                "level": feature_level,
                "reasons": feature_reasons,
            }

    # ----------------------------
    # 3. Model-domain reliability
    # ----------------------------
    domain_score = 3
    domain_reasons = []

    in_domain_sources = ["snana_elasticc", "snana_like_table", "elasticc", "lsst_like"]
    external_sources = ["generic", "generic_magnitude", "ztf_irsa", "fits_table"]

    if source_domain in external_sources:
        domain_score = min(domain_score, 2)
        domain_reasons.append(
            f"{source_domain} input may be out-of-domain relative to the ELAsTiCC/LSST-like training data."
        )

    elif source_domain in in_domain_sources:
        domain_reasons.append(
            f"{source_domain} is compatible with the main ELAsTiCC/LSST-like training domain."
        )

    else:
        domain_score = min(domain_score, 2)
        domain_reasons.append(
            f"Unknown source domain: {source_domain}."
        )

    if n_bands == 1:
        domain_score = min(domain_score, 2)
        domain_reasons.append(
            "Single-band input limits the reliability of a model trained with multiband light curves."
        )

    if missing_frac is not None:
        if missing_frac > 0.50:
            domain_score = min(domain_score, 1)
            domain_reasons.append(
                f"More than half of the expected features were filled with zero ({missing_frac:.1%})."
            )
        elif missing_frac > 0.25:
            domain_score = min(domain_score, 2)
            domain_reasons.append(
                f"Some expected features were filled with zero ({missing_frac:.1%})."
            )

    if prediction_result is not None:
        novelty_score = prediction_result.get("novelty_score", None)
        confidence = prediction_result.get("confidence", None)

        if novelty_score is not None and novelty_score >= 0.90:
            domain_score = 1
            domain_reasons.append(
                f"Extreme novelty score ({novelty_score:.3f}); the object is highly out-of-distribution."
            )

        if (
            confidence is not None
            and confidence >= 0.99
            and novelty_score is not None
            and novelty_score >= 0.90
        ):
            domain_reasons.append(
                "High confidence combined with extreme novelty may indicate overconfident out-of-domain prediction."
            )

    if domain_score >= 3:
        domain_level = "Good"
    elif domain_score == 2:
        domain_level = "Limited"
    else:
        domain_level = "Low"

    if not domain_reasons:
        domain_reasons.append("No major model-domain reliability warning detected.")

    return {
        # Backward-compatible fields used by the report exporter.
        "level": domain_level,
        "reasons": domain_reasons,
        "n_obs": int(n_obs),
        "n_bands": int(n_bands),
        "time_span": float(time_span),

        # New explicit structure.
        "input_quality": {
            "level": input_level,
            "n_obs": int(n_obs),
            "n_bands": int(n_bands),
            "time_span": float(time_span),
            "has_flux_uncertainty": bool(has_flux_err),
            "reasons": input_reasons,
        },
        "feature_completeness": feature_info,
        "model_domain_reliability": {
            "level": domain_level,
            "source_domain": source_domain,
            "reasons": domain_reasons,
        },
    }


def display_prediction_reliability(
    lc_obj: pd.DataFrame,
    auto_feature_report=None,
    source_domain="generic",
    prediction_result=None,
):
    reliability = compute_prediction_reliability(
        lc_obj=lc_obj,
        auto_feature_report=auto_feature_report,
        source_domain=source_domain,
        prediction_result=prediction_result,
    )

    input_quality = reliability["input_quality"]
    feature_completeness = reliability["feature_completeness"]
    model_domain = reliability["model_domain_reliability"]

    st.markdown("#### Scientific reliability assessment")

    c1, c2, c3 = st.columns(3)

    with c1:
        level = input_quality["level"]
        if level == "Good":
            st.success(f"Input quality: {level}")
        elif level == "Limited":
            st.warning(f"Input quality: {level}")
        else:
            st.error(f"Input quality: {level}")

    with c2:
        level = feature_completeness["level"]
        if level == "Good":
            st.success(f"Feature completeness: {level}")
        elif level == "Limited":
            st.warning(f"Feature completeness: {level}")
        elif level == "Low":
            st.error(f"Feature completeness: {level}")
        else:
            st.info(f"Feature completeness: {level}")

    with c3:
        level = model_domain["level"]
        if level == "Good":
            st.success(f"Model-domain reliability: {level}")
        elif level == "Limited":
            st.warning(f"Model-domain reliability: {level}")
        else:
            st.error(f"Model-domain reliability: {level}")

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric("Observations", input_quality["n_obs"])
    with m2:
        st.metric("Bands", input_quality["n_bands"])
    with m3:
        safe_metric("Time span", input_quality["time_span"])
    with m4:
        st.metric("Source domain", source_domain)

    if feature_completeness["matched_fraction"] is not None:
        m1, m2, m3 = st.columns(3)

        with m1:
            st.metric("Matched features", feature_completeness["n_matched_features"])
        with m2:
            st.metric("Missing / filled zero", feature_completeness["n_missing_filled_zero"])
        with m3:
            safe_metric("Matched fraction", feature_completeness["matched_fraction"])

    rows = []

    for reason in input_quality["reasons"]:
        rows.append({"category": "Input quality", "reason": reason})

    for reason in feature_completeness.get("reasons", []):
        rows.append({"category": "Feature completeness", "reason": reason})

    for reason in model_domain["reasons"]:
        rows.append({"category": "Model-domain reliability", "reason": reason})

    reason_df = pd.DataFrame(rows)
    st.dataframe(reason_df, use_container_width=True, hide_index=True)

    if model_domain["level"] in ["Limited", "Low"]:
        st.info(
            "Use this prediction as scientific triage support, not as a final classification, "
            "unless the source domain and validation conditions are compatible with the trained model."
        )


def show_snana_elasticc_pair_upload():
    st.markdown("### SNANA / ELAsTiCC HEAD + PHOT pair")
    st.caption(
        "Use this mode for ELAsTiCC/SNANA FITS files where HEAD stores object metadata "
        "and PHOT stores the photometric observations."
    )

    if not HAS_SNANA_UTILS:
        st.error(f"SNANA/ELAsTiCC utilities could not be loaded: {SNANA_UTILS_IMPORT_ERROR}")
        return

    input_mode = st.radio(
        "Input mode",
        ["Browser upload", "Local file path"],
        horizontal=True,
        key="snana_pair_input_mode",
    )

    max_objects = st.number_input(
        "Maximum objects to reconstruct",
        min_value=1,
        max_value=100000,
        value=1000,
        step=100,
        help="Use a smaller value for quick testing. Increase later if needed.",
    )

    try:
        if input_mode == "Browser upload":
            c1, c2 = st.columns(2)

            with c1:
                head_upload = st.file_uploader(
                    "Upload HEAD.FITS",
                    type=["fits", "fit", "fts"],
                    key="snana_head_upload",
                )

            with c2:
                phot_upload = st.file_uploader(
                    "Upload PHOT.FITS",
                    type=["fits", "fit", "fts"],
                    key="snana_phot_upload",
                )

            if head_upload is None or phot_upload is None:
                st.info("Upload both HEAD.FITS and PHOT.FITS to reconstruct light curves.")
                return

            with st.spinner("Reading HEAD + PHOT and reconstructing light curves..."):
                lc, meta, head_df, phot_df = load_snana_pair_from_uploads(
                    head_upload,
                    phot_upload,
                    max_objects=max_objects,
                )

        else:
            c1, c2 = st.columns(2)

            with c1:
                head_path = interactive_local_file_picker(
                    label="Local HEAD.FITS path",
                    suffixes=[".fits", ".fit", ".fts"],
                    default_dir=str(Path.home()),
                    key_prefix="snana_head_local",
                )

            with c2:
                phot_path = interactive_local_file_picker(
                    label="Local PHOT.FITS path",
                    suffixes=[".fits", ".fit", ".fts"],
                    default_dir=str(Path.home()),
                    key_prefix="snana_phot_local",
                )

            if head_path is None or phot_path is None:
                st.info("Select both HEAD.FITS and PHOT.FITS.")
                return

            with st.spinner("Reading HEAD + PHOT and reconstructing light curves..."):
                lc, meta, head_df, phot_df = load_snana_pair_from_paths(
                    head_path,
                    phot_path,
                    max_objects=max_objects,
                )

    except Exception as exc:
        st.error(f"Could not reconstruct SNANA/ELAsTiCC light curves: {exc}")
        return

    st.success(
        f"Reconstructed {meta['n_output_objects']:,} objects and "
        f"{meta['n_output_rows']:,} photometric rows."
    )

    with st.expander("Detected SNANA/ELAsTiCC metadata", expanded=False):
        st.json(meta)

    with st.expander("HEAD preview", expanded=False):
        st.dataframe(head_df.head(50), use_container_width=True)

    with st.expander("PHOT preview", expanded=False):
        st.dataframe(phot_df.head(50), use_container_width=True)

    object_values = sorted(pd.Series(lc["object_id"]).dropna().unique().tolist())

    if len(object_values) == 0:
        st.error("No object_id/SNID values were reconstructed.")
        return

    selected_object = st.selectbox(
        "Select SNID / object_id",
        object_values,
        key="snana_selected_object",
    )

    lc_obj = lc[lc["object_id"] == selected_object].copy()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Observations", len(lc_obj))
    with c2:
        st.metric("Bands", lc_obj["band"].nunique())
    with c3:
        safe_metric("Time span", float(lc_obj["mjd"].max() - lc_obj["mjd"].min()))
    with c4:
        st.metric("SNID", str(selected_object))

    plot_uploaded_lightcurve(lc_obj, title=f"SNANA/ELAsTiCC light curve: {selected_object}")

    st.markdown("#### Prediction readiness")

    checks = pd.DataFrame([
        {"check": "HEAD + PHOT joined", "status": "OK"},
        {"check": "Has MJD, band, flux", "status": "OK"},
        {"check": "Has flux uncertainty", "status": "OK" if lc_obj["flux_err"].notna().any() else "Recommended"},
        {"check": "At least 10 observations", "status": "OK" if len(lc_obj) >= 10 else "Low"},
        {"check": "At least 2 bands", "status": "OK" if lc_obj["band"].nunique() >= 2 else "Low"},
    ])

    st.dataframe(checks, use_container_width=True, hide_index=True)

    head_row = None

    for candidate_col in ["SNID", "object_id", "OBJECT_ID", "DIAOBJECTID", "diaObjectId"]:
        if candidate_col in head_df.columns:
            matches = head_df[
                head_df[candidate_col].astype(str).str.strip() == str(selected_object).strip()
            ]

            if not matches.empty:
                head_row = matches.iloc[0]
                break
    
    head_row = None
    for candidate_col in ["SNID", "object_id", "OBJECT_ID", "DIAOBJECTID", "diaObjectId"]:
        if candidate_col in head_df.columns:
            matches = head_df[
                head_df[candidate_col].astype(str).str.strip()
                == str(selected_object).strip()
            ]

            if not matches.empty:
                head_row = matches.iloc[0]
                break
            
    render_prediction_panel(
        lc_obj,
        selected_object,
        head_row=head_row,
        source_domain="snana_elasticc",
    )



def make_safe_filename(value):
    text = str(value)
    safe = "".join(c if c.isalnum() or c in ["-", "_"] else "_" for c in text)
    return safe[:120]


def json_safe(value):
    """Convert NumPy/Pandas objects to JSON-safe Python objects."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, list):
        return [json_safe(v) for v in value]

    if isinstance(value, tuple):
        return [json_safe(v) for v in value]

    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")

    if isinstance(value, pd.Series):
        return value.to_dict()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)

    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return value

    if value is pd.NA:
        return None

    return value


def build_prediction_export_report(
    selected_object,
    source_domain,
    lc_obj,
    prediction_result,
    reliability,
    auto_feature_report=None,
):
    """Build a structured scientific report for one AstroTrust-AI prediction."""

    top_classes = prediction_result.get("top_classes", [])

    feature_summary = None

    if auto_feature_report is not None:
        feature_summary = {
            "mode": auto_feature_report.get("mode"),
            "warning": auto_feature_report.get("warning"),
            "n_expected_features": auto_feature_report.get("n_expected_features"),
            "n_matched_features": auto_feature_report.get("n_matched_features"),
            "n_missing_filled_zero": auto_feature_report.get("n_missing_filled_zero"),
            "matched_by_category": auto_feature_report.get("matched_by_category", {}),
            "missing_by_category": auto_feature_report.get("missing_by_category", {}),
            "missing_features": auto_feature_report.get("missing_features", []),
        }

    report = {
        "object_id": str(selected_object),
        "source_domain": source_domain,
        "input_summary": {
            "n_rows": int(len(lc_obj)),
            "n_bands": int(lc_obj["band"].nunique()) if "band" in lc_obj.columns else None,
            "mjd_min": float(lc_obj["mjd"].min()) if "mjd" in lc_obj.columns and len(lc_obj) else None,
            "mjd_max": float(lc_obj["mjd"].max()) if "mjd" in lc_obj.columns and len(lc_obj) else None,
            "time_span": float(lc_obj["mjd"].max() - lc_obj["mjd"].min())
            if "mjd" in lc_obj.columns and len(lc_obj)
            else None,
            "bands": sorted(lc_obj["band"].astype(str).unique().tolist()) if "band" in lc_obj.columns else [],
        },
        "prediction": {
            "predicted_label": prediction_result.get("predicted_label"),
            "predicted_class_name": prediction_result.get("predicted_class_name"),
            "confidence": prediction_result.get("confidence"),
            "uncertainty_score": prediction_result.get("uncertainty_score"),
            "novelty_score": prediction_result.get("novelty_score"),
            "rarity_score": prediction_result.get("rarity_score"),
            "priority_score": prediction_result.get("priority_score"),
            "is_predicted_rare": prediction_result.get("is_predicted_rare"),
        },
        "top_classes": top_classes,
        "reliability": reliability,
        "feature_builder": feature_summary,
        "scientific_interpretation": {
            "recommended_use": (
                "Use as triage support, not as a final scientific classification, "
                "when reliability is Limited or Low."
            ),
            "feature_policy": (
                "AstroTrust-AI does not invent astrophysical contextual values. "
                "Unavailable host/context features are filled with zero only as model-compatible placeholders "
                "and are explicitly reported as missing_filled_zero."
            ),
            "notes": [
                "High confidence combined with high novelty may indicate an overconfident out-of-domain prediction.",
                "Generic real-survey inputs may differ from the ELAsTiCC/LSST-like training domain.",
                "Missing host-galaxy/redshift features can reduce scientific reliability.",
            ],
        },
    }

    return json_safe(report)


def render_prediction_export_buttons(
    selected_object,
    source_domain,
    lc_obj,
    prediction_result,
    reliability,
    auto_feature_report=None,
):
    safe_id = make_safe_filename(selected_object)

    report = build_prediction_export_report(
        selected_object=selected_object,
        source_domain=source_domain,
        lc_obj=lc_obj,
        prediction_result=prediction_result,
        reliability=reliability,
        auto_feature_report=auto_feature_report,
    )

    st.markdown("#### Export prediction report")

    json_text = json.dumps(report, indent=2, ensure_ascii=False)

    summary_row = {
        "object_id": report["object_id"],
        "source_domain": report["source_domain"],
        "predicted_class": report["prediction"]["predicted_class_name"],
        "confidence": report["prediction"]["confidence"],
        "uncertainty": report["prediction"]["uncertainty_score"],
        "novelty": report["prediction"]["novelty_score"],
        "rarity": report["prediction"]["rarity_score"],
        "priority_score": report["prediction"]["priority_score"],
        "is_predicted_rare": report["prediction"]["is_predicted_rare"],
        "reliability": report["reliability"]["level"],
        "n_rows": report["input_summary"]["n_rows"],
        "n_bands": report["input_summary"]["n_bands"],
        "time_span": report["input_summary"]["time_span"],
    }

    if report.get("feature_builder") is not None:
        summary_row["n_expected_features"] = report["feature_builder"]["n_expected_features"]
        summary_row["n_matched_features"] = report["feature_builder"]["n_matched_features"]
        summary_row["n_missing_filled_zero"] = report["feature_builder"]["n_missing_filled_zero"]

    summary_df = pd.DataFrame([summary_row])

    c1, c2, c3 = st.columns(3)

    with c1:
        st.download_button(
            "Download report JSON",
            data=json_text.encode("utf-8"),
            file_name=f"astrotrust_prediction_report_{safe_id}.json",
            mime="application/json",
            use_container_width=True,
            key=f"download_report_json_{safe_id}",
            on_click="ignore",
        )

    with c2:
        st.download_button(
            "Download summary CSV",
            data=summary_df.to_csv(index=False).encode("utf-8"),
            file_name=f"astrotrust_prediction_summary_{safe_id}.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"download_summary_csv_{safe_id}",
            on_click="ignore",
        )

    with c3:
        export_lc = lc_obj.copy()
        st.download_button(
            "Download normalized light curve",
            data=export_lc.to_csv(index=False).encode("utf-8"),
            file_name=f"astrotrust_normalized_lightcurve_{safe_id}.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"download_normalized_lc_{safe_id}",
            on_click="ignore",
        )


def merge_metadata_rows(base_row=None, extra_row=None):
    merged = {}

    def add_values(row):
        if row is None:
            return

        if isinstance(row, pd.Series):
            row = row.to_dict()

        if not isinstance(row, dict):
            return

        for key, value in row.items():
            try:
                value = float(value)

                if not np.isfinite(value):
                    continue

                # Evita valores absurdos que quebram float32/scaler/modelo.
                if abs(value) > 1e10:
                    continue

                merged[str(key)] = value

            except Exception:
                # Ignora texto/categorias não numéricas no contexto.
                continue

    add_values(base_row)
    add_values(extra_row)

    return merged if merged else None



def read_context_metadata_file(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(uploaded_file)

    if suffix == ".parquet":
        return pd.read_parquet(uploaded_file)

    raise ValueError("Unsupported context metadata file. Use CSV or Parquet.")



def select_context_row(context_df: pd.DataFrame, selected_object):
    if context_df.empty:
        return None, "Context table is empty. No context row was used."

    selected_object_str = str(selected_object).strip()

    if selected_object_str.lower() in ["nan", "none", ""]:
        return None, "Selected object_id is invalid or missing. No context row was used."

    object_candidates = [
        "object_id",
        "objectid",
        "oid",
        "snid",
        "diaobjectid",
        "diaObjectId",
        "id",
    ]

    object_col = None
    lower_map = {str(c).lower(): c for c in context_df.columns}

    for candidate in object_candidates:
        if candidate.lower() in lower_map:
            object_col = lower_map[candidate.lower()]
            break

    if object_col is not None:
        valid_context = context_df[context_df[object_col].notna()].copy()

        matches = valid_context[
            valid_context[object_col].astype(str).str.strip()
            == selected_object_str
        ]

        if not matches.empty:
            return matches.iloc[0], f"Matched context row by {object_col} = {selected_object_str}."

        return None, (
            f"No matching context row found for object_id = {selected_object_str}. "
            "No context metadata was used."
        )

    return None, "No object_id column found in context table. No context metadata was used."


def build_host_context_template(selected_object):
    columns = [
        "object_id",
        "ra",
        "dec",
        "ra_host",
        "dec_host",
        "hostgal_zphot",
        "hostgal_zphot_err",
        "hostgal_zphot_q000",
        "hostgal_zphot_q010",
        "hostgal_zphot_q020",
        "hostgal_zphot_q080",
        "hostgal_zphot_q090",
        "hostgal_zphot_q100",
        "hostgal_mag_u",
        "hostgal_mag_g",
        "hostgal_mag_r",
        "hostgal_mag_i",
        "hostgal_mag_z",
        "hostgal_mag_y",
        "hostgal_magerr_u",
        "hostgal_magerr_g",
        "hostgal_magerr_r",
        "hostgal_magerr_i",
        "hostgal_magerr_z",
        "hostgal_magerr_y",
        "hostgal_ellipticity",
        "hostgal_sqradius",
        "mwebv",
        "mwebv_err",
    ]

    row = {col: "" for col in columns}
    row["object_id"] = str(selected_object)

    return pd.DataFrame([row])


def render_prediction_panel(lc_obj, selected_object, head_row=None, source_domain="generic"):
    st.markdown("#### AstroTrust-AI prediction")

    if not HAS_SNANA_UTILS:
        st.error(f"Inference module could not be loaded: {SNANA_UTILS_IMPORT_ERROR}")
        return

    prediction_mode = st.radio(
        "Prediction mode",
        [
            "Full hybrid inference: auto-build v4 features from light curve",
            "Full hybrid inference: auto-match precomputed v4 features by object_id",
            "Diagnostic preview: light curve only + zero tabular features",
            "Full hybrid inference: light curve + uploaded v4 tabular features",
        ],
        horizontal=False,
        key=f"prediction_mode_{selected_object}",
    )

    tabular_features = None
    allow_zero_tabular = False
    extra_context_row = None

    with st.expander("Optional host/context metadata", expanded=False):
        st.caption(
            "Upload an optional CSV/Parquet table with host-galaxy or contextual features "
            "such as hostgal_zphot, hostgal_zphot_err, hostgal_mag_g, hostgal_color_g_r, hostgal_snsep."
        )

        template_df = build_host_context_template(selected_object)

        st.download_button(
            "Download host/context template CSV",
            data=template_df.to_csv(index=False).encode("utf-8"),
            file_name=f"astrotrust_host_context_template_{make_safe_filename(selected_object)}.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"download_context_template_{make_safe_filename(selected_object)}",
            on_click="ignore",
        )

        context_upload = st.file_uploader(
            "Upload host/context metadata table",
            type=["csv", "parquet"],
            key=f"context_metadata_{make_safe_filename(selected_object)}",
        )

        if context_upload is not None:
            try:
                context_df = read_context_metadata_file(context_upload)

                st.success(
                    f"Loaded context table with {len(context_df):,} rows and "
                    f"{len(context_df.columns):,} columns."
                )

                with st.expander("Context table preview", expanded=False):
                    st.dataframe(context_df.head(20), use_container_width=True)

                extra_context_row, context_message = select_context_row(
                    context_df,
                    selected_object,
                )

                if extra_context_row is None:
                    st.warning(context_message)
                else:
                    st.success(context_message)

            except Exception as exc:
                st.error(f"Could not read context metadata table: {exc}")
    use_auto_builder = prediction_mode.startswith("Full hybrid inference: auto-build")
    use_precomputed_features = prediction_mode.startswith("Full hybrid inference: auto-match")
    auto_feature_report = None

    if use_auto_builder:
        if not HAS_FEATURE_BUILDER:
            st.error(f"Automatic feature builder could not be loaded: {FEATURE_BUILDER_IMPORT_ERROR}")
            return

        st.info(
            "This mode automatically builds the tabular v4 feature vector from the selected light curve "
            "and available HEAD metadata. Features that require unavailable external context are filled with zero."
        )

    elif use_precomputed_features:
        st.info(
            "This mode tries to match the selected SNID/object_id with "
            "features_v4_temporal_shape_250000obj.parquet. If a match is found, "
            "the prediction uses the full hybrid model."
        )

    elif prediction_mode.startswith("Diagnostic"):
        allow_zero_tabular = True
        st.warning(
            "Diagnostic mode uses the uploaded light curve but fills the 319 tabular/context features with zeros. "
            "This tests the interface, but it is not the full scientific model used in the experiments."
        )

    else:
        st.info(
            "For full hybrid inference, upload a CSV/Parquet/FITS table containing the v4 tabular feature row "
            "for the same object."
        )

        tab_upload = st.file_uploader(
            "Upload v4 tabular feature row",
            type=["csv", "parquet", "fits", "fit", "fts"],
            key=f"tabular_features_{selected_object}",
        )

        if tab_upload is not None:
            try:
                tab_df, tab_source = read_uploaded_lightcurve_file(tab_upload)

                st.success(
                    f"Loaded tabular features as {tab_source}: "
                    f"{len(tab_df):,} rows and {len(tab_df.columns):,} columns."
                )

                if "object_id" in tab_df.columns:
                    matches = tab_df[tab_df["object_id"].astype(str) == str(selected_object)]

                    if not matches.empty:
                        tabular_features = matches.iloc[[0]].copy()
                        st.info(f"Matched tabular feature row by object_id = {selected_object}.")
                    else:
                        tabular_features = tab_df.iloc[[0]].copy()
                        st.warning("No matching object_id found. Using the first row.")
                else:
                    tabular_features = tab_df.iloc[[0]].copy()
                    st.warning("No object_id column found. Using the first row.")

                tabular_features = tabular_features.drop(
                    columns=[c for c in ["object_id", "label"] if c in tabular_features.columns],
                    errors="ignore",
                )

            except Exception as exc:
                st.error(f"Could not load tabular feature file: {exc}")
                tabular_features = None

        if tabular_features is None:
            st.warning("Full hybrid prediction requires a valid v4 tabular feature row.")

    run_prediction = st.button(
        "Predict with AstroTrust-AI",
        type="primary",
        use_container_width=True,
        key=f"predict_button_{selected_object}",
    )

    if not run_prediction:
        return

    if (
        prediction_mode.startswith("Full hybrid inference: light curve + uploaded")
        and tabular_features is None
    ):
        st.error("Please upload a valid v4 tabular feature row or switch to diagnostic preview mode.")
        return

    try:
        with st.spinner("Running AstroTrust-AI inference..."):
            engine = get_inference_engine()

            if use_auto_builder:
                combined_context_row = merge_metadata_rows(
                    base_row=head_row,
                    extra_row=extra_context_row,
                )

                tabular_features, auto_feature_report = build_v4_feature_row_auto(
                    lc_obj=lc_obj,
                    head_row=combined_context_row,
                    expected_columns=engine.tabular_columns,
                )
                allow_zero_tabular = False

            elif use_precomputed_features:
                tabular_features = engine.get_precomputed_tabular_features(selected_object)

                if tabular_features is None:
                    st.error(
                        f"No precomputed v4 tabular features found for object_id/SNID = {selected_object}. "
                        "Use auto-build mode or diagnostic preview."
                    )
                    return

                allow_zero_tabular = False

            result = engine.predict_from_lightcurve(
                lightcurve_df=lc_obj,
                tabular_features=tabular_features,
                allow_zero_tabular=allow_zero_tabular,
            )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric("Predicted class", result["predicted_class_name"])
        with c2:
            st.metric("Confidence", f"{result['confidence']:.4f}")
        with c3:
            st.metric("Priority score", f"{result['priority_score']:.4f}")
        with c4:
            st.metric("Rare prediction", "Yes" if result["is_predicted_rare"] else "No")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Uncertainty", f"{result['uncertainty_score']:.4f}")
        with c2:
            st.metric("Novelty", f"{result['novelty_score']:.4f}")
        with c3:
            st.metric("Rarity", f"{result['rarity_score']:.4f}")

        display_prediction_reliability(
            lc_obj=lc_obj,
            auto_feature_report=auto_feature_report,
            source_domain=source_domain,
            prediction_result=result,
        )

        reliability = compute_prediction_reliability(
            lc_obj=lc_obj,
            auto_feature_report=auto_feature_report,
            source_domain=source_domain,
            prediction_result=result,
        )

        if auto_feature_report is not None:
            st.info(
                f"Auto-built v4 features: "
                f"{auto_feature_report['n_matched_features']} matched, "
                f"{auto_feature_report['n_missing_filled_zero']} filled with zero "
                f"out of {auto_feature_report['n_expected_features']} expected features."
            )

            with st.expander("Automatic feature builder report", expanded=False):
                st.json({
                    "mode": auto_feature_report["mode"],
                    "warning": auto_feature_report["warning"],
                    "n_expected_features": auto_feature_report["n_expected_features"],
                    "n_matched_features": auto_feature_report["n_matched_features"],
                    "n_missing_filled_zero": auto_feature_report["n_missing_filled_zero"],
                    "matched_by_category": auto_feature_report.get("matched_by_category", {}),
                    "missing_by_category": auto_feature_report.get("missing_by_category", {}),
                })

                if "feature_coverage" in auto_feature_report:
                    coverage_df = auto_feature_report["feature_coverage"]

                    st.markdown("##### Feature coverage table")
                    st.dataframe(coverage_df, use_container_width=True, hide_index=True)

                    csv = coverage_df.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        "Download feature coverage CSV",
                        data=csv,
                        file_name=f"astrotrust_feature_coverage_{selected_object}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key=f"download_feature_coverage_{make_safe_filename(selected_object)}",
                        on_click="ignore",
                    )
            
            render_prediction_export_buttons(
                selected_object=selected_object,
                source_domain=source_domain,
                lc_obj=lc_obj,
                prediction_result=result,
                reliability=reliability,
                auto_feature_report=auto_feature_report,
            )
        if result.get("used_zero_tabular_preview"):
            st.warning(
                "This result used zero-filled tabular/context features. "
                "Use full hybrid inference with v4 tabular features for scientific results."
            )

        top_classes = pd.DataFrame(result["top_classes"])

        st.markdown("#### Top-5 predicted classes")
        st.dataframe(top_classes, use_container_width=True, hide_index=True)

        fig = px.bar(
            top_classes.sort_values("probability", ascending=True),
            x="probability",
            y="class_name",
            orientation="h",
            title="Top-5 class probabilities",
        )

        fig.update_layout(
            height=320,
            yaxis_title="",
            xaxis_title="Probability",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.42)",
            font=dict(color="#E5E7EB"),
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

def show_upload_alert_lightcurve():
    st.subheader("Upload Alert / Light Curve")
    st.caption(
        "Import a light-curve or alert-like table in CSV, Parquet, or FITS-table format. "
        "For large local files, use local path mode instead of browser upload."
    )

    input_format = st.radio(
        "Input format",
        [
            "Generic light-curve table",
            "SNANA / ELAsTiCC HEAD + PHOT pair",
        ],
        horizontal=True,
        key="upload_input_format",
    )

    if input_format == "SNANA / ELAsTiCC HEAD + PHOT pair":
        show_snana_elasticc_pair_upload()
        return

    if not HAS_DATA_ADAPTER:
        st.error(f"AstroTrust data adapter could not be loaded: {DATA_ADAPTER_IMPORT_ERROR}")
        return

    input_mode = st.radio(
        "Input mode",
        ["Browser upload", "Local file path", "URL"],
        horizontal=True,
        key="lightcurve_input_mode",
    )

    raw_df = None
    lc = None
    report = None
    source_label = None

    if input_mode == "Browser upload":
        uploaded = st.file_uploader(
            "Upload light-curve table",
            type=["csv", "parquet", "fits", "fit", "fts"],
            help=(
                "Accepted schemas include flux-based tables "
                "(mjd, band, flux, flux_err) and magnitude-based real survey tables "
                "(mjd, filter, mag, magerr)."
            ),
            key="lightcurve_upload",
        )

        if uploaded is None:
            st.info("Upload a CSV, Parquet, or FITS table containing a light curve to begin.")
            st.markdown(
                """
                **Expected minimum schema**

                | Column role | Accepted examples |
                |---|---|
                | Time | `mjd`, `time`, `jd`, `hjd`, `bjd` |
                | Band/filter | `band`, `filter`, `filtercode`, `passband`, `fid` |
                | Flux | `flux`, `fluxcal`, `forcediffimflux` |
                | Magnitude | `mag`, `magpsf`, `psfmag`, `magnitude` |
                | Uncertainty | `flux_err`, `fluxerr`, `magerr`, `sigmapsf` |
                | Object identifier | `object_id`, `oid`, `diaObjectId`, `SNID` |
                """
            )
            return

        try:
            raw_df, lc, report = read_and_normalize_from_uploaded(uploaded)
            source_label = uploaded.name
        except Exception as exc:
            st.error(f"Could not read uploaded file: {exc}")
            return

    elif input_mode == "Local file path":
        local_file = interactive_local_file_picker(
            label="Local light-curve path",
            suffixes=[".csv", ".parquet", ".fits", ".fit", ".fts"],
            default_dir=str(ROOT_DIR / "data" / "processed" / "elasticc2_large"),
            key_prefix="lightcurve_local",
        )

        if local_file is None:
            st.info("Browse a folder or paste a local path to a CSV, Parquet, or FITS table containing light-curve data.")
            return

        try:
            raw_df, lc, report = read_and_normalize_from_path(local_file)
            source_label = str(local_file)
        except Exception as exc:
            st.error(f"Could not read local file: {exc}")
            return

    else:
        st.info(
            "Paste a public URL pointing to a CSV/Parquet light-curve table. "
            "For IRSA/ZTF, use a URL with FORMAT=CSV."
        )

        url = st.text_input(
            "Light-curve table URL",
            placeholder="https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?...&FORMAT=CSV",
            key="lightcurve_url",
        )

        if not url:
            return

        try:
            with st.spinner("Downloading and normalizing light-curve table from URL..."):
                raw_df, lc, report = read_and_normalize_from_url(url)
                source_label = url
        except Exception as exc:
            st.error(f"Could not read URL: {exc}")
            return

    report_dict = report.to_dict()

    st.success(
        f"Loaded `{source_label}` as `{report.source_type}` with "
        f"{report.n_raw_rows:,} rows and {report.n_raw_columns:,} columns."
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Detected domain", report.source_domain)
    with c2:
        st.metric("Objects", report.n_objects)
    with c3:
        st.metric("Normalized rows", report.n_output_rows)
    with c4:
        st.metric("Mag → flux", "Yes" if report.used_magnitude_conversion else "No")

    for warning in report.warnings:
        st.warning(warning)

    for issue in report.issues:
        st.error(issue)

    with st.expander("Adapter report", expanded=False):
        st.json(report_dict)

    with st.expander("Raw table preview", expanded=False):
        st.dataframe(raw_df.head(500), use_container_width=True)

    st.markdown("#### Column mapping")
    mapping_df = pd.DataFrame(
        [
            {
                "role": role,
                "detected_column": col if col is not None else "—",
            }
            for role, col in report.column_mapping.items()
        ]
    )
    st.dataframe(mapping_df, use_container_width=True, hide_index=True)

    if lc.empty:
        st.error("The uploaded file could not be converted into a valid light curve.")
        return

    object_summary = adapter_summarize_objects(lc)

    if object_summary.empty:
        selected_object = "single_object"
        lc["object_id"] = selected_object
        lc_obj = lc.copy()

    else:
        object_values = object_summary["object_id"].astype(str).tolist()

        def format_object_option(object_id):
            row = object_summary[object_summary["object_id"].astype(str) == str(object_id)].iloc[0]
            return (
                f"{row['object_id']}  |  "
                f"bands={row['n_bands']}  |  "
                f"obs={row['n_obs']}  |  "
                f"span={row['time_span']:.1f}"
            )

        selected_object = st.selectbox(
            "Select object",
            object_values,
            index=0,
            format_func=format_object_option,
            help="Objects are sorted automatically by number of bands, observations, and time span.",
        )

        lc_obj = lc[lc["object_id"].astype(str) == str(selected_object)].copy()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Rows", len(lc_obj))
    with c2:
        st.metric("Bands", lc_obj["band"].nunique())
    with c3:
        safe_metric("Time span", float(lc_obj["mjd"].max() - lc_obj["mjd"].min()))
    with c4:
        st.metric("Object", str(selected_object))

    plot_uploaded_lightcurve(lc_obj, title=f"Uploaded light curve: {selected_object}")

    st.markdown("#### Prediction readiness")

    checks = pd.DataFrame([
        {
            "check": "Has time, band, and flux",
            "status": "OK" if all(c in lc_obj.columns for c in ["mjd", "band", "flux"]) else "Missing",
        },
        {
            "check": "Has flux uncertainty",
            "status": "OK" if ("flux_err" in lc_obj.columns and lc_obj["flux_err"].notna().any()) else "Recommended",
        },
        {
            "check": "At least 10 observations",
            "status": "OK" if len(lc_obj) >= 10 else "Low",
        },
        {
            "check": "At least 2 bands",
            "status": "OK" if lc_obj["band"].nunique() >= 2 else "Low",
        },
    ])

    st.dataframe(checks, use_container_width=True, hide_index=True)

    render_prediction_panel(
        lc_obj,
        selected_object,
        source_domain=report.source_domain,
    )


def show_overview(assets):
    perf = assets["performance"]
    cal = assets["calibration"]
    policy = assets["policy_summary"]

    st.markdown(
        """
        <div class="hero">
            <h1>AstroTrust-AI</h1>
            <p>Interactive decision-support dashboard for astronomical alert triage. Explore hybrid temporal-tabular classification, calibrated uncertainty, novelty, rarity, and follow-up ranking.</p>
            <div class="chip-row">
                <span class="chip">Hybrid temporal-tabular AI</span>
                <span class="chip">Calibrated probabilities</span>
                <span class="chip">Novelty + rarity follow-up</span>
                <span class="chip">FITS viewer</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not perf.empty:
        best = perf.sort_values("macro_f1", ascending=False).iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            safe_metric("Best Accuracy", best.get("accuracy"))
        with c2:
            safe_metric("Best Macro-F1", best.get("macro_f1"))
        with c3:
            safe_metric("Balanced Accuracy", best.get("balanced_accuracy"))
        with c4:
            safe_metric("Best model", best.get("model"))

    c1, c2 = st.columns([1.25, 1])
    with c1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Model evolution")
        plot_model_performance(perf)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Calibration")
        if not cal.empty:
            temp_test = cal[cal["model"].astype(str).str.contains("temp_test", na=False)]
            raw_test = cal[cal["model"].astype(str).str.contains("raw_test", na=False)]
            c21, c22 = st.columns(2)
            with c21:
                safe_metric("Raw ECE", raw_test["ece"].iloc[0] if not raw_test.empty else None)
            with c22:
                safe_metric("Temp. ECE", temp_test["ece"].iloc[0] if not temp_test.empty else None)
            plot_calibration(cal)
        else:
            st.info("Calibration file not found.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Follow-up policy summary")
    plot_policy_enrichment(policy)
    st.markdown('</div>', unsafe_allow_html=True)


def show_candidate_explorer(class_names):
    st.subheader("Candidate Explorer")
    st.caption("Filter and inspect candidates ranked for follow-up. Use this page as a broker-like triage view.")

    with st.sidebar:
        st.markdown("---")
        st.subheader("Candidate filters")
        policy_label = st.selectbox("Follow-up policy", list(POLICY_FILES.keys()), index=0)
        top_n = st.slider("Top N candidates", min_value=50, max_value=5000, value=500, step=50)

    ranking = load_ranking(policy_label)
    if ranking.empty:
        st.error("Ranking data not found. Run app/prepare_dashboard_data.py first.")
        return

    ranking = add_class_names(ranking, class_names)
    filtered = ranking.copy()

    with st.sidebar:
        if "predicted_class_name" in filtered.columns:
            classes = ["All"] + sorted(filtered["predicted_class_name"].dropna().unique().tolist())
            selected_class = st.selectbox("Predicted class", classes)
            if selected_class != "All":
                filtered = filtered[filtered["predicted_class_name"] == selected_class]

        status = st.selectbox("Prediction status", ["All", "Correct only", "Errors only"])
        if status == "Correct only" and "correct" in filtered.columns:
            filtered = filtered[filtered["correct"] == True]
        elif status == "Errors only" and "correct" in filtered.columns:
            filtered = filtered[filtered["correct"] == False]

        if "true_is_rare" in filtered.columns:
            rare_filter = st.selectbox("True rarity", ["All", "Rare only", "Non-rare only"])
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
            if col in filtered.columns and not filtered.empty:
                min_v, max_v = float(filtered[col].min()), float(filtered[col].max())
                if min_v < max_v:
                    lo, hi = st.slider(label, min_value=min_v, max_value=max_v, value=(min_v, max_v))
                    filtered = filtered[(filtered[col] >= lo) & (filtered[col] <= hi)]

    filtered = filtered.sort_values("priority_score", ascending=False).head(top_n)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Candidates", len(filtered))
    with c2:
        safe_metric("Rare true rate", filtered["true_is_rare"].mean() if "true_is_rare" in filtered.columns and len(filtered) else None)
    with c3:
        safe_metric("Selection accuracy", filtered["correct"].mean() if "correct" in filtered.columns and len(filtered) else None)
    with c4:
        safe_metric("Mean priority", filtered["priority_score"].mean() if "priority_score" in filtered.columns and len(filtered) else None)

    display_cols = [
        "priority_rank", "object_id", "true_class_name", "predicted_class_name", "correct",
        "confidence", "uncertainty_score", "novelty_score", "rarity_score", "priority_score",
        "true_is_rare", "predicted_is_rare",
    ]
    display_cols = [c for c in display_cols if c in filtered.columns]

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("Inspect candidate")
    if filtered.empty:
        st.warning("No candidates match the filters.")
        return

    object_ids = filtered["object_id"].astype(int).tolist()
    selected_object = st.selectbox("Select object_id", object_ids, index=0)
    selected_row = filtered[filtered["object_id"].astype(int) == int(selected_object)].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Object", int(selected_row["object_id"]))
    with c2:
        st.metric("Predicted", selected_row.get("predicted_class_name", "—"))
    with c3:
        st.metric("True", selected_row.get("true_class_name", "—"))
    with c4:
        status_text = "Correct" if bool(selected_row.get("correct", False)) else "Error"
        st.metric("Status", status_text)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        safe_metric("Confidence", selected_row.get("confidence"))
    with c2:
        safe_metric("Uncertainty", selected_row.get("uncertainty_score"))
    with c3:
        safe_metric("Novelty", selected_row.get("novelty_score"))
    with c4:
        safe_metric("Rarity", selected_row.get("rarity_score"))

    lc_path = CACHE_DIR / "dashboard_test_lightcurves.parquet"
    if lc_path.exists():
        lc = load_lightcurves(lc_path)
        lc_object = lc[lc["object_id"].astype(int) == int(selected_object)]
        plot_light_curve(lc_object)
    else:
        st.warning("Light-curve cache not found. Run: python .\\app\\prepare_dashboard_data.py")


def show_broker_view(class_names):
    st.subheader("Broker view")
    st.caption("Operational view for top follow-up budgets. This is useful for simulating limited observing resources.")

    policy_label = st.selectbox("Policy", list(POLICY_FILES.keys()), index=0, key="broker_policy")
    ranking = load_ranking(policy_label)
    if ranking.empty:
        st.error("Ranking data not found.")
        return
    ranking = add_class_names(ranking, class_names)

    budget = st.radio("Follow-up budget", ["Top 5%", "Top 10%", "Top 20%"], horizontal=True)
    frac = {"Top 5%": 0.05, "Top 10%": 0.10, "Top 20%": 0.20}[budget]
    k = max(1, int(np.ceil(len(ranking) * frac)))
    top = ranking.sort_values("priority_score", ascending=False).head(k)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Selected", k)
    with c2:
        safe_metric("Rare rate", top["true_is_rare"].mean() if "true_is_rare" in top.columns else None)
    with c3:
        safe_metric("Accuracy", top["correct"].mean() if "correct" in top.columns else None)
    with c4:
        safe_metric("Mean priority", top["priority_score"].mean() if "priority_score" in top.columns else None)

    if "predicted_class_name" in top.columns:
        counts = top["predicted_class_name"].value_counts().reset_index()
        counts.columns = ["predicted_class", "n"]
        fig = px.bar(counts.head(20), x="n", y="predicted_class", orientation="h", title="Top selected candidates by predicted class")
        fig.update_layout(height=520, yaxis_title="", xaxis_title="Count", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.42)", font=dict(color="#E5E7EB"))
        st.plotly_chart(fig, use_container_width=True)

    display_cols = ["priority_rank", "object_id", "predicted_class_name", "true_class_name", "correct", "confidence", "novelty_score", "rarity_score", "priority_score", "true_is_rare"]
    display_cols = [c for c in display_cols if c in top.columns]
    st.dataframe(top[display_cols], use_container_width=True, hide_index=True)


def main():
    class_names = load_class_names()
    assets = load_dashboard_assets() if CACHE_DIR.exists() else {}

    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/telescope.png", width=58)
        st.title("AstroTrust-AI")
        st.caption("Scientific alert triage dashboard")

        if not CACHE_DIR.exists():
            st.error("Dashboard cache not found.")
            st.code("python .\\app\\prepare_dashboard_data.py")

    tab_overview, tab_candidates, tab_broker, tab_upload, tab_fits = st.tabs([
        "Overview",
        "Candidate Explorer",
        "Broker View",
        "Upload Alert / Light Curve",
        "Advanced Data Inspector",
    ])

    with tab_overview:
        if not CACHE_DIR.exists():
            st.error("Dashboard cache not found. Run: python .\\app\\prepare_dashboard_data.py")
        else:
            show_overview(assets)

    with tab_candidates:
        if not CACHE_DIR.exists():
            st.error("Dashboard cache not found. Run: python .\\app\\prepare_dashboard_data.py")
        else:
            show_candidate_explorer(class_names)

    with tab_broker:
        if not CACHE_DIR.exists():
            st.error("Dashboard cache not found. Run: python .\\app\\prepare_dashboard_data.py")
        else:
            show_broker_view(class_names)

    with tab_upload:
        show_upload_alert_lightcurve()

    with tab_fits:
        show_fits_viewer()


if __name__ == "__main__":
    main()
