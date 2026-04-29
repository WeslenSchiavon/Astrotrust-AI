from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT_DIR / "results" / "hybrid_inference_artifacts_250k"
PRECOMPUTED_V4_FEATURES_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "elasticc2_large"
    / "features_v4_temporal_shape_250000obj.parquet"
)

BANDS = ["u", "g", "r", "i", "z", "Y"]
BAND_TO_IDX = {b: i for i, b in enumerate(BANDS)}

NUMERIC_BAND_MAP = {
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

EPS = 1e-8


class HybridTemporalTabularCNN(nn.Module):
    def __init__(self, n_channels=18, n_tabular=319, n_classes=32):
        super().__init__()

        self.temporal_branch = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.10),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.15),

            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.20),

            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.20),
        )

        self.temporal_pool = nn.AdaptiveAvgPool1d(1)

        self.temporal_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
        )

        self.tabular_branch = nn.Sequential(
            nn.Linear(n_tabular, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Dropout(0.35),

            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(192, n_classes),
        )

    def forward(self, x_lc, x_tab):
        z_lc = self.temporal_branch(x_lc)
        z_lc = self.temporal_pool(z_lc)
        z_lc = self.temporal_head(z_lc)

        z_tab = self.tabular_branch(x_tab)
        z = torch.cat([z_lc, z_tab], dim=1)
        return self.classifier(z)


def normalize_band(value):
    if value in NUMERIC_BAND_MAP:
        return NUMERIC_BAND_MAP[value]

    value_str = str(value).strip()

    if value_str in NUMERIC_BAND_MAP:
        return NUMERIC_BAND_MAP[value_str]

    if value_str.lower() == "y":
        return "Y"

    return value_str.lower()


def compute_snr(df: pd.DataFrame) -> np.ndarray:
    if "snr" in df.columns:
        return pd.to_numeric(df["snr"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

    flux = pd.to_numeric(df["flux"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

    for err_col in ["flux_err", "fluxerr", "flux_error"]:
        if err_col in df.columns:
            err = pd.to_numeric(df[err_col], errors="coerce").fillna(np.nan).to_numpy(dtype=np.float32)
            err = np.where(np.isfinite(err) & (np.abs(err) > EPS), err, np.nan)
            snr = flux / err
            snr = np.nan_to_num(snr, nan=0.0, posinf=0.0, neginf=0.0)
            return snr.astype(np.float32)

    return np.zeros(len(df), dtype=np.float32)


def robust_flux_scale(flux: np.ndarray) -> float:
    flux = flux[np.isfinite(flux)]

    if len(flux) == 0:
        return 1.0

    scale = np.nanpercentile(np.abs(flux), 95)

    if not np.isfinite(scale) or scale < EPS:
        scale = np.nanstd(flux)

    if not np.isfinite(scale) or scale < EPS:
        scale = 1.0

    return float(scale)


def lightcurve_to_tensor(lightcurve_df: pd.DataFrame, n_bins: int = 64) -> np.ndarray:
    """Convert one object's light curve into the [18, n_bins] tensor used by AstroTrust-AI.

    Required normalized columns: mjd, band, flux.
    Optional columns: flux_err or snr.
    """
    required = ["mjd", "band", "flux"]
    missing = [c for c in required if c not in lightcurve_df.columns]
    if missing:
        raise ValueError(f"Missing required light-curve columns: {missing}")

    tensor = np.zeros((18, n_bins), dtype=np.float32)
    counts = np.zeros((6, n_bins), dtype=np.float32)

    group = lightcurve_df.copy()
    group["band_norm"] = group["band"].map(normalize_band)

    mjd = pd.to_numeric(group["mjd"], errors="coerce").to_numpy(dtype=np.float32)
    flux = pd.to_numeric(group["flux"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    snr = compute_snr(group)

    valid = np.isfinite(mjd) & np.isfinite(flux)

    if valid.sum() == 0:
        return tensor

    group = group.loc[valid].copy()
    mjd = mjd[valid]
    flux = flux[valid]
    snr = snr[valid]

    t_min = np.nanmin(mjd)
    t_max = np.nanmax(mjd)

    if not np.isfinite(t_min) or not np.isfinite(t_max) or abs(t_max - t_min) < EPS:
        bin_idx = np.zeros(len(mjd), dtype=int)
    else:
        t_norm = (mjd - t_min) / (t_max - t_min)
        bin_idx = np.floor(t_norm * (n_bins - 1)).astype(int)
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    scale = robust_flux_scale(flux)
    flux_norm = flux / scale
    flux_norm = np.clip(flux_norm, -10.0, 10.0) / 10.0

    snr_norm = np.clip(snr, -20.0, 20.0) / 20.0
    bands = group["band_norm"].to_numpy()

    for i in range(len(group)):
        band = bands[i]
        if band not in BAND_TO_IDX:
            continue

        b = BAND_TO_IDX[band]
        t = bin_idx[i]

        tensor[b, t] += flux_norm[i]
        tensor[6 + b, t] += snr_norm[i]
        tensor[12 + b, t] = 1.0
        counts[b, t] += 1.0

    nonzero = counts > 0
    for b in range(6):
        for t in range(n_bins):
            if nonzero[b, t]:
                tensor[b, t] /= counts[b, t]
                tensor[6 + b, t] /= counts[b, t]

    return tensor


def normalized_entropy(probabilities: np.ndarray) -> float:
    eps = 1e-12
    k = probabilities.shape[0]
    entropy = -np.sum(probabilities * np.log(probabilities + eps))
    return float(entropy / np.log(k))


def minmax_transform(value: float, min_v: float, max_v: float) -> float:
    if np.isclose(min_v, max_v):
        return 0.0
    out = (value - min_v) / (max_v - min_v)
    return float(np.clip(out, 0.0, 1.0))


class AstroTrustInferenceEngine:
    def __init__(self, artifact_dir: Path = ARTIFACT_DIR, device: str | None = None):
        self.artifact_dir = Path(artifact_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.metadata = self._load_json("inference_metadata.json")
        self.temperature_meta = self._load_json("temperature_scaling.json")
        self.label_meta = self._load_json("label_and_rarity_metadata.json")
        self.novelty_norm = self._load_json("novelty_normalization.json")

        self.temperature = float(self.temperature_meta["temperature"])
        self.tabular_columns = self.metadata["tabular_columns"]
        self.n_bins = int(self.metadata.get("n_bins", 64))
        self.n_channels = int(self.metadata.get("n_channels", 18))
        self.n_classes = int(self.metadata.get("n_classes", 32))

        self.label_to_name = {int(k): v for k, v in self.label_meta["label_to_name"].items()}
        self.global_counts = {int(k): int(v) for k, v in self.label_meta["global_counts"].items()}
        self.rare_classes = [int(v) for v in self.label_meta["rare_classes"]]
        self.rarity_norm = self.label_meta["rarity_score_normalization"]

        self.scaler = joblib.load(self.artifact_dir / self.metadata["files"]["scaler"])
        self.novelty_models = joblib.load(self.artifact_dir / self.metadata["files"]["novelty_models"])

        checkpoint_path = self.artifact_dir / self.metadata["files"]["checkpoint"]
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model = HybridTemporalTabularCNN(
            n_channels=int(checkpoint["n_channels"]),
            n_tabular=int(checkpoint["n_tabular"]),
            n_classes=int(checkpoint["n_classes"]),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self._precomputed_features = None

    def _load_json(self, filename: str):
        path = self.artifact_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        return json.loads(path.read_text(encoding="utf-8"))

    def load_precomputed_v4_features(self) -> pd.DataFrame:
        """Load precomputed v4 tabular/context features for known ELAsTiCC objects.

        This enables full hybrid inference for objects that are already present in the
        processed AstroTrust-AI feature table. For truly new external alerts, a full
        feature-extraction pipeline is still required.
        """
        if self._precomputed_features is not None:
            return self._precomputed_features

        if not PRECOMPUTED_V4_FEATURES_PATH.exists():
            raise FileNotFoundError(PRECOMPUTED_V4_FEATURES_PATH)

        df = pd.read_parquet(PRECOMPUTED_V4_FEATURES_PATH)

        if "object_id" not in df.columns:
            raise ValueError(f"Precomputed feature file does not contain object_id: {PRECOMPUTED_V4_FEATURES_PATH}")

        df = df.copy()
        df["_object_id_key"] = df["object_id"].astype(str).str.strip()
        df = df.drop_duplicates("_object_id_key").set_index("_object_id_key", drop=False)

        self._precomputed_features = df
        return self._precomputed_features

    def get_precomputed_tabular_features(self, object_id) -> pd.DataFrame | None:
        """Return one v4 feature row matched by object_id/SNID, or None if not found."""
        df = self.load_precomputed_v4_features()
        key = str(object_id).strip()

        if key in df.index:
            return df.loc[[key]].copy()

        # Fallback for IDs represented as float-like strings, e.g. '10044575.0'.
        try:
            key_int = str(int(float(key)))
            if key_int in df.index:
                return df.loc[[key_int]].copy()
        except Exception:
            pass

        return None

    def build_tabular_vector(self, tabular_features: dict | pd.Series | pd.DataFrame | None, allow_zero_tabular: bool = False):
        if tabular_features is None:
            if not allow_zero_tabular:
                raise ValueError(
                    "Full hybrid inference requires tabular/context features. "
                    "Provide the v4 tabular feature columns or set allow_zero_tabular=True for a limited diagnostic preview."
                )
            return np.zeros((1, len(self.tabular_columns)), dtype=np.float32), self.tabular_columns.copy()

        if isinstance(tabular_features, pd.DataFrame):
            if len(tabular_features) != 1:
                raise ValueError("tabular_features DataFrame must contain exactly one row for single-object inference.")
            source = tabular_features.iloc[0].to_dict()
        elif isinstance(tabular_features, pd.Series):
            source = tabular_features.to_dict()
        elif isinstance(tabular_features, dict):
            source = tabular_features
        else:
            raise TypeError("tabular_features must be dict, pandas Series, pandas DataFrame, or None.")

        missing = []
        values = []
        for col in self.tabular_columns:
            if col in source:
                values.append(source[col])
            else:
                values.append(0.0)
                missing.append(col)

        x = pd.DataFrame([values], columns=self.tabular_columns)
        x = x.apply(pd.to_numeric, errors="coerce")
        x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        return x.to_numpy(dtype=np.float32), missing

    def _compute_raw_novelty(self, x_tab_scaled: np.ndarray) -> float:
        distances = []
        for model in self.novelty_models.values():
            mean = model["mean"]
            precision = model["precision"]
            diff = x_tab_scaled - mean
            d = np.sum(diff @ precision * diff, axis=1)
            distances.append(d[0])
        return float(np.min(distances))

    def _compute_rarity_score(self, predicted_label: int) -> float:
        count = self.global_counts.get(int(predicted_label))
        if count is None:
            return 0.0
        raw = float(1.0 / np.log1p(count))
        return minmax_transform(raw, float(self.rarity_norm["min"]), float(self.rarity_norm["max"]))

    def predict_from_arrays(self, x_lc: np.ndarray, x_tab: np.ndarray):
        if x_lc.ndim == 2:
            x_lc = x_lc[None, :, :]
        if x_tab.ndim == 1:
            x_tab = x_tab[None, :]

        x_tab_scaled = self.scaler.transform(x_tab).astype(np.float32)
        x_tab_scaled = np.nan_to_num(x_tab_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        x_lc_tensor = torch.from_numpy(x_lc.astype(np.float32)).to(self.device)
        x_tab_tensor = torch.from_numpy(x_tab_scaled.astype(np.float32)).to(self.device)

        with torch.no_grad():
            logits = self.model(x_lc_tensor, x_tab_tensor)
            logits = logits / self.temperature
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

        predicted_label = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities))
        uncertainty = normalized_entropy(probabilities)

        raw_novelty = self._compute_raw_novelty(x_tab_scaled)
        novelty = minmax_transform(raw_novelty, float(self.novelty_norm["min"]), float(self.novelty_norm["max"]))

        rarity = self._compute_rarity_score(predicted_label)
        priority = 0.5 * novelty + 0.5 * rarity

        top_idx = np.argsort(probabilities)[::-1][:5]
        top_classes = [
            {
                "label": int(i),
                "class_name": self.label_to_name.get(int(i), str(i)),
                "probability": float(probabilities[i]),
            }
            for i in top_idx
        ]

        return {
            "predicted_label": predicted_label,
            "predicted_class_name": self.label_to_name.get(predicted_label, str(predicted_label)),
            "confidence": confidence,
            "uncertainty_score": uncertainty,
            "raw_novelty": raw_novelty,
            "novelty_score": novelty,
            "rarity_score": rarity,
            "priority_score": priority,
            "is_predicted_rare": predicted_label in self.rare_classes,
            "top_classes": top_classes,
            "probabilities": probabilities,
        }

    def predict_from_lightcurve(
        self,
        lightcurve_df: pd.DataFrame,
        tabular_features: dict | pd.Series | pd.DataFrame | None = None,
        allow_zero_tabular: bool = False,
    ):
        x_lc = lightcurve_to_tensor(lightcurve_df, n_bins=self.n_bins)
        x_tab, missing_tabular = self.build_tabular_vector(tabular_features, allow_zero_tabular=allow_zero_tabular)
        result = self.predict_from_arrays(x_lc=x_lc, x_tab=x_tab)
        result["missing_tabular_features"] = missing_tabular
        result["used_zero_tabular_preview"] = tabular_features is None and allow_zero_tabular
        return result


_ENGINE = None


def get_inference_engine():
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = AstroTrustInferenceEngine()
    return _ENGINE
