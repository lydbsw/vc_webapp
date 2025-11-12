# vc_interactive_app/model_utils.py

import os, re, json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from joblib import dump, load

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack, csr_matrix


# -----------------------------------------------------------
# Paths & defaults
# -----------------------------------------------------------
# Prefer an app-local dataset directory unless VC_DATA_DIR is explicitly set.
# This avoids hard-coding '/mnt/data' which may not exist when running locally.
APP_DIR = Path(__file__).parent.resolve()
DATA_DIR = Path(os.environ.get("VC_DATA_DIR")) if os.environ.get("VC_DATA_DIR") else (APP_DIR / "dataset")
# Default models dir lives under the app unless VC_MODEL_DIR overrides it
MODEL_DIR = Path(os.environ.get("VC_MODEL_DIR")) if os.environ.get("VC_MODEL_DIR") else (APP_DIR / "models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_FILES = {
    "preseed": "companies-canada-pre-seed-10-28-2025.csv",
    "seed": "companies-canada-seed-10-28-2025.csv",
    "seriesa": "companies-canada-seriesa-10-28-2025.csv",
    "seriesb": "companies-canada-seriesb-10-28-2025.csv",
    "seriesc": "companies-canada-seriesc-10-28-2025.csv",
}


# -----------------------------------------------------------
# Small utils
# -----------------------------------------------------------
def to_float_safe(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float)): return float(x)
    s = str(x).strip().replace(",", "")
    s = re.sub(r"\$", "", s)
    m = re.match(r"^(-?\d*\.?\d+)([KMB]?)$", s, flags=re.I)
    if m:
        val = float(m.group(1)); suf = m.group(2).upper()
        return val * {"":1, "K":1e3, "M":1e6, "B":1e9}[suf]
    try:
        return float(s)
    except Exception:
        return np.nan


def detect_col(df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
    for pat in patterns:
        rx = re.compile(pat, re.I)
        for c in df.columns:
            if rx.search(str(c)):
                return c
    return None


def infer_unique_id(df):         return detect_col(df, [r"^(id|uuid|cb_?id)$", r"crunchbase.*id"])
def infer_id_name(df):           return detect_col(df, [r"^(name|company.*name)$", r"(org|company).*name", r"permalink|slug"])
def infer_last_round_date(df):   return detect_col(df, [r"last.*fund.*date", r"latest.*round.*date", r"last.*financ.*date"])
def infer_total_funding(df):     return detect_col(df, [r"^total.*fund", r"sum.*fund", r"cumulative.*fund"])
def infer_last_round_amount(df): return detect_col(df, [r"last.*fund.*amount", r"latest.*round.*amount"])
def infer_num_rounds(df):        return detect_col(df, [r"num.*round", r"rounds.*count"])
def infer_num_investors(df):     return detect_col(df, [r"num.*investor", r"investor.*count"])
def infer_num_articles(df):      return detect_col(df, [r"num.*article", r"press|news.*count", r"media.*count"])
def infer_employees(df):         return detect_col(df, [r"(employee|staff|headcount)", r"num.*employee"])
def infer_cb_rank(df):           return detect_col(df, [r"crunchbase.*rank", r"cb.*rank"])
def infer_location(df):          return detect_col(df, [r"province|state|region", r"location", r"headquarters.*(province|state)"])
def infer_sector(df):            return detect_col(df, [r"category|industry|sector", r"tag.*industry"])


# Robust exit field inference → return LISTS (we OR them together)
def infer_exit_cols(df: pd.DataFrame):
    acq_cols = [c for c in df.columns if re.search(r"acquir|acquisition|acquired[_ ]?by|acquired[_ ]?on|acq[_ ]?date", str(c), re.I)]
    ipo_cols = [c for c in df.columns if re.search(r"\bipo\b|went[_ ]?public|public[_ ]?date|ipo[_ ]?date", str(c), re.I)]
    exit_date_cols = [c for c in df.columns if re.search(r"exit[_ ]?date|acquis[_ ]?date|ipo[_ ]?date", str(c), re.I)]
    status_cols = [c for c in df.columns if re.search(r"(operat(ing)?[_ ]?)?status|company[_ ]?status", str(c), re.I)]
    # common explicit headers
    acq_cols += [c for c in df.columns if c in {"acquired_on","acquirer","acquisition_date"}]
    ipo_cols += [c for c in df.columns if c in {"ipo_date","went_public_on"}]
    status_cols += [c for c in df.columns if c in {"operating_status","status"}]
    return acq_cols, ipo_cols, exit_date_cols, status_cols


def _to_bool_like(x: object) -> bool:
    if x is None or (isinstance(x, float) and pd.isna(x)): return False
    s = str(x).strip().lower()
    if s in {"true","1","yes","y","t"}: return True
    if s in {"false","0","no","n","f",""}: return False
    if any(k in s for k in ["acquir","ipo","went public","public","exit"]): return True
    return False


def _contains_any(s: str, words) -> bool:
    if s is None: return False
    s = str(s).lower()
    return any(w in s for w in words)


def _norm_name(s: str) -> str:
    if s is None: return ""
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    s = re.sub(r"\b(incorporated|inc|corp|corporation|ltd|limited|gmbh|sa|sas|bv|oy|ab)\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _usable_norm_name(s: str) -> str:
    n = _norm_name(s)
    if len(n) >= 3 and not n.isdigit():
        return n
    return ""


# -----------------------------------------------------------
# Normalization & features
# -----------------------------------------------------------
def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    out = df.copy()

    id_col = infer_unique_id(out)
    name_col = infer_id_name(out)
    last_dt_col = infer_last_round_date(out)
    total_f_col = infer_total_funding(out)
    last_amt_col = infer_last_round_amount(out)
    n_rounds_col = infer_num_rounds(out)
    n_inv_col = infer_num_investors(out)
    n_art_col = infer_num_articles(out)
    emp_col = infer_employees(out)
    cb_rank_col = infer_cb_rank(out)
    loc_col = infer_location(out)
    sec_col = infer_sector(out)

    acq_cols, ipo_cols, exit_dt_cols, status_cols = infer_exit_cols(out)

    out["_last_round_date"] = pd.to_datetime(out[last_dt_col], errors="coerce") if last_dt_col else pd.NaT
    out["_total_funding_usd"] = out[total_f_col].apply(to_float_safe) if total_f_col else np.nan
    out["_last_round_amount_usd"] = out[last_amt_col].apply(to_float_safe) if last_amt_col else np.nan
    out["_num_rounds"] = pd.to_numeric(out[n_rounds_col], errors="coerce") if n_rounds_col else np.nan
    out["_num_investors"] = pd.to_numeric(out[n_inv_col], errors="coerce") if n_inv_col else np.nan
    out["_num_articles"] = pd.to_numeric(out[n_art_col], errors="coerce") if n_art_col else 0.0
    out["_employees"] = pd.to_numeric(out[emp_col], errors="coerce") if emp_col else np.nan
    out["_cb_rank"] = pd.to_numeric(out[cb_rank_col], errors="coerce") if cb_rank_col else np.nan
    out["_location"] = out[loc_col].astype(str) if loc_col else "Unknown"
    out["_sector_raw"] = out[sec_col].astype(str) if sec_col else "Unknown"

    # exit flags (robust OR over many columns)
    acq_flag = pd.Series(False, index=out.index)
    for c in acq_cols:    acq_flag |= out[c].apply(_to_bool_like)
    for c in status_cols: acq_flag |= out[c].astype(str).apply(lambda s: _contains_any(s, ["acquir","acquisition"]))

    ipo_flag = pd.Series(False, index=out.index)
    for c in ipo_cols:    ipo_flag |= out[c].apply(_to_bool_like)
    for c in status_cols: ipo_flag |= out[c].astype(str).apply(lambda s: _contains_any(s, ["ipo","went public","public"]))

    exit_dt_flag = pd.Series(False, index=out.index)
    for c in exit_dt_cols:
        exit_dt_flag |= pd.to_datetime(out[c], errors="coerce").notna()

    out["_acquired_flag"] = acq_flag.astype(bool)
    out["_ipo_flag"] = ipo_flag.astype(bool)
    out["_exitdate_flag"] = exit_dt_flag.astype(bool)

    if id_col is None and name_col is None:
        out["_key_id"] = np.arange(len(out)).astype(str)
        out["_key_name"] = out["_key_id"]
    else:
        out["_key_id"] = out[id_col].astype(str) if id_col else out.index.astype(str)
        out["_key_name"] = out[name_col].astype(str) if name_col else out["_key_id"]

    return out


def simplify_sector(x: str) -> str:
    s = str(x).lower()
    maps = {
        "AI/ML": r"ai|ml|artificial",
        "Fintech": r"fintech|finance|bank|payment",
        "Health/Bio": r"health|bio|med|pharma",
        "E-Commerce/Retail": r"e-?com|retail|marketplace",
        "Software/Data": r"saas|software|cloud|devops|data",
        "Gaming": r"game|gaming",
        "Climate/Energy": r"clean|climate|energy",
        "Hardware/IoT": r"hardware|robot|iot",
    }
    for lab, rx in maps.items():
        if re.search(rx, s): return lab
    return "Other/Unknown"


def engineer_factors(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    out = df.copy()
    out["_sector"] = out["_sector_raw"].apply(simplify_sector)
    ref_date = pd.Timestamp("2025-10-28")
    out["_days_since_last_round"] = (ref_date - out["_last_round_date"]).dt.days.clip(lower=0)
    # preserve a single descriptive/text column if present so TF-IDF can be used in the pipeline
    text_candidates = [
        "description", "about", "bio", "summary", "long_description", "short_description", "full description",
    ]
    text_col = None
    for c in text_candidates:
        if c in out.columns:
            text_col = c
            break

    keep = [
        "_key_id","_key_name","_last_round_date","_sector","_location",
        "_total_funding_usd","_last_round_amount_usd","_num_rounds",
        "_num_investors","_num_articles","_employees","_cb_rank",
        "_days_since_last_round",
    ]
    if text_col:
        # keep original column name and also populate a canonical '_text' column
        keep.append(text_col)
        out["_text"] = out[text_col].astype(str).fillna("")
    return out[keep]


# -----------------------------------------------------------
# Labeling (strict/loose linking; optional proxy)
# -----------------------------------------------------------
def label_success_per_stage(frames: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    later_keys = {
        "preseed": {"seed","seriesa","seriesb","seriesc"},
        "seed": {"seriesa","seriesb","seriesc"},
        "seriesa": {"seriesb","seriesc"},
        "seriesb": {"seriesc"},
        "seriesc": set(),
    }
    earlier_keys = {
        "preseed": set(),
        "seed": {"preseed"},
        "seriesa": {"preseed","seed"},
        "seriesb": {"preseed","seed","seriesa"},
        "seriesc": {"preseed","seed","seriesa","seriesb"},
    }

    def build_lookup(stages):
        ids, names = set(), set()
        for s in stages:
            df = frames.get(s, pd.DataFrame())
            if df.empty: continue
            ids |= set(df["_key_id"].astype(str))
            names |= { _usable_norm_name(v) for v in df["_key_name"].astype(str) }
        names.discard("")
        return ids, names

    later_lookup   = {st: build_lookup(v) for st, v in later_keys.items()}
    earlier_lookup = {st: build_lookup(v) for st, v in earlier_keys.items()}

    out = {}
    mode = os.getenv("VC_LINK_MODE","strict")

    for st, df in frames.items():
        if df.empty:
            out[st] = df
            continue

        later_ids,   later_names   = later_lookup[st]
        earlier_ids, earlier_names = earlier_lookup[st]

        nn = df["_key_name"].astype(str).map(_usable_norm_name)

        # STRICT name matches; also require some metadata to reduce collisions
        strict_later   = nn.isin(later_names)
        strict_earlier = nn.isin(earlier_names)
        if "_location" in df.columns:
            strict_later   = strict_later   & df["_location"].astype(str).ne("")
            strict_earlier = strict_earlier & df["_location"].astype(str).ne("")
        if "_sector_raw" in df.columns:
            strict_later   = strict_later   | (nn.isin(later_names)   & df["_sector_raw"].astype(str).ne(""))
            strict_earlier = strict_earlier | (nn.isin(earlier_names) & df["_sector_raw"].astype(str).ne(""))

        name_later   = nn.isin(later_names)   if mode == "loose" else strict_later
        name_earlier = nn.isin(earlier_names) if mode == "loose" else strict_earlier

        suc = (
            df["_key_id"].astype(str).isin(later_ids) |
            name_later |
            df.get("_ipo_flag", False) |
            df.get("_acquired_flag", False) |
            df.get("_exitdate_flag", False)
        )

        # Optional proxy positives to avoid single-class during demos
        if os.getenv("VC_ALLOW_PROXY_LABELS", "0") == "1":
            proxy = (
                (df.get("_num_investors", 0).fillna(0) >= 3) |
                (df.get("_num_articles", 0).fillna(0) >= 2)
            )
            if suc.sum() == 0:
                suc = proxy
            else:
                suc = suc | (proxy & ~suc)

        cold = ~(
            df["_key_id"].astype(str).isin(earlier_ids) |
            name_earlier
        )

        cp = df.copy()
        cp["_success"] = suc.astype(int)
        cp["_cold_start_from_this_stage"] = cold.astype(int)
        out[st] = cp

    return out


# -----------------------------------------------------------
# Modeling table
# -----------------------------------------------------------
def make_modeling_frame(df: pd.DataFrame):
    """Return modeling frame plus lists of numeric, categorical and text columns.
    Returns (model_df, present_num, present_cat, text_cols)
    """
    if df.empty: return df, [], [], []
    dfe = engineer_factors(df)

    num = [
        "_total_funding_usd","_last_round_amount_usd","_num_rounds",
        "_num_investors","_num_articles","_employees","_cb_rank",
        "_days_since_last_round",
    ]
    cat = ["_sector","_location"]
    present_num = [c for c in num if c in dfe.columns]
    present_cat = [c for c in cat if c in dfe.columns]
    text_cols = [c for c in ["_text"] if c in dfe.columns]
    feats = present_num + present_cat + text_cols

    lbl = df[["_key_id","_success"]].copy() if "_success" in df.columns else df[["_key_id"]].assign(_success=np.nan)
    out = dfe.merge(lbl, on="_key_id", how="left").dropna(subset=["_success"])

    # --- Fallback: if we ended up with zero features, add a bias feature so the model can still train
    if len(present_num) + len(present_cat) == 0:
        dfe["_bias"] = 1.0
        present_num = ["_bias"]
        feats = present_num + text_cols
        out = dfe.merge(lbl, on="_key_id", how="left").dropna(subset=["_success"])

    return out[["_success"] + feats], present_num, present_cat, text_cols


# -----------------------------------------------------------
# Pipeline
# -----------------------------------------------------------
def _onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_pipeline(num_cols, cat_cols, text_col: Optional[str] = None):
    """Build preprocessing + classifier pipeline. If text_col is provided (single column name),
    a TF-IDF transformer will be applied to that column.
    """
    transformers = []
    if num_cols:
        num_tf = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
        transformers.append(("num", num_tf, num_cols))
    if cat_cols:
        cat_tf = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", _onehot_encoder())])
        transformers.append(("cat", cat_tf, cat_cols))
    if text_col:
        # TfidfVectorizer expects 1d input; ColumnTransformer will pass the single column
        text_tf = Pipeline([("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1,2), min_df=2))])
        transformers.append(("text", text_tf, text_col))

    pre = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)

    # Use a calibrated logistic regression which works with sparse TF-IDF outputs
    base = LogisticRegression(solver="saga", penalty="elasticnet", l1_ratio=0.2, C=1.0, max_iter=300)
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    return Pipeline([("pre", pre), ("clf", clf)])


# -----------------------
# Component / stacking helpers
# -----------------------
def _quantile_scale(s: pd.Series, q=0.98):
    if s.isna().all():
        return pd.Series(0.0, index=s.index)
    vmax = np.nanpercentile(s.fillna(0), q*100)
    if vmax == 0:
        return (s.fillna(0) > 0).astype(float)
    return (s.fillna(0) / float(vmax)).clip(0,1)


def build_block_features(df: pd.DataFrame, num_cols: List[str], cat_cols: List[str], text_col: Optional[str]=None, tfidf_max: int = 2000):
    """Build a sparse feature matrix (csr) for component models using provided column lists.
    Returns X (csr_matrix) and a fitted preprocessor (ColumnTransformer-like structure wrapped as Pipeline).
    """
    if df.empty:
        return csr_matrix((0,0)), None

    parts = []
    # numeric
    if num_cols:
        num_df = df[num_cols].copy()
        num_df = num_df.fillna(num_df.median())
        # standardize without centering for sparse
        sc = StandardScaler()
        try:
            num_arr = sc.fit_transform(num_df.values)
        except Exception:
            # fallback: no scaling
            num_arr = num_df.values
        parts.append(csr_matrix(num_arr))

    # categorical (one-hot)
    if cat_cols:
        cat_df = df[cat_cols].astype(str).fillna("")
        ohe = _onehot_encoder()
        try:
            cat_mat = ohe.fit_transform(cat_df)
        except Exception:
            # if dense output, convert to csr
            cat_mat = csr_matrix(ohe.fit_transform(cat_df))
        if not isinstance(cat_mat, csr_matrix):
            cat_mat = csr_matrix(cat_mat)
        parts.append(cat_mat)

    # text
    if text_col and text_col in df.columns:
        tv = TfidfVectorizer(max_features=tfidf_max, ngram_range=(1,2), min_df=2)
        txt = df[text_col].fillna("").astype(str).tolist()
        try:
            txt_mat = tv.fit_transform(txt)
        except Exception:
            txt_mat = csr_matrix((len(txt), 0))
        parts.append(txt_mat)

    if len(parts) == 0:
        return csr_matrix((len(df),1)), None

    X = parts[0]
    for p in parts[1:]:
        X = hstack([X, p], format="csr")
    return X, None


def fit_fast_prob(X, y, random_state=42):
    """Train a calibrated logistic classifier quickly and return the fitted estimator and metrics.
    """
    if isinstance(X, pd.DataFrame):
        Xmat = csr_matrix(X.values)
    else:
        Xmat = X

    if len(y.unique()) < 2:
        return None, None

    try:
        Xtr, Xte, ytr, yte = train_test_split(Xmat, y, test_size=0.2, stratify=y, random_state=random_state)
    except Exception:
        # fallback split
        n = len(y)
        idx = np.arange(n)
        np.random.seed(random_state)
        np.random.shuffle(idx)
        cut = int(n*0.8)
        Xtr, Xte = Xmat[idx[:cut]], Xmat[idx[cut:]]
        ytr, yte = y.iloc[idx[:cut]], y.iloc[idx[cut:]]

    base = LogisticRegression(solver="saga", penalty="elasticnet", l1_ratio=0.2, C=1.0, max_iter=400)
    clf = CalibratedClassifierCV(base, cv=3, method="isotonic")
    clf.fit(Xtr, ytr)

    proba = clf.predict_proba(Xte)[:,1] if hasattr(clf, "predict_proba") else clf.decision_function(Xte)
    metrics = {
        "roc_auc": float(roc_auc_score(yte, proba)),
        "avg_precision": float(average_precision_score(yte, proba)),
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "pos_rate_train": float(ytr.mean()),
        "pos_rate_test": float(yte.mean()),
    }
    return clf, metrics


def train_components_for_stage(df: pd.DataFrame, stage_key: str):
    """Train funding/brand/operations component models for a stage and persist them.
    Saves component models as {stage_key}_component_{name}_model.joblib and meta JSONs.
    """
    if df.empty:
        return {}

    out = {}
    dfe = engineer_factors(df)

    # define simple proxies/labels for components
    funding_label = (_quantile_scale(dfe.get("_total_funding_usd", pd.Series(0))) > 0.1).astype(int)
    brand_label = (_quantile_scale(dfe.get("_num_articles", pd.Series(0))) > 0.0).astype(int)
    ops_label = (_quantile_scale(dfe.get("_employees", pd.Series(0))) > 0.0).astype(int)

    # common candidate columns
    num = [c for c in ["_total_funding_usd","_last_round_amount_usd","_num_rounds",
                       "_num_investors","_num_articles","_employees","_cb_rank",
                       "_days_since_last_round"] if c in dfe.columns]
    cat = [c for c in ["_sector","_location"] if c in dfe.columns]
    text_col = "_text" if "_text" in dfe.columns else None

    comps = {
        "funding": (funding_label, ["_total_funding_usd","_num_rounds","_num_investors"], ["_sector"], text_col),
        "brand": (brand_label, ["_num_articles","_cb_rank"], ["_sector"], text_col),
        "operations": (ops_label, ["_employees","_cb_rank"], ["_location"], None),
    }

    for cname, (label, ncols, ccols, tcol) in comps.items():
        cols_num = [c for c in ncols if c in dfe.columns]
        cols_cat = [c for c in ccols if c in dfe.columns]
        if len(cols_num) + len(cols_cat) + (1 if tcol else 0) == 0:
            out[cname] = {"status": "skipped_no_features"}
            continue

        X, _ = build_block_features(dfe, cols_num, cols_cat, tcol)
        y = label
        if y.nunique() < 2:
            out[cname] = {"status": "skipped_single_class"}
            continue

        clf, metrics = fit_fast_prob(X, y)
        if clf is None:
            out[cname] = {"status": "failed_train"}
            continue

        fname = MODEL_DIR / f"{stage_key}_component_{cname}_model.joblib"
        dump(clf, fname)
        # store meta about which cols we used for this component
        meta = {"num_cols": cols_num, "cat_cols": cols_cat, "text_col": tcol}
        (MODEL_DIR / f"{stage_key}_component_{cname}_meta.json").write_text(json.dumps(meta), encoding="utf-8")
        out[cname] = {"metrics": metrics, "meta": meta, "model_file": str(fname)}

    return out


def train_meta_learner(df: pd.DataFrame, stage_key: str, component_names: List[str] = ["funding","brand","operations"]):
    """Train a stacked meta-learner that uses component model probabilities as inputs.
    Saves stacker to {stage_key}_stacker_model.joblib and {stage_key}_stacker_meta.json
    """
    if df.empty:
        return {"status": "no_data"}

    dfe = engineer_factors(df)
    # load component models and their meta; compute probs on the same df
    comp_feats = []
    comp_preds = {}
    for cname in component_names:
        mfile = MODEL_DIR / f"{stage_key}_component_{cname}_model.joblib"
        mmeta = MODEL_DIR / f"{stage_key}_component_{cname}_meta.json"
        if not mfile.exists() or not mmeta.exists():
            continue
        clf = load(mfile)
        meta = json.loads(mmeta.read_text(encoding="utf-8"))
        cols = meta.get("num_cols", []) + meta.get("cat_cols", []) + ([meta.get("text_col")] if meta.get("text_col") else [])
        for c in cols:
            if c not in dfe.columns: dfe[c] = np.nan
        Xcomp, _ = build_block_features(dfe, meta.get("num_cols", []), meta.get("cat_cols", []), meta.get("text_col"))
        try:
            probs = clf.predict_proba(Xcomp)[:,1]
        except Exception:
            # fallback: zeros
            probs = np.zeros(len(dfe))
        comp_preds[cname] = probs
        comp_feats.append(cname)

    if len(comp_feats) == 0:
        return {"status": "no_components"}

    Xmeta = pd.DataFrame({k: v for k, v in comp_preds.items()})
    if "_success" not in df.columns:
        return {"status": "no_labels_for_meta"}
    y = df["_success"].astype(int)

    # train meta classifier
    clf_meta, metrics = fit_fast_prob(csr_matrix(Xmeta.values), y)
    if clf_meta is None:
        return {"status": "meta_failed"}

    fname = MODEL_DIR / f"{stage_key}_stacker_model.joblib"
    dump(clf_meta, fname)
    meta = {"component_order": comp_feats}
    (MODEL_DIR / f"{stage_key}_stacker_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return {"metrics": metrics, "meta": meta, "model_file": str(fname)}


# -----------------------------------------------------------
# Training
# -----------------------------------------------------------
def train_stage(df: pd.DataFrame, stage_key: str):
    if df.empty:
        if os.getenv("VC_DEBUG","0") == "1":
            print(f"[DEBUG] {stage_key}: dataframe is EMPTY after normalization.")
        return None, None

    dfx, num_cols, cat_cols, text_cols = make_modeling_frame(df)
    if dfx.empty:
        if os.getenv("VC_DEBUG","0") == "1":
            print(f"[DEBUG] {stage_key}: modeling frame is EMPTY (no features or no labels).")
        return None, None

    # select columns for X; include text column name if present (text_cols is list)
    select_cols = list(num_cols) + list(cat_cols) + (text_cols if text_cols else [])
    X = dfx[select_cols]
    y = dfx["_success"].astype(int)

    if os.getenv("VC_DEBUG","0") == "1":
        print(f"[DEBUG] {stage_key}: rows={len(dfx)}  y_counts={y.value_counts().to_dict()}")
        missing = [c for c in (num_cols+cat_cols) if c not in dfx.columns]
        print(f"[DEBUG] {stage_key}: num_cols={len(num_cols)} cat_cols={len(cat_cols)} missing_feats={missing}")

    # overall label diversity check
    if y.nunique() < 2:
        if os.getenv("VC_DEBUG","0") == "1":
            print(f"[DEBUG] {stage_key}: SKIP REASON -> single-class labels overall.")
        return None, None

    # decide split strategy
    minority = y.value_counts().min()
    if minority < 10:
        use_time_split = False
    else:
        dates = df["_last_round_date"]
        use_time_split = dates.notna().sum() > 0.5 * len(df)

    X_train = X_test = y_train = y_test = None

    if use_time_split:
        years = df["_last_round_date"].dt.year
        thresh = np.nanpercentile(years.dropna(), 80)
        mask = years <= thresh
        X_train, X_test, y_train, y_test = X[mask], X[~mask], y[mask], y[~mask]
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            if os.getenv("VC_DEBUG","0") == "1":
                print(f"[DEBUG] {stage_key}: time-split lost a class -> falling back to stratified.")
            X_train = X_test = y_train = y_test = None

    if X_train is None:
        from sklearn.model_selection import train_test_split
        test_size = 0.2 if minority >= 5 else 0.15
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=42
            )
        except ValueError:
            if os.getenv("VC_DEBUG","0") == "1":
                print(f"[DEBUG] {stage_key}: SKIP REASON -> stratified split failed.")
            return None, None

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        if os.getenv("VC_DEBUG","0") == "1":
            print(f"[DEBUG] {stage_key}: SKIP REASON -> split still single-class.")
        return None, None

    pipe = build_pipeline(num_cols, cat_cols, text_cols[0] if text_cols else None)
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe.named_steps["clf"], "predict_proba") else pipe.decision_function(X_test)
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "avg_precision": float(average_precision_score(y_test, proba)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "pos_rate_train": float(y_train.mean()),
        "pos_rate_test": float(y_test.mean()),
    }

    dump(pipe, MODEL_DIR / f"{stage_key}_model.joblib")
    meta = {"num_cols": num_cols, "cat_cols": cat_cols, "text_cols": text_cols}
    (MODEL_DIR / f"{stage_key}_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return pipe, metrics


# -----------------------------------------------------------
# IO helpers
# -----------------------------------------------------------
def load_raw_frames(files: Dict[str, str] = None) -> Dict[str, pd.DataFrame]:
    files = files or DEFAULT_FILES
    frames = {}
    for st, fname in files.items():
        p = DATA_DIR / fname
        if p.exists():
            df = pd.read_csv(p)
            df["__stage__"] = st
            frames[st] = normalize_df(df)
        else:
            frames[st] = pd.DataFrame()
    return frames


def load_model(stage_key: str):
    p = MODEL_DIR / f"{stage_key}_model.joblib"
    m = MODEL_DIR / f"{stage_key}_meta.json"
    if p.exists() and m.exists():
        return load(p), json.loads(m.read_text(encoding="utf-8"))
    return None, None


def available_models(model_dir: str = "./models"):
    md = Path(model_dir)
    if not md.exists(): return []
    return sorted([p.stem.replace("_model","") for p in md.glob("*_model.joblib")])


# -----------------------------------------------------------
# Scoring / outputs for UI
# -----------------------------------------------------------
def score_stage_to_csv(df: pd.DataFrame, stage_key: str, out_dir: str = "./outputs"):
    pipe, meta = load_model(stage_key)
    if pipe is None or meta is None or df.empty: return False

    engineered = engineer_factors(df)
    feats = meta.get("num_cols", []) + meta.get("cat_cols", []) + meta.get("text_cols", [])
    for c in feats:
        if c not in engineered.columns: engineered[c] = np.nan

    X = engineered[feats]
    scores = pipe.predict_proba(X)[:, 1] if hasattr(pipe.named_steps["clf"], "predict_proba") else pipe.decision_function(X)

    out = df.copy()
    out["_score"] = scores
    if "_sector" not in out.columns:   out["_sector"] = engineered.get("_sector", "Other/Unknown")
    if "_location" not in out.columns: out["_location"] = engineered.get("_location", "Unknown")

    cols = ["_key_name","_score","_success","_sector","_location",
            "_last_round_amount_usd","_num_investors","_num_articles","_cb_rank"]
    cols = [c for c in cols if c in out.columns]
    Path(out_dir).mkdir(exist_ok=True)
    out[cols].to_csv(Path(out_dir) / f"scored_{stage_key}.csv", index=False)
    return True


def train_all(files: Dict[str, str] = None):
    frames = load_raw_frames(files)
    frames = label_success_per_stage(frames)

    if os.getenv("VC_DEBUG","0") == "1":
        for st, d in frames.items():
            if d.empty:
                print(f"[DEBUG] {st}: EMPTY after labeling")
            else:
                print(f"[DEBUG] {st}: label counts -> {d['_success'].value_counts(dropna=False).to_dict()}")

    report = {}
    for st, df in frames.items():
        pipe, metrics = train_stage(df, st)
        if metrics:
            report[st] = metrics
            ok = score_stage_to_csv(df, st, out_dir="./outputs")
            report[st]["scored_csv"] = f"written to ./outputs/scored_{st}.csv" if ok else "not written"
        else:
            report[st] = {"status": "skipped (insufficient label diversity or data)"}
        # train component models for this stage
        try:
            comps = train_components_for_stage(df, st)
            report[st]["components"] = comps
        except Exception as e:
            report[st]["components"] = {"status": "error", "error": str(e)}

        # train meta-learner / stacker
        try:
            stack = train_meta_learner(df, st)
            report[st]["stacker"] = stack
        except Exception as e:
            report[st]["stacker"] = {"status": "error", "error": str(e)}

    return report


# -----------------------------------------------------------
# Score user-uploaded CSV (robust, with fallback)
# -----------------------------------------------------------
def score_user_file(user_df: pd.DataFrame, stage_key: str, fallback_to_any: bool = True) -> pd.DataFrame:
    nd = normalize_df(user_df)
    engineered = engineer_factors(nd)

    pipe, meta = load_model(stage_key)
    if pipe is None or meta is None:
        train_all()  # try once
        pipe, meta = load_model(stage_key)

    if (pipe is None or meta is None) and fallback_to_any:
        avail = available_models()
        if len(avail) > 0:
            preferred = [s for s in ["seed","seriesa","seriesb","seriesc","preseed"] if s in avail]
            use_stage = preferred[0] if preferred else avail[0]
            pipe, meta = load_model(use_stage)
            stage_key = use_stage
        else:
            raise ValueError("No trained models are available. Check dataset and restart the app.")

    if pipe is None or meta is None:
        raise ValueError(f"No model trained for stage '{stage_key}'. It was likely skipped due to label sparsity.")

    feats = meta.get("num_cols", []) + meta.get("cat_cols", []) + meta.get("text_cols", [])
    for c in feats:
        if c not in engineered.columns: engineered[c] = np.nan

    X = engineered[feats]
    # First attempt: if a stacked meta model exists, use component models -> stacker
    stack_file = MODEL_DIR / f"{stage_key}_stacker_model.joblib"
    stack_meta_file = MODEL_DIR / f"{stage_key}_stacker_meta.json"
    out = nd.copy()

    if stack_file.exists() and stack_meta_file.exists():
        stacker = load(stack_file)
        stack_meta = json.loads(stack_meta_file.read_text(encoding="utf-8"))
        comp_order = stack_meta.get("component_order", [])
        comp_scores = {}
        for cname in comp_order:
            cmeta_file = MODEL_DIR / f"{stage_key}_component_{cname}_meta.json"
            cmodel_file = MODEL_DIR / f"{stage_key}_component_{cname}_model.joblib"
            if not cmeta_file.exists() or not cmodel_file.exists():
                comp_scores[cname] = np.zeros(len(engineered))
                continue
            cm = json.loads(cmeta_file.read_text(encoding="utf-8"))
            cols = cm.get("num_cols", []) + cm.get("cat_cols", []) + ([cm.get("text_col")] if cm.get("text_col") else [])
            for c in cols:
                if c not in engineered.columns: engineered[c] = np.nan
            Xcomp, _ = build_block_features(engineered, cm.get("num_cols", []), cm.get("cat_cols", []), cm.get("text_col"))
            clf = load(cmodel_file)
            try:
                probs = clf.predict_proba(Xcomp)[:,1]
            except Exception:
                probs = np.zeros(len(engineered))
            comp_scores[cname] = probs

        Xmeta = pd.DataFrame(comp_scores)
        try:
            final = stacker.predict_proba(csr_matrix(Xmeta.values))[:,1]
        except Exception:
            final = stacker.predict_proba(Xmeta.values)[:,1] if hasattr(stacker, "predict_proba") else stacker.decision_function(Xmeta.values)

        out["_score"] = final
        # also attach component scores
        for k, v in comp_scores.items():
            out[f"_comp_{k}"] = v
        out["_scored_with_stage"] = stage_key
        out["_sector"] = engineered.get("_sector", "Other/Unknown")
        out["_location"] = engineered.get("_location", "Unknown")
        return out

    # Fallback: use the single-stage pipeline
    try:
        scores = pipe.predict_proba(X)[:, 1] if hasattr(pipe.named_steps["clf"], "predict_proba") else pipe.decision_function(X)
    except Exception:
        # fallback to zeros
        scores = np.zeros(len(X))

    out["_score"] = scores
    out["_sector"] = engineered.get("_sector", "Other/Unknown")
    out["_location"] = engineered.get("_location", "Unknown")
    out["_scored_with_stage"] = stage_key
    return out


def score_dataframe_with_saved_model(engineered_df: pd.DataFrame, model_path: str) -> pd.DataFrame:
    """Load a saved model (joblib/pickle) and score an already-engineered dataframe.

    This is a thin compatibility wrapper used by the legacy `modelProcessor.py` which expects
    a single model file named `nb_model.pkl`. The function will attempt to load the
    model with joblib and compute a `_score` column on `engineered_df`. If anything
    fails it will return the original dataframe unchanged.
    """
    try:
        import joblib
        mp = Path(model_path)
        if not mp.exists():
            return engineered_df.copy()
        model = joblib.load(mp)
        out = engineered_df.copy()
        # try predict_proba first, then decision_function, else predict
        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(out)[:, 1]
            elif hasattr(model, "decision_function"):
                probs = model.decision_function(out)
            else:
                preds = model.predict(out)
                probs = preds
        except Exception:
            # some saved models are pipelines that expect DataFrame columns in a particular order
            # try converting to numpy matrix as a fallback
            try:
                arr = csr_matrix(out.values) if not isinstance(out, csr_matrix) else out
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(arr)[:, 1]
                elif hasattr(model, "decision_function"):
                    probs = model.decision_function(arr)
                else:
                    preds = model.predict(arr)
                    probs = preds
            except Exception:
                return engineered_df.copy()

        out["_score"] = np.array(probs)
        return out
    except Exception:
        return engineered_df.copy()
