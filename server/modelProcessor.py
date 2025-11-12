
import argparse, json, sys, os, uuid
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model_utils import normalize_df, engineer_factors, score_dataframe_with_saved_model

def aggregate_importance(model):
    try:
        pre = getattr(model, "named_steps", {}).get("pre")
        clf = getattr(model, "named_steps", {}).get("clf")
        if clf is None or not hasattr(clf, "feature_importances_") or pre is None:
            return None
        feat_names = pre.get_feature_names_out()
        importances = clf.feature_importances_
        if len(importances) != len(feat_names):
            return None
        base_cols = []
        for n in feat_names:
            n = str(n)
            if "__" in n:
                group, rest = n.split("__", 1)
                if group == "num":
                    base_cols.append(rest)
                else:
                    base_cols.append(rest.split("_", 1)[0])
            else:
                base_cols.append(n)
        df_imp = (pd.DataFrame({"feature": base_cols, "importance": importances})
                  .groupby("feature", as_index=False)["importance"].sum()
                  .sort_values("importance", ascending=False))
        return df_imp
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--outputs', required=True)
    ap.add_argument('--models', required=True)
    ap.add_argument('--ticket', required=False, default=None)
    args = ap.parse_args()

    models_dir = Path(args.models)
    outputs = Path(args.outputs)
    outputs.mkdir(parents=True, exist_ok=True)
    # prefer an explicitly named canonical model, otherwise try to autodetect a joblib/pkl
    model_path = models_dir / 'nb_model.pkl'
    chosen_model_name = None
    if not model_path.exists():
        # search for common model filename patterns
        candidates = []
        for p in models_dir.glob('*_model.joblib'):
            candidates.append(p)
        for p in models_dir.glob('*.joblib'):
            if p not in candidates:
                candidates.append(p)
        for p in models_dir.glob('*.pkl'):
            if p not in candidates:
                candidates.append(p)
        # prefer seed_model if present, otherwise take the first candidate sorted by name
        pref = None
        for c in candidates:
            if c.name == 'seed_model.joblib':
                pref = c
                break
        if pref is not None:
            model_path = pref
        elif candidates:
            candidates = sorted(candidates, key=lambda x: x.name)
            model_path = candidates[0]
    if model_path.exists():
        chosen_model_name = model_path.name

    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f'Failed to read CSV: {e}', file=sys.stderr)
        sys.exit(1)

    nd = normalize_df(df)
    eng = engineer_factors(nd)

    has_model = model_path.exists()
    if has_model:
        try:
            scored = score_dataframe_with_saved_model(eng, model_path)
        except Exception as e:
            has_model = False
            scored = eng.copy()
    else:
        scored = eng.copy()

    ticket = args.ticket or str(uuid.uuid4())
    out_csv = outputs / f'result_{ticket}.csv'
    scored.to_csv(out_csv, index=False)

    kpis = {"rows": int(len(scored)), "cols": int(scored.shape[1]), "model": chosen_model_name or ("nb_model.pkl" if has_model else "—")}

    hist = None
    if has_model and "_score" in scored.columns:
        vals = scored["_score"].dropna().values
        counts, edges = np.histogram(vals, bins=30)
        hist = {"counts": counts.tolist(), "edges": edges.tolist()}

    topk = None
    if has_model and "_score" in scored.columns and "_key_name" in scored.columns:
        top = scored[["_key_name","_score"]].dropna().sort_values("_score", ascending=False).head(20)
        topk = {"labels": top["_key_name"].tolist(), "values": top["_score"].tolist()}

    sector = None
    if "_sector" in scored.columns:
        vc = scored["_sector"].fillna("Other/Unknown").value_counts().sort_values(ascending=False).head(20)
        sector = {"labels": vc.index.tolist(), "values": vc.values.tolist()}

    scatter = None
    if has_model and "_score" in scored.columns and "_last_round_amount_usd" in scored.columns:
        d = scored[["_score","_last_round_amount_usd"]].dropna().copy()
        if len(d) > 3000:
            d = d.sample(3000, random_state=42)
        scatter = {"x": d["_last_round_amount_usd"].astype(float).tolist(),
                   "y": d["_score"].astype(float).tolist()}

    num_cols = [c for c in ["_total_funding_usd","_last_round_amount_usd","_num_rounds","_num_investors",
                            "_num_articles","_employees","_cb_rank","_days_since_last_round",
                            "_score" if "_score" in scored.columns else None] if c in scored.columns]
    corr = None
    if len(num_cols) >= 2:
        c = scored[num_cols].copy().dropna(how="all")
        if not c.empty:
            mat = c.corr(numeric_only=True)
            corr = {"labels": list(mat.columns), "matrix": mat.values.tolist()}

    importance = None
    if has_model:
        try:
            import joblib
            model = joblib.load(model_path)
            imp = aggregate_importance(model)
            if imp is not None and not imp.empty:
                t = imp.head(12)
                importance = {"labels": t["feature"].tolist(), "values": t["importance"].tolist()}
        except Exception:
            pass

    show_cols = [c for c in ["_key_name","_score","_sector","_location","_last_round_amount_usd",
                              "_num_investors","_num_articles","_cb_rank","_days_since_last_round"]
                 if c in scored.columns]
    preview = scored.head(50)[show_cols].fillna("").to_dict(orient="records") if show_cols else []

    payload = {"ticket": ticket, "kpis": kpis, "hist": hist, "topk": topk, "sector": sector,
               "scatter": scatter, "corr": corr, "importance": importance,
               "preview_columns": show_cols, "preview_rows": preview}
    # Sanitize payload for JSON: convert numpy types and NaN/Inf -> None so JS JSON.parse succeeds.
    def _sanitize(o):
        try:
            import numpy as _np
        except Exception:
            _np = None
        if o is None:
            return None
        if isinstance(o, dict):
            return {k: _sanitize(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_sanitize(v) for v in o]
        # numpy arrays -> lists
        if _np is not None and isinstance(o, (_np.ndarray,)):
            return _sanitize(o.tolist())
        # numbers
        if isinstance(o, float):
            if (o != o) or (o == float('inf')) or (o == float('-inf')):
                return None
            return float(o)
        if isinstance(o, (int,)):
            return int(o)
        # pandas types
        try:
            import pandas as _pd
            if _pd is not None and isinstance(o, (_pd.Series, _pd.DataFrame)):
                return _sanitize(o.to_dict())
        except Exception:
            pass
        return o

    safe = _sanitize(payload)
    print(json.dumps(safe))

if __name__ == "__main__":
    main()
