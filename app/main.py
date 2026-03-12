"""
Visit:    http://localhost:5000

"""

import os, sys, pickle, json, sqlite3, datetime
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for
from dotenv import load_dotenv

load_dotenv()   # reads .env file automatically

# ── Setup paths ── #
ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MDL_DIR = os.path.join(ROOT, "models")
DB_PATH = os.path.join(ROOT, "predictions.db")

app = Flask(__name__)

CAT_COLS = ["BusinessTravel","Department","EducationField",
            "Gender","JobRole","MaritalStatus","OverTime"]

# ── Database setup ── #
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT,
            age           INTEGER,
            department    TEXT,
            job_role      TEXT,
            monthly_income INTEGER,
            overtime      TEXT,
            sk_prob       REAL,
            tf_prob       REAL,
            pt_prob       REAL,
            avg_prob      REAL,
            prediction    TEXT,
            risk_level    TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_prediction(data):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO predictions
        (timestamp,age,department,job_role,monthly_income,overtime,
         sk_prob,tf_prob,pt_prob,avg_prob,prediction,risk_level)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        data["age"], data["department"], data["job_role"],
        data["monthly_income"], data["overtime"],
        data["sk_prob"], data["tf_prob"], data["pt_prob"],
        data["avg_prob"], data["prediction"], data["risk_level"]
    ))
    conn.commit()
    conn.close()

def get_history():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT 50"
    ).fetchall()
    conn.close()
    return rows

def get_stats():
    conn = sqlite3.connect(DB_PATH)
    total   = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    leaving = conn.execute("SELECT COUNT(*) FROM predictions WHERE prediction='Leave'").fetchone()[0]
    conn.close()
    return total, leaving

def get_recent(n=5):
    """Fetch the last N predictions for the dashboard activity log."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (n,)
    ).fetchall()
    conn.close()
    return rows

# ── Model loading ── #
_cache = {}

def _pkl(name):
    if name not in _cache:
        with open(os.path.join(MDL_DIR, name), "rb") as f:
            _cache[name] = pickle.load(f)
    return _cache[name]

def preprocess(data: dict) -> np.ndarray:
    df  = pd.DataFrame([data])
    enc = _pkl("label_encoders.pkl")
    for col in CAT_COLS:
        if col in df.columns and col in enc:
            df[col] = enc[col].transform(df[col].astype(str))
    df = df.reindex(columns=_pkl("feature_columns.pkl"), fill_value=0)
    return _pkl("scaler.pkl").transform(df)

def run_all_models(X):
    import torch, torch.nn as nn, tensorflow as tf

    # scikit-learn
    sk = float(_pkl("sklearn_model.pkl").predict_proba(X)[0][1])

    # TensorFlow
    if "tf_model" not in _cache:
        _cache["tf_model"] = tf.keras.models.load_model(
            os.path.join(MDL_DIR, "tf_model"))
    tf_prob = float(_cache["tf_model"].predict(X, verbose=0)[0][0])

    # PyTorch
    if "pt_model" not in _cache:
        class AttritionNet(nn.Module):
            def __init__(self, d):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(d,256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.4),
                    nn.Linear(256,128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.3),
                    nn.Linear(128,64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.2),
                    nn.Linear(64,32), nn.GELU(), nn.Linear(32,1), nn.Sigmoid())
            def forward(self, x): return self.net(x)
        ckpt = torch.load(os.path.join(MDL_DIR,"pytorch_model.pth"), map_location="cpu")
        m = AttritionNet(ckpt["input_dim"])
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        _cache["pt_model"] = m

    with torch.no_grad():
        pt = float(_cache["pt_model"](torch.FloatTensor(X))[0][0])

    avg  = (sk + tf_prob + pt) / 3
    pred = "Leave" if avg >= 0.5 else "Stay"
    risk = "High" if avg >= 0.7 else "Medium" if avg >= 0.4 else "Low"
    return sk, tf_prob, pt, avg, pred, risk

# ══════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════

@app.route("/")
def home():
    total, leaving = get_stats()
    staying = total - leaving
    rate = round(leaving / total * 100, 1) if total else 0
    sk = {}
    tf = {}
    pt = {}
    try:
        sk = _pkl("sklearn_metrics.pkl")
        tf = _pkl("tf_metrics.pkl")
        pt = _pkl("pytorch_metrics.pkl")
    except Exception:
        pass
    recent = get_recent()
    return render_template("home.html", total=total, leaving=leaving,
        staying=staying, rate=rate, sk=sk, tf=tf, pt=pt,
        recent=recent)


@app.route("/predict", methods=["GET","POST"])
def predict():
    result = None
    if request.method == "POST":
        form = request.form
        employee = {
            "Age":            int(form.get("Age", 30)),
            "Gender":         form.get("Gender","Male"),
            "MaritalStatus":  form.get("MaritalStatus","Single"),
            "DistanceFromHome": int(form.get("DistanceFromHome", 5)),
            "Department":     form.get("Department","Sales"),
            "JobRole":        form.get("JobRole","Sales Executive"),
            "JobLevel":       int(form.get("JobLevel", 2)),
            "JobSatisfaction": int(form.get("JobSatisfaction", 3)),
            "OverTime":       form.get("OverTime","No"),
            "BusinessTravel": form.get("BusinessTravel","Non-Travel"),
            "MonthlyIncome":  int(form.get("MonthlyIncome", 5000)),
            "YearsAtCompany": int(form.get("YearsAtCompany", 3)),
            "TotalWorkingYears": int(form.get("TotalWorkingYears", 5)),
            "WorkLifeBalance": int(form.get("WorkLifeBalance", 3)),
            "EnvironmentSatisfaction": int(form.get("EnvironmentSatisfaction", 3)),
            "StockOptionLevel": int(form.get("StockOptionLevel", 0)),
            "EducationField": "Life Sciences",
            "Education": 3, "DailyRate": 800, "HourlyRate": 60,
            "MonthlyRate": 15000, "NumCompaniesWorked": 2,
            "PercentSalaryHike": 14, "PerformanceRating": 3,
            "RelationshipSatisfaction": 3, "TrainingTimesLastYear": 2,
            "YearsInCurrentRole": 2, "YearsSinceLastPromotion": 1,
            "YearsWithCurrManager": 2,
        }
        try:
            X = preprocess(employee)
            sk, tf_p, pt, avg, pred, risk = run_all_models(X)
            result = {
                "prediction": pred, "risk": risk,
                "avg": round(avg, 4),
                "sk":  round(sk, 4),
                "tf":  round(tf_p, 4),
                "pt":  round(pt, 4),
                "employee": employee,
            }
            save_prediction({
                "age": employee["Age"], "department": employee["Department"],
                "job_role": employee["JobRole"], "monthly_income": employee["MonthlyIncome"],
                "overtime": employee["OverTime"], "sk_prob": round(sk,4),
                "tf_prob": round(tf_p,4), "pt_prob": round(pt,4),
                "avg_prob": round(avg,4), "prediction": pred, "risk_level": risk,
            })
        except Exception as e:
            result = {"error": str(e)}
    return render_template("predict.html", result=result)

@app.route("/history")
def history():
    rows = get_history()
    total, leaving = get_stats()
    return render_template("history.html", rows=rows, total=total, leaving=leaving)

@app.route("/compare")
def compare():
    try:
        sk = _pkl("sklearn_metrics.pkl")
        tf = _pkl("tf_metrics.pkl")
        pt = _pkl("pytorch_metrics.pkl")
        fi = sk.get("feature_importance", {})
        top_fi = dict(sorted(fi.items(), key=lambda x:x[1], reverse=True)[:10])
    except:
        sk=tf=pt={}
        top_fi={}
    return render_template("compare.html", sk=sk, tf=tf, pt=pt, fi=top_fi)

# ── AI Insights API ───────────────────────────────────────────────────
@app.route("/api/ai", methods=["POST"])
def ai_insights():
    try:
        from groq import Groq

        body     = request.get_json()
        employee = body.get("employee", {})
        result   = body.get("result", {})

        api_key = os.environ.get("GROQ_API_KEY") or body.get("api_key", "")
        if not api_key:
            return jsonify({"error": "No API key found."}), 400

        client = Groq(api_key=api_key)

        prompt = f"""You are an expert HR analytics consultant.
ML models predict this employee has a {result.get('risk','?')} attrition risk
({result.get('avg','?')}% probability of leaving).
Employee profile: {json.dumps(employee, indent=2)}
Give exactly:
1. Top 3 risk factors (1 line each)
2. Top 3 retention actions (1 line each)
3. One priority recommendation (1 sentence)"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        return jsonify({"insights": response.choices[0].message.content})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    init_db()
    print("\n" + "="*45)
    print("           Attrition Predictor")
    print("           http://localhost:5000")
    print("="*45 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
