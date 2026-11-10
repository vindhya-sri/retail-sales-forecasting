from flask import Flask, request, jsonify, render_template
from google import genai
import pandas as pd
import json
import re
import xgboost as xgb
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import base64
from io import BytesIO
import requests
import os

app = Flask(__name__)

# ==========================================================
# MISTRAL API CONFIG
# ==========================================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-3.6-flash"


def ask_mistral(prompt: str, max_tokens: int = 1800) -> str:
    """Generate AI business insights using Google Gemini."""

    try:
        if not GEMINI_API_KEY:
            return "Gemini API error: API key is not configured."

        client = genai.Client(api_key=GEMINI_API_KEY)

        system_instruction = (
            "You are a senior retail business strategist and analytics consultant. "
            "Write in clear, professional business English. "
            "Use plain text ONLY. Do NOT use markdown symbols such as #, **, *, or backticks. "
            "Use CAPITAL LETTERS for all section headings. "
            "Use numbered lists like 1. 2. 3. "
            "Always write complete sentences. "
            "Never stop mid-sentence or mid-section. "
            "Never repeat raw numbers already shown on the dashboard."
        )

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=system_instruction + "\n\n" + prompt,
            config={
                "max_output_tokens": max_tokens,
                "temperature": 0.60,
            }
        )

        return response.text.strip() if response.text else "Gemini returned an empty response."

    except Exception as exc:
        print("GEMINI API ERROR:", str(exc))
        return f"Gemini API error: {str(exc)}"

# ==========================================================
# FORMAT RESPONSE
# All styles are INLINE so they work regardless of style.css
#
# FIX 1: Headings → bold thick near-black text + blue left bar
# FIX 2: Number badge and content have guaranteed 14px gap
# ==========================================================

# ── Inline style constants ──────────────────────────────
_HEADING_WRAP = (
    "display:flex; align-items:center; gap:12px; "
    "margin:28px 0 14px 0; padding:0; background:none;"
)
_HEADING_BAR = (
    "display:inline-block; width:5px; min-width:5px; height:22px; "
    "background:linear-gradient(180deg,#2563eb,#4f8ef7); "
    "border-radius:3px; flex-shrink:0;"
)
_HEADING_TEXT = (
    "font-size:15px; font-weight:800; letter-spacing:0.10em; "
    "text-transform:uppercase; color:#0f172a; line-height:1.2; "
    "font-family:'Plus Jakarta Sans',sans-serif;"
)
_LIST_ITEM_WRAP = (
    "display:flex; align-items:flex-start; gap:14px; "       # gap:14px = guaranteed space
    "margin-bottom:12px; padding:12px 16px; "
    "background:#ffffff; border:1px solid #e0e9ff; "
    "border-radius:10px; "
    "box-shadow:0 2px 6px rgba(79,142,247,0.06);"
)
_ITEM_NUM = (
    "display:inline-flex; align-items:center; justify-content:center; "
    "min-width:28px; width:28px; height:28px; flex-shrink:0; "  # flex-shrink:0 keeps circle shape
    "background:linear-gradient(135deg,#2563eb,#4f8ef7); "
    "color:#ffffff; font-size:12px; font-weight:800; "
    "border-radius:50%; margin-top:1px;"
)
_ITEM_CONTENT = (
    "font-size:14px; color:#4b5a80; line-height:1.75; flex:1;"
)
_ITEM_TITLE = (
    "font-weight:800; font-size:12.5px; letter-spacing:0.07em; "
    "text-transform:uppercase; color:#1e3a8a; "
    "background:#e0e9ff; padding:2px 8px; border-radius:4px; "
    "margin-right:6px; display:inline;"
)
_PARA = (
    "margin:0 0 11px 0; font-size:14.5px; "
    "color:#4b5a80; line-height:1.85;"
)
_SPACER = "height:6px;"
# ────────────────────────────────────────────────────────


def format_response(text: str) -> str:
    """
    Convert Mistral plain-text output into clean HTML with
    fully self-contained inline styles.

    Rules:
      - ALL-CAPS line            → bold dark heading with blue left bar
      - "1. CAPS TITLE: body"   → numbered card + highlighted title pill + body
      - "1. plain sentence"     → numbered card + plain body
      - Blank line              → spacer div
      - Anything else           → paragraph
    """
    if not text:
        return f'<p style="{_PARA}">No content returned.</p>'

    def esc(s: str) -> str:
        return (s
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;"))

    lines  = text.split("\n")
    output = []

    for line in lines:
        trimmed = line.strip()

        # ── Blank line → spacer
        if not trimmed:
            output.append(f'<div style="{_SPACER}"></div>')
            continue

        # ── Detect ALL-CAPS section heading
        upper = sum(1 for c in trimmed if c.isupper())
        lower = sum(1 for c in trimmed if c.islower())
        is_heading = (
            len(trimmed) >= 4
            and upper > lower * 2
            and len(trimmed) < 140
            and not re.match(r'^\d+\.', trimmed)   # exclude numbered items
        )

        if is_heading:
            output.append(
                f'<div style="{_HEADING_WRAP}">'
                f'<span style="{_HEADING_BAR}"></span>'           # coloured bar
                f'<span style="{_HEADING_TEXT}">{esc(trimmed)}</span>'  # bold dark text
                f'</div>'
            )
            continue

        # ── Detect numbered list item: "1. ..." or "1. CAPS TITLE: body"
        num_match = re.match(r'^(\d+)\.\s+(.*)', trimmed)
        if num_match:
            num  = num_match.group(1)
            rest = num_match.group(2).strip()

            # Sub-pattern: inline ALL-CAPS title followed by colon
            # e.g. "SUPPLY CHAIN DISRUPTIONS: Logistical bottlenecks..."
            title_match = re.match(r'^([A-Z][A-Z\s&\/\-]{2,60}):\s*(.*)', rest)

            if title_match:
                pill = esc(title_match.group(1).strip())
                body = esc(title_match.group(2).strip())
                output.append(
                    f'<div style="{_LIST_ITEM_WRAP}">'
                    f'<span style="{_ITEM_NUM}">{num}</span>'       # circle badge
                    f'<span style="{_ITEM_CONTENT}">'               # 14px gap from badge
                    f'<span style="{_ITEM_TITLE}">{pill}:</span> {body}'
                    f'</span>'
                    f'</div>'
                )
            else:
                output.append(
                    f'<div style="{_LIST_ITEM_WRAP}">'
                    f'<span style="{_ITEM_NUM}">{num}</span>'       # circle badge
                    f'<span style="{_ITEM_CONTENT}">{esc(rest)}</span>'  # 14px gap from badge
                    f'</div>'
                )
            continue

        # ── Normal paragraph
        output.append(f'<p style="{_PARA}">{esc(trimmed)}</p>')

    return "\n".join(output)


# ==========================================================
# 1  LOAD TRAINED DATASET (HuggingFace)
# ==========================================================
print("Loading retail dataset from HuggingFace...")
try:
    from datasets import load_dataset
    dataset = load_dataset(
        "electricsheepafrica/nigerian_retail_and_ecommerce_purchase_history_records",
        split="train",
    )
    trained_raw_df = dataset.to_pandas()
    trained_raw_df["purchase_date"] = pd.to_datetime(trained_raw_df["purchase_date"])
    print(f"Dataset ready: {trained_raw_df.shape}")
except Exception as e:
    print(f"Could not load HuggingFace dataset: {e}")
    trained_raw_df = pd.DataFrame(columns=[
        "customer_id", "purchase_date",
        "total_amount_ngn", "city",
        "product_category", "is_repeat_purchase"
    ])

uploaded_raw_df = None

# ==========================================================
# 2 LOAD XGBOOST FORECAST MODEL
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

model = xgb.XGBRegressor()

model.load_model(
    os.path.join(MODEL_DIR, "xgb_model.json")
)

with open(
    os.path.join(MODEL_DIR, "feature_columns.json"),
    "r"
) as f:
    feature_cols = json.load(f)

historical_df = pd.read_csv(
    os.path.join(DATA_DIR, "processed_company_daily.csv")
)

historical_df["date"] = pd.to_datetime(
    historical_df["date"]
)

print("XGBoost model + historical data loaded.")


# ==========================================================
# HELPERS
# ==========================================================
def get_active_dataset():
    return uploaded_raw_df.copy() if uploaded_raw_df is not None else trained_raw_df.copy()


def extract_horizon_days(query: str) -> int:
    q = query.lower()
    m = re.search(r'(\d+)\s*day', q)
    if m:
        return int(m.group(1))
    m = re.search(r'(\d+)\s*week', q)
    if m:
        return int(m.group(1)) * 7
    m = re.search(r'(\d+)\s*month', q)
    if m:
        return int(m.group(1)) * 30
    if "quarter" in q:
        return 90
    m = re.search(r'(\d+)\s*year', q)
    if m:
        return int(m.group(1)) * 365
    if "year" in q:
        return 365
    return 30


def extract_top_n(query: str, default: int = 10) -> int:
    m = re.search(r'top\s+(\d+)', query.lower())
    return int(m.group(1)) if m else default


# ==========================================================
# FORECAST ENGINE
# ==========================================================
def forecast_future(days_ahead: int):
    base_df = historical_df.copy().reset_index(drop=True)
    future_preds = []
    for _ in range(days_ahead):
        latest_features = base_df[feature_cols].iloc[-1:].copy()
        pred = float(model.predict(latest_features)[0])
        pred = max(pred, 0)
        noise = np.random.normal(0, pred * 0.04)
        pred = max(pred + noise, 0)
        future_preds.append(pred)
        next_date = base_df["date"].iloc[-1] + pd.Timedelta(days=1)
        new_row = base_df.iloc[-1:].copy()
        new_row["date"] = next_date
        new_row["sales"] = pred
        base_df = pd.concat([base_df, new_row], ignore_index=True)
    return future_preds


# ==========================================================
# CHART GENERATION (unchanged)
# ==========================================================

def generate_forecast_chart(predictions):
    dates = pd.date_range(
        start=historical_df["date"].max() + pd.Timedelta(days=1),
        periods=len(predictions),
    )
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#f0f4ff")
    ax.set_facecolor("#f8faff")

    ax.fill_between(dates, predictions, alpha=0.18, color="#4f8ef7")
    ax.plot(dates, predictions, color="#2563eb", linewidth=2.2, marker="o",
            markersize=3, markevery=max(1, len(predictions) // 20))

    ax.set_title("AI Revenue Forecast", fontsize=14, fontweight="bold",
                 color="#1e3a6e", pad=12)
    ax.set_xlabel("Month & Year", color="#4b5563")
    ax.set_ylabel("Revenue (NGN)", color="#4b5563")

    if len(predictions) > 730:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # every 3 months for 3 years
    elif len(predictions) > 365:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))  # every 2 months for 2 years
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # every month for 1 year

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate(rotation=45)
    ax.grid(True, linestyle="--", alpha=0.4, color="#c7d2fe")
    ax.spines[["top", "right"]].set_visible(False)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode()
    plt.close()
    return f"data:image/png;base64,{img}"


# ==========================================================
# INTELLIGENCE ENGINE
# ==========================================================
def extract_n_and_order(query: str):
    q = query.lower()
    ascending = False
    if any(w in q for w in ["least", "bottom", "worst", "lowest", "last", "poor", "weak"]):
        ascending = True
    m = re.search(r'(?:top|bottom|least|worst|lowest|best|highest|last|show|display)\s+(\d+)', q)
    if not m:
        m = re.search(r'(\d+)\s*(?:product|categor|city|cities|customer|item)', q)
    if m:
        n = int(m.group(1))
    else:
        n = 3 if ascending else 10
    return n, ascending


def smart_group_column(df: pd.DataFrame, q: str) -> str:
    col_map = {
        "customer": ["customer_id", "customer", "client_id", "client"],
        "city":     ["city", "cities", "location", "region", "state", "town"],
        "subcategory": ["product_subcategory", "subcategory", "sub_category",
                        "sub-category", "subcat"],
        "product":  ["product_category", "product", "category", "item", "sku",
                     "product_name", "item_name"],
        "gender":   ["gender", "sex"],
        "payment":  ["payment_method", "payment", "pay_method"],
        "channel":  ["channel", "platform", "source"],
    }
    for key, aliases in col_map.items():
        if any(a in q for a in aliases):
            for alias in aliases:
                for col in df.columns:
                    if alias in col.lower():
                        return col
            break
    for col in df.columns:
        if df[col].dtype == object and col not in ["purchase_date", "date"]:
            return col
    return None


def smart_value_column(df: pd.DataFrame) -> str:
    preferred = [
        "total_amount_ngn", "total_amount", "revenue", "sales",
        "amount", "price", "value", "spend", "purchase_amount"
    ]
    for p in preferred:
        for col in df.columns:
            if p in col.lower():
                return col
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            return col
    return None


def generate_intelligence_from_query(query: str) -> dict:
    df = get_active_dataset()
    q  = query.lower()
    result = {}

    n, ascending = extract_n_and_order(query)
    order_label  = "bottom" if ascending else "top"

    value_col = smart_value_column(df)

    # ════════════════════════════════════════════════════════
    # SUBCATEGORY / DRILL-DOWN QUERIES
    # Triggered by: "subcategory", "sub-category", "items in", "within", "inside"
    # e.g. "in health category which subcategory items are best selling"
    # ════════════════════════════════════════════════════════
    if any(w in q for w in ["subcategory", "sub-category", "items in", "within", "inside", "which item", "what item"]):
        # Detect which top-level category was mentioned
        cat_col = smart_group_column(df, "product category")
        sub_col = None
        # Look for a subcategory / item name column
        for col in df.columns:
            if any(k in col.lower() for k in ["sub", "item_name", "product_name", "sku", "item"]):
                sub_col = col
                break

        if cat_col and sub_col and value_col:
            # Find which category was mentioned in the query
            categories = df[cat_col].dropna().unique()
            matched_cat = None
            for cat in categories:
                if str(cat).lower() in q:
                    matched_cat = cat
                    break

            if matched_cat:
                filtered = df[df[cat_col].str.lower() == matched_cat.lower()]
                grouped = (
                    filtered.groupby(sub_col)[value_col]
                    .sum()
                    .sort_values(ascending=ascending)
                    .head(n)
                )
                result["filtered_category"] = str(matched_cat)
                result[f"{order_label}_{n}_items_in_{matched_cat.replace(' ', '_').lower()}"] = {
                    str(k): round(float(v), 2) for k, v in grouped.items()
                }
                result["total_items_in_category"] = int(filtered[sub_col].nunique())
            else:
                # No specific category matched — return top items across all
                group_col = sub_col
                grouped = (
                    df.groupby(group_col)[value_col]
                    .sum()
                    .sort_values(ascending=ascending)
                    .head(n)
                )
                result[f"{order_label}_{n}_items_overall"] = {
                    str(k): round(float(v), 2) for k, v in grouped.items()
                }
        elif cat_col and value_col:
            # No sub_col found — fallback to category level
            grouped = (
                df.groupby(cat_col)[value_col]
                .sum()
                .sort_values(ascending=ascending)
                .head(n)
            )
            result[f"{order_label}_{n}_categories"] = {
                str(k): round(float(v), 2) for k, v in grouped.items()
            }
            result["note"] = "No subcategory column found. Showing category-level results."
        return result

    # ════════════════════════════════════════════════════════
    # FILTER BY SPECIFIC CATEGORY VALUE
    # e.g. "best customers in electronics", "top cities for health"
    # ════════════════════════════════════════════════════════
    cat_col = smart_group_column(df, "product category")
    if cat_col:
        categories = df[cat_col].dropna().unique()
        matched_cat = None
        for cat in categories:
            if str(cat).lower() in q:
                matched_cat = cat
                break

        if matched_cat:
            df = df[df[cat_col].str.lower() == matched_cat.lower()]
            result["filtered_by_category"] = str(matched_cat)
            result["records_in_category"]  = int(len(df))

    # ════════════════════════════════════════════════════════
    # OVERVIEW / SUMMARY
    # ════════════════════════════════════════════════════════
    if any(w in q for w in ["overview", "summary", "all", "total", "overall"]):
        result["total_records"] = int(len(df))
        if value_col:
            result["total_revenue"]   = round(float(df[value_col].sum()), 2)
            result["average_revenue"] = round(float(df[value_col].mean()), 2)
            result["max_revenue"]     = round(float(df[value_col].max()), 2)
            result["min_revenue"]     = round(float(df[value_col].min()), 2)
        for col in df.columns:
            if df[col].dtype == object and col not in ["purchase_date", "date"]:
                result[f"unique_{col}"] = int(df[col].nunique())
        return result

    # ════════════════════════════════════════════════════════
    # CUSTOMER QUERIES
    # ════════════════════════════════════════════════════════
    if "customer" in q or "client" in q:
        cust_col = smart_group_column(df, "customer")
        if cust_col and value_col:
            total_unique_cust = int(df[cust_col].nunique())
            effective_n_cust = min(n, total_unique_cust)
            if ascending and n == 3:
                effective_n_cust = min(3, total_unique_cust)
            grouped = (
                df.groupby(cust_col)[value_col]
                .sum()
                .sort_values(ascending=ascending)
                .head(effective_n_cust)
            )
            if not ascending:
                result["total_customers"]        = total_unique_cust
                result["avg_spend_per_customer"] = round(
                    float(df.groupby(cust_col)[value_col].sum().mean()), 2
                )
                if "is_repeat_purchase" in df.columns:
                    result["repeat_rate_percent"] = round(
                        float(df["is_repeat_purchase"].mean()) * 100, 2
                    )
            result[f"{order_label}_{effective_n_cust}_customers"] = {
                str(k): round(float(v), 2) for k, v in grouped.items()
            }
        return result

    # ════════════════════════════════════════════════════════
    # CITY / REGION QUERIES
    # ════════════════════════════════════════════════════════
    if any(w in q for w in ["city", "cities", "region", "location", "state", "town"]):
        city_col = smart_group_column(df, "city")
        if city_col and value_col:
            city_rev = (
                df.groupby(city_col)[value_col]
                .sum()
                .sort_values(ascending=ascending)
            )
            total_unique_cities = int(df[city_col].nunique())
            effective_n_cities  = min(n, total_unique_cities)
            if ascending and n == 3:
                effective_n_cities = min(3, total_unique_cities)
            total_rev = city_rev.sum()
            if not ascending:
                result["total_cities"] = total_unique_cities
                result["best_city"]    = str(city_rev.idxmax())
                result["weakest_city"] = str(city_rev.idxmin())
            result[f"{order_label}_{effective_n_cities}_cities"] = {
                str(k): round(float(v), 2)
                for k, v in city_rev.head(effective_n_cities).items()
            }
            result[f"{order_label}_{effective_n_cities}_cities_share_percent"] = {
                str(k): round(float(v / total_rev * 100), 2)
                for k, v in city_rev.head(effective_n_cities).items()
            }
        return result

    # ════════════════════════════════════════════════════════
    # PRODUCT / CATEGORY QUERIES
    # ════════════════════════════════════════════════════════
    if any(w in q for w in ["product", "category", "item", "sku", "brand"]):
        prod_col = smart_group_column(df, "product category")
        if prod_col and value_col:
            cat_rev = (
                df.groupby(prod_col)[value_col]
                .sum()
                .sort_values(ascending=ascending)
            )
            total_unique = int(df[prod_col].nunique())
            effective_n  = min(n, total_unique)
            if ascending and n == 3:
                effective_n = min(3, total_unique)
            total_rev = cat_rev.sum()
            if not ascending:
                result["total_categories"] = total_unique
                result["best_category"]    = str(cat_rev.idxmax())
                result["weakest_category"] = str(cat_rev.idxmin())
            result[f"{order_label}_{effective_n}_categories"] = {
                str(k): round(float(v), 2)
                for k, v in cat_rev.head(effective_n).items()
            }
            result[f"{order_label}_{effective_n}_categories_share_percent"] = {
                str(k): round(float(v / total_rev * 100), 2)
                for k, v in cat_rev.head(effective_n).items()
            }
        return result

    # ════════════════════════════════════════════════════════
    # COMPARISON QUERIES
    # ════════════════════════════════════════════════════════
    if "compare" in q or "vs" in q or "versus" in q:
        group_col = smart_group_column(df, q)
        if group_col and value_col:
            all_groups = (
                df.groupby(group_col)[value_col]
                .agg(["sum", "mean", "count"])
                .round(2)
                .sort_values("sum", ascending=False)
            )
            result["full_comparison"] = all_groups.head(20).to_dict()
        return result

    # ════════════════════════════════════════════════════════
    # TREND / MONTHLY QUERIES
    # ════════════════════════════════════════════════════════
    if any(w in q for w in ["trend", "monthly", "yearly", "over time", "growth"]):
        date_col = None
        for col in df.columns:
            if "date" in col.lower() or "time" in col.lower():
                date_col = col
                break
        if date_col and value_col:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            monthly = (
                df.groupby(df[date_col].dt.to_period("M"))[value_col]
                .sum()
                .sort_index()
            )
            result["monthly_trend"] = {
                str(k): round(float(v), 2) for k, v in monthly.items()
            }
        return result

   # ════════════════════════════════════════════════════════
    # SUBCATEGORY QUERIES
    # Triggered by: "sub category", "subcategory", "sub-category"
    # ════════════════════════════════════════════════════════
    if any(w in q for w in ["sub category", "subcategory", "sub-category", "subcat",
                             "sub cat", "which sub", "best sub", "top sub"]):
        sub_col = None
        for col in df.columns:
            if "subcategory" in col.lower() or "sub_category" in col.lower():
                sub_col = col
                break

        # Check if a specific parent category is mentioned in the query
        cat_col = None
        for col in df.columns:
            if col.lower() == "product_category" or col.lower() == "category":
                cat_col = col
                break

        if sub_col and value_col:
            working_df = df.copy()

            # Filter by parent category if mentioned
            if cat_col:
                categories = df[cat_col].dropna().unique()
                matched_cat = None
                for cat in categories:
                    if str(cat).lower() in q:
                        matched_cat = cat
                        break
                if matched_cat:
                    working_df = df[df[cat_col].str.lower() == matched_cat.lower()]
                    result["filtered_by_category"] = str(matched_cat)

            sub_rev = (
                working_df.groupby(sub_col)[value_col]
                .sum()
                .sort_values(ascending=ascending)
            )
            total_unique_sub = int(working_df[sub_col].nunique())
            effective_n = min(n, total_unique_sub)
            total_rev = sub_rev.sum()

            result[f"{order_label}_{effective_n}_subcategories"] = {
                str(k): round(float(v), 2)
                for k, v in sub_rev.head(effective_n).items()
            }
            result[f"{order_label}_{effective_n}_subcategories_share_percent"] = {
                str(k): round(float(v / total_rev * 100), 2)
                for k, v in sub_rev.head(effective_n).items()
            }
            result["total_subcategories"] = total_unique_sub
            if not ascending:
                result["best_subcategory"]    = str(sub_rev.idxmax())
                result["weakest_subcategory"] = str(sub_rev.idxmin())
        return result

    # ════════════════════════════════════════════════════════
    # GENERIC DIMENSION FALLBACK
    # Exclude id columns like purchase_id from grouping
    # ════════════════════════════════════════════════════════
    group_col = smart_group_column(df, q)
    # Never group by purchase_id or any raw ID column
    if group_col and any(x in group_col.lower() for x in ["purchase_id", "_id"]):
        group_col = None
    if group_col and value_col:
        grouped = (
            df.groupby(group_col)[value_col]
            .sum()
            .sort_values(ascending=ascending)
            .head(n)
        )
        result[f"{order_label}_{n}_{group_col}"] = {
            str(k): round(float(v), 2) for k, v in grouped.items()
        }
        if not ascending:
            result["total_unique"] = int(df[group_col].nunique())
        return result

    # ════════════════════════════════════════════════════════
    # FINAL FALLBACK — pass all column summaries to Mistral
    # so it can still answer intelligently
    # ════════════════════════════════════════════════════════
    result["available_columns"] = list(df.columns)
    result["sample_data"]       = df.head(5).to_dict(orient="records")
    if value_col:
        result["total_revenue"] = round(float(df[value_col].sum()), 2)
    result["note"] = (
        f"Query '{query}' could not be matched to a specific pattern. "
        "Providing column info and sample data for AI analysis."
    )
    return result


# ==========================================================
# ROUTES
# ==========================================================
@app.route("/")
def home():
    return render_template("dashboard.html")


@app.route("/upload", methods=["POST"])
def upload_data():
    global uploaded_raw_df
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file provided"}), 400
    df = pd.read_csv(file)
    if "purchase_date" in df.columns:
        df["purchase_date"] = pd.to_datetime(df["purchase_date"])
    uploaded_raw_df = df
    last_date = (
        str(df["purchase_date"].max()) if "purchase_date" in df.columns else "N/A"
    )
    return jsonify({"message": "Dataset uploaded successfully", "last_date": last_date})


@app.route("/forecast", methods=["POST"])
def forecast():
    user_query  = request.json.get("query", "forecast next 30 days")
    horizon     = min(extract_horizon_days(user_query), 1095)
    predictions = forecast_future(horizon)

    total     = float(sum(predictions))
    avg_daily = float(total / len(predictions))
    chart     = generate_forecast_chart(predictions)

    prompt = f"""
You are a senior retail business strategist analysing an AI-generated revenue forecast for a Nigerian retail company.

Forecast period   : {horizon} days
Total revenue     : {total:,.2f} NGN
Average daily rev : {avg_daily:,.2f} NGN

Write a complete professional executive business report. Use CAPITAL LETTERS for each section heading. Do not use any markdown symbols. Structure it with these exact sections and write every section fully without stopping:

EXECUTIVE SUMMARY
Write 3 to 4 complete sentences summarising the revenue forecast outlook, consumer demand trends, and what the business must prioritize during this period.

GROWTH OUTLOOK FOR THE NEXT PERIOD
Write 4 to 5 complete sentences covering the main demand drivers such as seasonal peaks, digital sales growth, regional expansion opportunities, product mix improvements, and pricing strategies.

POSSIBLE RISKS AFFECTING THE FORECAST
List 5 risks using numbered points (1. 2. 3. 4. 5.). Each point must have a SHORT TITLE IN CAPITALS followed by a colon and then a full explanation sentence. Example format: 1. SUPPLY CHAIN DISRUPTIONS: explanation here.

OPERATIONAL RECOMMENDATIONS
List 5 numbered recommendations (1. 2. 3. 4. 5.). Each must be a complete sentence explaining what action to take and why it will improve performance.

STRATEGIC ACTIONS FOR MANAGEMENT
List 5 numbered actions (1. 2. 3. 4. 5.). Each must be a complete actionable sentence directing management on what to implement immediately.

CONCLUSION
Write 2 to 3 complete sentences summarising the overall business outlook and the most critical next step management should take.

Important: Complete every single section fully. Do not stop mid-sentence or mid-section. Do not use #, *, -, or --- symbols anywhere.
"""
    narrative = ask_mistral(prompt, max_tokens=1800)
    formatted = format_response(narrative)

    return jsonify({
        "horizon_days"         : horizon,
        "total_revenue"        : round(total, 2),
        "average_daily_revenue": round(avg_daily, 2),
        "forecast_chart"       : chart,
        "business_insight"     : formatted,
    })


@app.route("/intelligence", methods=["POST"])
def intelligence():
    query      = request.json.get("query", "")
    structured = generate_intelligence_from_query(query)


    prompt = f"""
You are a senior retail analytics consultant for a Nigerian retail company with deep expertise in product performance, customer behaviour, and market segmentation.

A business user asked: "{query}"

The analytics engine returned this structured data:
{json.dumps(structured, indent=2)}

Your job is to answer the user's question directly and completely using the data above. If the data contains item names, subcategory names, category names, or customer IDs, mention them specifically by name.

Write a complete structured consulting report. Use CAPITAL LETTERS for all section headings. Do not use any markdown symbols. Include these exact sections and write every section fully without stopping:

DIRECT ANSWER TO YOUR QUESTION
Write 2 to 3 complete sentences directly answering exactly what the user asked. Name the specific top items, categories, or segments from the data above.

KEY INSIGHTS
Write 3 to 4 complete sentences identifying the most important findings. Mention specific city names, customer IDs, or category names from the data above.

BUSINESS IMPLICATIONS
Write 3 to 4 complete sentences explaining what these insights mean for operations, supply chain management, inventory planning, and customer strategy.

REGIONAL OR SEGMENT DEEP DIVE
Write 3 to 4 complete sentences doing a deeper analysis of the top performers, explaining why they lead and what differentiates them from lower-performing segments.

STRATEGIC RECOMMENDATIONS
List 5 numbered recommendations (1. 2. 3. 4. 5.). Each must be a complete sentence with a clear action and the business reason behind it.

SUGGESTED MANAGEMENT ACTIONS
List 4 numbered actions (1. 2. 3. 4.). Each must be a complete, immediately actionable sentence that management can begin executing this quarter.

CONCLUSION
Write 2 complete sentences summarising the strategic direction and the single most important priority for the business right now.

Important: Complete every single section fully. Do not stop mid-sentence or mid-section. Do not use #, *, -, or --- symbols. Do not paste the raw JSON data.
"""
    explanation = ask_mistral(prompt, max_tokens=1600)
    formatted   = format_response(explanation)

    return jsonify({
        "structured_data": structured,
        "llm_explanation" : formatted,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), threaded=True)