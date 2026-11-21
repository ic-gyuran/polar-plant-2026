import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── 한글 폰트 CSS + 배경 장식 스타일 ────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
}
/* ── 배경 장식 스타일 ─────────────────────────────────────────────────── */
polar-banner {
   display: flex;
   justify-content: center;
   margin-bottom: 2px;
}
polar-svg {
   width: 100%;
   max-width: 1100px;
   height: 100%;
   max-height: 130px;
}
.stApp {
   background-color: #f5fafe;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── CSV 읽기 (인코딩 처리, 한 번만 읽기) ────────────────────────────────────
@st.cache_data
def read_csv(file):
    if file is None:
        return pd.DataFrame()
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()


# ── 열 이름 표준화 ─────────────────────────────────────────────────────────
def std_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)
    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(
                df[c].astype(str).str.replace(",", "")
                .str.extract(r"(-?\d+\.?\d*)")[0],
                errors="coerce",
            )
            if num.notna().mean() >= 0.5:
                df[c] = num
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = pd.to_datetime(
                parts[0], format="%Y%m%d"
            ) + pd.to_timedelta(parts[1].astype(int), unit="h")
        df["time"] = parsed
    return df


# ── 학교명 추출 ────────────────────────────────────────────────────────────
def extract_school_name(filename):
    base = os.path.splitext(os.path.basename(filename))[0]
    name = base.split("_")[0].split("-")[0].strip()
    if name.isdigit():
        name = base
    return name or "우리 학교"


# ── 공통 스타일 ────────────────────────────────────────────────────────────
def style_figure(fig):
    fig.update_layout(
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        margin=dict(l=40, r=20, t=30, b=30),
    )
    return fig


# ── 날짜별 온도·습도 꺾은선 ─────────────────────────────────────────────────
def make_env_line(env_df):
    if "time" not in env_df.columns:
        return None
    df = env_df.dropna(subset=["time"]).copy()
    if df.empty:
        return None
    df["date"] = df["time"].dt.date
    agg = df.groupby("date").agg(
        temperature=("temperature", "mean"),
        humidity=("humidity", "mean"),
    ).reset_index()
    if agg.empty:
        return None
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("평균 온도 (°C)", "평균 습도 (%)"),
        vertical_spacing=0.12,
    )
    fig.add_trace(
        go.Scatter(
            x=agg["date"],
            y=agg["temperature"],
            mode="lines+markers",
            name="온도",
            line=dict(color="#e74c3c", width=1.5),
            marker=dict(size=3),
            hovertemplate="날짜: %{x}<br>평균 온도: %{y:.1f}°C<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=agg["date"],
            y=agg["humidity"],
            mode="lines+markers",
            name="습도",
            line=dict(color="#3498db", width=1.5),
            marker=dict(size=3),
            hovertemplate="날짜: %{x}<br>평균 습도: %{y:.1f}%<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.update_xaxes(title_text="날짜", row=2, col=1)
    fig.update_yaxes(title_text="°C", row=1, col=1)
    fig.update_yaxes(title_text="%", row=2, col=1)
    fig.update_layout(height=480, title_text="날짜별 평균 온도·습도 변화", showlegend=False)
    return style_figure(fig)


# ── 생중량 히스토그램 ──────────────────────────────────────────────────────
def make_weight_hist(growth_df):
    if "생중량" not in growth_df.columns:
        return None
    s = growth_df["생중량"].dropna()
    if len(s) < 2:
        return None
    fig = px.histogram(
        x=s,
        nbins=min(20, max(5, len(s) // 5)),
        title="개체별 생중량(g) 분포",
        labels={"x": "생중량 (g)", "count": "개체 수"},
        color_discrete_sequence=["#27ae60"],
    )
    fig.update_traces(hovertemplate="생중량: %{x:.1f}g<br>개체 수: %{y}<extra></extra>")
    fig.update_layout(
        xaxis_title="생중량 (g)",
        yaxis_title="개체 수",
        bargap=0.1,
    )
    return style_figure(fig)


# ── 상위·하위 25% 비교 막대그래프 ─────────────────────────────────────────
def make_top_bottom_bar(growth_df):
    if "생중량" not in growth_df.columns:
        return None
    cols = [
        c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량"]
        if c in growth_df.columns
    ]
    if not cols:
        return None
    s = growth_df.dropna(subset=cols).copy()
    if len(s) < 4:
        return None
    q25 = s["생중량"].quantile(0.25)
    q75 = s["생중량"].quantile(0.75)
    top = s[s["생중량"] >= q75]
    bottom = s[s["생중량"] <= q25]
    if top.empty or bottom.empty:
        return None

    def mean_row(sub, label):
        row = {"집단": label}
        for c in cols:
            v = sub[c].dropna()
            row[c] = v.mean() if len(v) >= 1 else 0
        return row

    rows = []
    rows.append(mean_row(top, "생중량 상위 25%"))
    rows.append(mean_row(bottom, "생중량 하위 25%"))
    df_plot = pd.DataFrame(rows)

    fig = make_subplots(
        rows=len(cols),
        cols=1,
        subplot_titles=[c for c in cols],
        vertical_spacing=0.10,
    )
    colors = ["#e74c3c", "#3498db"]
    for i, c in enumerate(cols):
        for j, (_, row) in enumerate(df_plot.iterrows()):
            fig.add_trace(
                go.Bar(
                    x=[row["집단"]],
                    y=[row[c]],
                    name=row["집단"] if j == 0 else "",
                    marker=dict(color=colors[j], opacity=0.85),
                    text=[f"{row[c]:.1f}"],
                    textposition="outside",
                    hovertemplate=f"{c}<br>{row['집단']}: %{y:.1f}<extra></extra>",
                ),
                row=i + 1,
                col=1,
            )
    for i, c in enumerate(cols):
        fig.update_yaxes(title_text=c, row=i + 1, col=1)
    fig.update_layout(
        height=280 + 100 * len(cols),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    return style_figure(fig)


# ── 산점도 + 회귀 추세선 ────────────────────────────────────────────────────
def make_scatter_with_trend(x_series, y_series, x_label, y_label, color="#2c3e50"):
    df = pd.DataFrame({"x": x_series, "y": y_series}).dropna()
    if len(df) < 2:
        return None
    x = df["x"].values
    y = df["y"].values
    r = float(np.corrcoef(x, y)[0, 1])
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ys = slope * xs + intercept

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name="데이터",
            marker=dict(
                color=color, size=7, opacity=0.7,
                line=dict(width=0.5, color="white"),
            ),
            hovertemplate=f"{x_label}: %{{x:.1f}}<br>{y_label}: %{{y:.1f}}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            name=f"추세선 (기울기 {slope:.3f})",
            line=dict(color="#e74c3c", width=2),
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
    )
    styled = style_figure(fig)
    styled._trend_info = {"r": r, "slope": slope, "intercept": intercept}
    return styled


# ── 상관계수 비교 막대 ─────────────────────────────────────────────────────
def make_corr_bar(corr_list, title):
    if not corr_list:
        return None
    labels = [item[0] for item in corr_list]
    vals = [item[1] for item in corr_list]
    colors = ["#27ae60" if v >= 0 else "#e74c3c" for v in vals]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=vals,
            marker=dict(color=colors, opacity=0.85),
            text=[f"{v:.3f}" for v in vals],
            textposition="outside",
            hovertemplate="%{x}<br>상관계수: %{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="생육 지표",
        yaxis_title="상관계수 (r)",
        yaxis=dict(range=[-1, 1]),
        showlegend=False,
    )
    return style_figure(fig)


# ── 다운로드용 CSV 인코딩 ───────────────────────────────────────────────────
def csv_for_download(df):
    return df.to_csv(index=False).encode("utf-8-sig")


# ═══════════════════════════════════════════════════════════════════════════
#  메인 앱
# ═══════════════════════════════════════════════════════════════════════════
def main():
    st.set_page_config(page_title="극지생육 대시보드", layout="wide")
    st.title("인천여고와 극지연구소의 콜라보레이션! 🌱")

    # ── 상단 배경 장식 (덩굴 + 무지개 + 펭귄·극지) ─────────────────────────
    st.markdown(
        """
<div class="polar-banner">
  <svg viewBox="0 0 1200 260" preserveAspectRatio="xMidYMid meet" class="polar-svg">
    <defs>
      <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#e8f4f8"/>
        <stop offset="100%" stop-color="#d4eef7"/>
      </linearGradient>
      <linearGradient id="ice" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#e8f4f8"/>
        <stop offset="100%" stop-color="#cfe6f0"/>
      </linearGradient>
      <radialGradient id="rainbow_glow" cx="0.5" cy="0.5" r="0.5">
        <stop offset="0%" stop-color="#ffffff" stop-opacity="0.5"/>
        <stop offset="100%" stop-color="#ffffff" stop-opacity="0"/>
      </radialGradient>
    </defs>

    <rect x="0" y="0" width="1200" height="260" fill="url(#sky)"/>

    <g stroke-linecap="round" fill="none">
      <path d="M1100,20 C1095,60 1105,90 1098,130 C1108,165 1093,195 1103,230"
            stroke="#2e7d32" stroke-width="2.5"/>
      <path d="M1140,15 C1150,55 1130,85 1148,125 C1135,1
