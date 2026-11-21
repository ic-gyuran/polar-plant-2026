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
   max-height: 120px;
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

    <!-- 하늘 배경 -->
    <rect x="0" y="0" width="1200" height="260" fill="url(#sky)"/>

    <!-- 우측: 덩굴 -->
    <g stroke-linecap="round" fill="none">
      <path d="M1100,20 C1095,60 1105,90 1098,130 C1108,165 1093,195 1103,230"
            stroke="#2e7d32" stroke-width="2.5"/>
      <path d="M1140,15 C1150,55 1130,85 1148,125 C1135,160 1152,190 1140,225"
            stroke="#388e3c" stroke-width="2"/>
      <path d="M1065,30 C1055,65 1070,100 1060,140 C1072,170 1058,200 1070,235"
            stroke="#1b5e20" stroke-width="2"/>
      <path d="M1175,40 C1182,75 1168,110 1180,145 C1170,175 1185,205 1178,235"
            stroke="#2e7d32" stroke-width="1.8"/>
      <ellipse cx="1098" cy="128" rx="7" ry="4" fill="#4caf50" transform="rotate(-30 1098 128)"/>
      <ellipse cx="1148" cy="122" rx="7" ry="4" fill="#66bb6a" transform="rotate(25 1148 122)"/>
      <ellipse cx="1060" cy="138" rx="7" ry="4" fill="#43a047" transform="rotate(-20 1060 138)"/>
      <ellipse cx="1180" cy="142" rx="6" ry="3.5" fill="#81c784" transform="rotate(15 1180 142)"/>
      <ellipse cx="1098" cy="168" rx="7" ry="4" fill="#4caf50" transform="rotate(35 1098 168)"/>
      <ellipse cx="1148" cy="162" rx="6" ry="3.5" fill="#66bb6a" transform="rotate(-25 1148 162)"/>
      <ellipse cx="1060" cy="178" rx="6.5" ry="3.5" fill="#43a047" transform="rotate(30 1060 178)"/>
      <ellipse cx="1180" cy="182" rx="6" ry="3" fill="#81c784" transform="rotate(-15 1180 182)"/>
      <ellipse cx="1103" cy="228" rx="6" ry="3" fill="#4caf50" transform="rotate(20 1103 228)"/>
      <ellipse cx="1140" cy="222" rx="6" ry="3" fill="#66bb6a" transform="rotate(-30 1140 222)"/>
      <ellipse cx="1070" cy="232" rx="6" ry="3" fill="#43a047" transform="rotate(25 1070 232)"/>
      <ellipse cx="1178" cy="232" rx="5.5" ry="3" fill="#81c784" transform="rotate(-20 1178 232)"/>
    </g>

    <!-- 중앙: 무지개 -->
    <g>
      <ellipse cx="600" cy="125" rx="380" ry="100" fill="url(#rainbow_glow)"/>
      <path d="M225,125 A375,100 0 0,1 975,125" fill="none" stroke="#ef5350" stroke-width="12" stroke-linecap="round"/>
      <path d="M256,125 A344,92 0 0,1 944,125" fill="none" stroke="#ff8a65" stroke-width="12" stroke-linecap="round"/>
      <path d="M288,125 A314,84 0 0,1 912,125" fill="none" stroke="#ffc107" stroke-width="12" stroke-linecap="round"/>
      <path d="M319,125 A284,76 0 0,1 881,125" fill="none" stroke="#cddc39" stroke-width="12" stroke-linecap="round"/>
      <path d="M350,125 A254,68 0 0,1 850,125" fill="none" stroke="#8bc34a" stroke-width="12" stroke-linecap="round"/>
      <path d="M381,125 A224,60 0 0,1 819,125" fill="none" stroke="#4caf50" stroke-width="12" stroke-linecap="round"/>
      <path d="M412,125 A194,52 0 0,1 788,125" fill="none" stroke="#26c6da" stroke-width="12" stroke-linecap="round"/>
      <path d="M443,125 A164,44 0 0,1 757,125" fill="none" stroke="#42a5f5" stroke-width="12" stroke-linecap="round"/>
      <path d="M474,125 A134,36 0 0,1 726,125" fill="none" stroke="#7e57c2" stroke-width="12" stroke-linecap="round"/>
      <path d="M505,125 A104,28 0 0,1 695,125" fill="none" stroke="#e040fb" stroke-width="12" stroke-linecap="round"/>
      <ellipse cx="470" cy="95" rx="40" ry="14" fill="#ffffff" opacity="0.85"/>
      <ellipse cx="495" cy="92" rx="28" ry="11" fill="#ffffff" opacity="0.85"/>
      <ellipse cx="735" cy="95" rx="40" ry="14" fill="#ffffff" opacity="0.85"/>
      <ellipse cx="710" cy="92" rx="28" ry="11" fill="#ffffff" opacity="0.85"/>
    </g>

    <!-- 하단 얼음 -->
    <path d="M0,210 Q100,205 200,212 T420,210 T560,214 T720,210 T880,213 T1050,210 L1200,212 L1200,260 L0,260 Z"
          fill="url(#ice)" opacity="0.85"/>
    <path d="M0,225 Q150,222 300,228 T600,225 T900,228 T1200,225 L1200,260 L0,260 Z"
          fill="#b3dfe5" opacity="0.5"/>

    <!-- 좌측 하단: 펭귄 + 극지 -->
    <g transform="translate(40, 185)">
      <path d="M0,25 L40,5 L75,25 Z" fill="#e3f2fd" stroke="#90caf9" stroke-width="1.5"/>
      <path d="M70,25 L110,8 L150,25 Z" fill="#bbdefb" stroke="#90caf9" stroke-width="1.5"/>
      <path d="M140,25 L180,12 L220,25 Z" fill="#e3f2fd" stroke="#90caf9" stroke-width="1.5"/>
      <ellipse cx="100" cy="16" rx="15" ry="12" fill="#212121"/>
      <ellipse cx="100" cy="16" rx="9" ry="8" fill="#f5f5f5"/>
      <circle cx="100" cy="4" r="6" fill="#212121"/>
      <circle cx="97" cy="3" r="1.2" fill="#ffffff"/>
      <circle cx="103" cy="3" r="1.2" fill="#ffffff"/>
      <path d="M98,6 L100,8 L102,6 Z" fill="#ff8f00"/>
      <path d="M86,14 Q80,18 84,20" fill="none" stroke="#212121" stroke-width="2" stroke-linecap="round"/>
      <path d="M114,14 Q120,18 116,20" fill="none" stroke="#212121" stroke-width="2" stroke-linecap="round"/>
      <ellipse cx="93" cy="27" rx="4" ry="2" fill="#ff8f00"/>
      <ellipse cx="107" cy="27" rx="4" ry="2" fill="#ff8f00"/>
      <circle cx="20" cy="6" r="1" fill="#b3e5fc" opacity="0.7"/>
      <circle cx="50" cy="10" r="1" fill="#b3e5fc" opacity="0.7"/>
      <circle cx="130" cy="4" r="1" fill="#b3e5fc" opacity="0.7"/>
      <circle cx="165" cy="8" r="1" fill="#b3e5fc" opacity="0.7"/>
      <circle cx="185" cy="14" r="0.8" fill="#b3e5fc" opacity="0.7"/>
      <circle cx="25" cy="14" r="0.8" fill="#b3e5fc" opacity="0.7"/>
    </g>
  </svg>
</div>
""",
        unsafe_allow_html=True,
    )

    # ── 사이드바 ────────────────────────────────────────────────────────────
    st.sidebar.header("📁 데이터 업로드")
    env_file = st.sidebar.file_uploader("환경 CSV 업로드", type="csv")
    growth_file = st.sidebar.file_uploader("생육 결과 CSV 업로드", type="csv")

    st.sidebar.markdown("---")
    st.sidebar.header("🌿 생육 측정 항목 선택")
    all_metric_names = [
        "잎 수", "지상부 길이", "지하부 길이", "생중량",
        "지상부 질량", "지하부 질량",
    ]
    for m in all_metric_names:
        st.sidebar.checkbox(m, value=True, key=f"chk_{m}")

    # ── 데이터 로딩 (한 번만) ──────────────────────────────────────────────
    if env_file is not None:
        with st.spinner("환경 데이터를 읽는 중..."):
            env_df = std_cols(read_csv(env_file))
        env_school = extract_school_name(env_file.name)
    else:
        env_df = pd.DataFrame()
        env_school = None

    if growth_file is not None:
        with st.spinner("생육 결과 데이터를 읽는 중..."):
            growth_df = std_cols(read_csv(growth_file))
        growth_school = extract_school_name(growth_file.name)
    else:
        growth_df = pd.DataFrame()
        growth_school = None

    # ── 탭 구성 ────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(
        [
            "연구 개요 및 스마트팜 환경",
            "개체별 생육 차이 분석",
            "생육 지표와 생중량의 상관관계 및 예측",
        ]
    )

    # ─────────────────────────────────────────────────────────────────────────
    #  탭 1
    # ─────────────────────────────────────────────────────────────────────────
    with tab1:
        try:
            st.markdown("### 📌 연구 목적")
            st.info(
                "극지 환경에서 자라는 식물이 같은 스마트팜 안에서도 개체마다 "
                "어떻게 다르게 자라는지 살펴보고, 생육 지표 사이의 관계를 알아보기 위한 연구예요."
            )

            st.markdown("### 📌 가설")
            st.info(
                "• 잎 수가 많을수록 생중량(전체 무게)도 무거울 것이다.  "
                "• 지상부(위로 자란 부분) 길이와 생중량은 서로 관련이 있을 것이다.  "
                "• 지하부(뿌리 부분) 길이와 생중량은 서로 관련이 있을 것이다."
            )

            if env_df.empty and growth_df.empty:
                st.info("📎 사이드바에서 CSV 파일을 업로드하면 분석 결과가 화면에 나타나요.")
            else:
                school = env_school or growth_school or "우리 학교"
                st.markdown(f"### 🏫 분석 대상 학교: {school}")

                if not env_df.empty:
                    st.markdown("---")
                    st.markdown("### 🌡️ 스마트팜 환경 데이터 요약")

                    if "temperature" in env_df.columns:
                        t = env_df["temperature"].dropna()
                        if len(t) >= 2:
                            c1, c2, c3 = st.columns(3)
                            c1.metric("평균 온도", f"{t.mean():.1f} °C")
                            c2.metric("최고 온도", f"{t.max():.1f} °C")
                            c3.metric("최저 온도", f"{t.min():.1f} °C")
                        else:
                            st.info("온도 값이 충분하지 않아 평균을 표시할 수 없어요.")
                    else:
                        st.info("⚠️ 온도(temperature) 열이 없어 온도 정보를 표시할 수 없어요.")

                    if "humidity" in env_df.columns:
                        h = env_df["humidity"].dropna()
                        if len(h) >= 2:
                            c1, c2, c3 = st.columns(3)
                            c1.metric("평균 습도", f"{h.mean():.1f} %")
                            c2.metric("최고 습도", f"{h.max():.1f} %")
                            c3.metric("최저 습도", f"{h.min():.1f} %")
                        else:
                            st.info("습도 값이 충분하지 않아 평균을 표시할 수 없어요.")
                    else:
                        st.info("⚠️ 습도(humidity) 열이 없어 습도 정보를 표시할 수 없어요.")

                    st.markdown("### 📈 날짜별 평균 온도·습도 변화")
                    fig_line = make_env_line(env_df)
                    if fig_line is not None:
                        st.plotly_chart(fig_line, width="stretch")
                    else:
                        st.info("날짜별 변화를 그릴 수 있는 데이터가 충분하지 않아요.")

                if not growth_df.empty:
                    st.markdown("---")
                    st.markdown("### 🌿 생육 결과 데이터 요약")
                    n = len(growth_df)
                    st.metric("총 개체 수", f"{n} 개체")

                    if "생중량" in growth_df.columns:
                        w = growth_df["생중량"].dropna()
                        if len(w) >= 2:
                            c1, c2, c3 = st.columns(3)
                            c1.metric("평균 생중량", f"{w.mean():.2f} g")
                            c2.metric("최대 생중량", f"{w.max():.2f} g")
                            c3.metric("최소 생중량", f"{w.min():.2f} g")
                        else:
                            st.info("생중량 값이 충분하지 않아요.")
                    else:
                        st.info("⚠️ 생중량 열이 없어 생중량 요약을 표시할 수 없어요.")

                    if "잎 수" in growth_df.columns:
                        lf = growth_df["잎 수"].dropna()
                        if len(lf) >= 2:
                            st.metric("평균 잎 수", f"{lf.mean():.1f} 장")
                    if "지상부 길이" in growth_df.columns:
                        sh = growth_df["지상부 길이"].dropna()
                        if len(sh) >= 2:
                            st.metric("평균 지상부 길이", f"{sh.mean():.1f} mm")
                    if "지하부 길이" in growth_df.columns:
                        ub = growth_df["지하부 길이"].dropna()
                        if len(ub) >= 2:
                            st.metric("평균 지하부 길이", f"{ub.mean():.1f} mm")

        except Exception as e:
            st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    #  탭 2
    # ─────────────────────────────────────────────────────────────────────────
    with tab2:
        try:
            if growth_df.empty:
                st.info("📎 사이드바에서 생육 결과 CSV를 업로드하면 분석 결과가 나타나요.")
            else:
                st.markdown("### 📊 개체별 생중량 분포")
                fig_hist = make_weight_hist(growth_df)
                if fig_hist is not None:
                    st.plotly_chart(fig_hist, width="stretch")
                else:
                    st.info(
                        "⚠️ 생중량 열이 없거나 값이 충분하지 않아 히스토그램을 그릴 수 없어요."
                    )

                st.markdown("### 📏 생육 지표별 평균·표준편차·변동계수")
                st.info(
                    "• **평균**: 모든 개체 값을 더해 개수로 나눈 값  "
                    "• **표준편차**: 개체의 값이 평균에서 얼마나 흩어져 있는지 나타내는 값  "
                    "• **변동계수(CV)**: 표준편차를 평균으로 나눈 값으로, 서로 다른 단위끼리도 비교할 수 있어요."
                )

                metric_cols_avail = [
                    c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량"]
                    if c in growth_df.columns
                ]
                if metric_cols_avail:
                    for i in range(0, len(metric_cols_avail), 2):
                        num_cols = min(2, len(metric_cols_avail) - i)
                        cols = st.columns(num_cols)
                        for j, c in enumerate(metric_cols_avail[i : i + num_cols]):
                            s = growth_df[c].dropna()
                            if len(s) >= 2:
                                mean_v = s.mean()
                                std_v = s.std()
                                cv = (std_v / mean_v * 100) if mean_v != 0 else 0
                                with cols[j]:
                                    st.metric(
                                        label=c,
                                        value=f"{mean_v:.2f}",
                                        delta=f"표준편차 {std_v:.2f}  |  CV {cv:.1f}%",
                                    )
                            else:
                                with cols[j]:
                                    st.info(f"⚠️ {c} 값이 충분하지 않아요.")
                else:
                    st.info("⚠️ 표시할 생육 지표 열이 없어요.")

                st.markdown("### 🔬 생중량 상위 25% vs 하위 25% 비교")
                st.info(
                    "생중량이 무거운 상위 25% 개체와 가벼운 하위 25% 개체를 모아 "
                    "각 생육 지표의 평균을 비교한 그래프예요."
                )
                fig_bar = make_top_bottom_bar(growth_df)
                if fig_bar is not None:
                    st.plotly_chart(fig_bar, width="stretch")
                else:
                    st.info(
                        "⚠️ 비교할 데이터가 충분하지 않아요. "
                        "(개체 수가 너무 적거나 생중량 열이 없음)"
                    )

        except Exception as e:
            st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    #  탭 3
    # ─────────────────────────────────────────────────────────────────────────
    with tab3:
        try:
            if growth_df.empty:
                st.info("📎 사이드바에서 생육 결과 CSV를 업로드하면 분석 결과가 나타나요.")
            elif "생중량" not in growth_df.columns:
                st.info("⚠️ 생중량(g) 열이 없어 상관관계 분석을 할 수 없어요.")
            else:
                predictors = [
                    c for c in ["잎 수", "지상부 길이", "지하부 길이"]
                    if c in growth_df.columns
                ]
                if not predictors:
                    st.info(
                        "⚠️ 상관분석에 사용할 생육 지표(잎 수·지상부 길이·지하부 길이) 열이 없어요."
                    )
                else:
                    st.markdown("### 🔗 생육 지표와 생중량의 상관계수 비교")
                    st.info(
                        "**상관계수(r)**는 -1에서 1 사이의 값으로,  "
                        "1에 가까울수록 함께 증가하고, -1에 가까울수록 함께 감소하며,  "
                        "0에 가까우면 관계가 거의 없다는 뜻이에요."
                    )

                    corr_results = []
                    for p in predictors:
                        x_series = growth_df[p].dropna()
                        y_series = growth_df["생중량"].dropna()
                        common_idx = x_series.index.intersection(y_series.index)
                        if len(common_idx) >= 2:
                            xv = growth_df.loc[common_idx, p].values
                            yv = growth_df.loc[common_idx, "생중량"].values
                            r_val = float(np.corrcoef(xv, yv)[0, 1])
                            corr_results.append((p, r_val))

                    if corr_results:
                        fig_corr = make_corr_bar(
                            corr_results,
                            "생육 지표 vs 생중량 상관계수 (r)",
                        )
                        if fig_corr is not None:
                            st.plotly_chart(fig_corr, width="stretch")

                    st.markdown("---")
                    st.markdown("### 📈 산점도 및 회귀 추세선")
                    st.info(
                        "각 생육 지표와 생중량의 관계를 점과 추세선으로 나타낸 그래프예요.  "
                        "**추세선**은 np.polyfit으로 직접 구한 직선이에요 "
                        "(px.scatter의 trendline 인자는 statsmodels가 필요해 사용하지 않았어요)."
                    )

                    for p in predictors:
                        x_series = growth_df[p].dropna()
                        y_series = growth_df["생중량"].dropna()
                        common_idx = x_series.index.intersection(y_series.index)
                        if len(common_idx) < 2:
                            continue
                        x_s = growth_df.loc[common_idx, p]
                        y_s = growth_df.loc[common_idx, "생중량"]
                        figure = make_scatter_with_trend(
                            x_s, y_s, p, "생중량 (g)", color="#2c3e50"
                        )
                        if figure is not None:
                            info = getattr(figure, "_trend_info", None)
                            r_val = info["r"] if info else float("nan")
                            st.markdown(
                                f"#### {p} vs 생중량  (상관계수 r = {r_val:.3f})"
                            )
                            st.plotly_chart(figure, width="stretch")

                    st.markdown("---")
                    st.markdown("### 🔮 잎 수를 이용한 생중량 예측")
                    if "잎 수" in growth_df.columns:
                        x_series = growth_df["잎 수"].dropna()
                        y_series = growth_df["생중량"].dropna()
                        common_idx = x_series.index.intersection(y_series.index)
                        if len(common_idx) >= 2:
                            x_l = growth_df.loc[common_idx, "잎 수"].values
                            y_l = growth_df.loc[common_idx, "생중량"].values
                            slope_l, intercept_l = np.polyfit(x_l, y_l, 1)
                            y_pred = slope_l * x_l + intercept_l
                            ss_res = float(((y_l - y_pred) ** 2).sum())
                            ss_tot = float(((y_l - y_l.mean()) ** 2).sum())
                            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                            st.success(
                                "**예측 회귀식:**  생중량(g) = "
                                f"{slope_l:.3f} × 잎 수 + {intercept_l:.3f}\n\n"
                                f"**결정계수 (R²):** {r2:.3f}\n\n"
                                "R²는 0~1 사이의 값으로, 1에 가까울수록 "
                                "잎 수로 생중량을 잘 예측할 수 있다는 뜻이에요."
                            )
                        else:
                            st.info(
                                "⚠️ 잎 수와 생중량 값이 충분하지 않아 "
                                "회귀식을 구할 수 없어요."
                            )
                    else:
                        st.info(
                            "⚠️ 잎 수 열이 없어 회귀 예측을 할 수 없어요."
                        )

                    st.markdown("---")
                    st.markdown("### 📥 생육 데이터 다운로드 (한글 깨짐 방지)")
                    if not growth_df.empty:
                        st.download_button(
                            label="생육 결과 CSV 다운로드",
                            data=csv_for_download(growth_df),
                            file_name="생육결과_가공.csv",
                            mime="text/csv",
                        )

        except Exception as e:
            st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


if __name__ == "__main__":
    main()
