import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── 한글 폰트 CSS ──────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
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
    q25, q75 = s["생중량"].quantile(0.25), s["생중량"].quantile(0.75)
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


# ── 산점도 + 회귀 추세선 (반환값을 1개로 정리) ────────────────────────────
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
            marker=dict(color=color, size=7, opacity=0.7, line=dict(width=0.5, color="white")),
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
    labels = [l for l, _, _ in corr_list]
    vals = [v for _, v, _ in corr_list]
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

    # ── 사이드바: 파일 업로드 ──────────────────────────────────────────────
    st.sidebar.header("📁 데이터 업로드")
    env_file = st.sidebar.file_uploader("환경 CSV 업로드", type="csv")
    growth_file = st.sidebar.file_uploader("생육 결과 CSV 업로드", type="csv")

    st.sidebar.markdown("---")
    st.sidebar.header("🌿 생육 측정 항목 선택")
    all_metric_names = ["잎 수", "지상부 길이", "지하부 길이", "생중량",
                        "지상부 질량", "지하부 질량"]
    growth_metrics = []
    for m in all_metric_names:
        if st.sidebar.checkbox(m, value=True, key=f"chk_{m}"):
            growth_metrics.append(m)

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
        ["연구 개요 및 스마트팜 환경", "개체별 생육 차이 분석", "생육 지표와 생중량의 상관관계 및 예측"]
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
                            st.metric("평균 온도", f"{t.mean():.1f} °C")
                            st.metric("최고 온도", f"{t.max():.1f} °C")
                            st.metric("최저 온도", f"{t.min():.1f} °C")
                        else:
                            st.info("온도 값이 충분하지 않아 평균을 표시할 수 없어요.")
                    else:
                        st.info("⚠️ 온도(temperature) 열이 없어 온도 정보를 표시할 수 없어요.")

                    if "humidity" in env_df.columns:
                        h = env_df["humidity"].dropna()
                        if len(h) >= 2:
                            st.metric("평균 습도", f"{h.mean():.1f} %")
                            st.metric("최고 습도", f"{h.max():.1f} %")
                            st.metric("최저 습도", f"{h.min():.1f} %")
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
                            st.metric("평균 생중량", f"{w.mean():.2f} g")
                            st.metric("최대 생중량", f"{w.max():.2f} g")
                            st.metric("최소 생중량", f"{w.min():.2f} g")
                        else:
                            st.info("생중량 값이 충분하지 않아요.")
                    else:
                        st.info("⚠️ 생중량 열이 없어 생중량 요약을 표시할 수 없어요.")

                    if "잎 수" in growth_df.columns:
                        lf = growth_df["잎 수"].dropna()
                        if len(lf) >= 2:
                            st.metric("평균 잎 수", f"{lf.mean():.1f} 장")
                    if "지상부 길이" in growth_df.columns:
                        sh = growth_df["
