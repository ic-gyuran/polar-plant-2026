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

# ── CSV 읽기 (인코딩 처리) ─────────────────────────────────────────────────
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
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "지하부" in k and ("길이" in k or "길이" in c):
            rename[c] = "지하부 길이"
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
    # 밑줄 또는 공백 기준 첫 토큰
    name = base.split("_")[0].split("-")[0].strip()
    # 숫자만 있으면 제거
    if name.isdigit():
        name = base
    return name or "우리 학교"


# ── 공통 그래프 스타일 ─────────────────────────────────────────────────────
def style_figure(fig):
    fig.update_layout(
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        margin=dict(l=40, r=20, t=30, b=30),
    )
    return fig


# ── 날짜별 온도·습도 꺾은선 그래프 ─────────────────────────────────────────
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
            row[c] = v.mean() if len(v) > 0 else 0
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
        return None, None, None
    x = df["x"].values
    y = df["y"].values
    r = np.corrcoef(x, y)[0, 1]
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
    return style_figure(fig), r, (slope, intercept)


# ── 상관계수 비교 막대그래프 ────────────────────────────────────────────────
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


# ── download용 CSV 인코딩 ────────────────────────────────────────────────────
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
    growth_metrics = []
    all_metric_names = ["잎 수", "지상부 길이", "지하부 길이", "생중량",
                        "지상부 질량", "지하부 질량"]
    # 미리 읽기 전이므로 일단 전체 선택 가능하게 — 실제 사용 여부는 각 탭에서 확인
    for m in all_metric_names:
        if st.sidebar.checkbox(m, value=True, key=f"chk_{m}"):
            growth_metrics.append(m)

    # ── 데이터 로딩 ────────────────────────────────────────────────────────
    if env_file is not None:
        with st.spinner("환경 데이터를 읽는 중..."):
            env_raw = read_csv(env_file)
        env_df = std_cols(env_raw)
        env_school = extract_school_name(env_file.name)
    else:
        env_df = pd.DataFrame()
        env_school = None

    if growth_file is not None:
        with st.spinner("생육 결과 데이터를 읽는 중..."):
            growth_raw = read_csv(growth_file)
        growth_df = std_cols(growth_raw)
        growth_school = extract_school_name(growth_file.name)
    else:
        growth_df = pd.DataFrame()
        growth_school = None

    # ── 탭 구성 ────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(
        ["연구 개요 및 스마트팜 환경", "개체별 생육 차이 분석", "생육 지표와 생중량의 상관관계 및 예측"]
    )

    # ─────────────────────────────────────────────────────────────────────────
    #  탭 1: 연구 개요 및 스마트팜 환경
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
                # 학교명 표시
                school = env_school or growth_school or "우리 학교"
                st.markdown(f"### 🏫 분석 대상 학교: {school}")

                # 환경 데이터 요약
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

                # 생육 데이터 요약
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
    #  탭 2: 개체별 생육 차이 분석
    # ─────────────────────────────────────────────────────────────────────────
    with tab2:
        try:
            if growth_df.empty:
                st.info("📎 사이드바에서 생육 결과 CSV를 업로드하면 분석 결과가 나타나요.")
            else:
                # 생중량 히스토그램
                st.markdown("### 📊 개체별 생중량 분포")
                fig_hist = make_weight_hist(growth_df)
                if fig_hist is not None:
                    st.plotly_chart(fig_hist, width="stretch")
                else:
                    st.info("⚠️ 생중량 열이 없거나 값이 충분하지 않아 히스토그램을 그릴 수 없어요.")

                # 평균·표준편차·변동계수 카드
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
                        cols = st.columns(min(2, len(metric_cols_avail) - i))
                        for j, c in enumerate(metric_cols_avail[i:i + 2]):
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

                # 상위 25% / 하위 25% 비교 막대그래프
                st.markdown("### 🔬 생중량 상위 25% vs 하위 25% 비교")
                st.info(
                    "생중량이 무거운 상위 25% 개체와 가벼운 하위 25% 개체를 모아 "
                    "각 생육 지표의 평균을 비교한 그래프예요."
                )
                fig_bar = make_top_bottom_bar(growth_df)
                if fig_bar is not None:
                    st.plotly_chart(fig_bar, width="stretch")
                else:
                    st.info("⚠️ 비교할 데이터가 충분하지 않아요. (개체 수가 너무 적거나 생중량 열이 없음)")

        except Exception as e:
            st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    #  탭 3: 생육 지표와 생중량의 상관관계 및 예측
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
                    st.info("⚠️ 상관분석에 사용할 생육 지표(잎 수·지상부 길이·지하부 길이) 열이 없어요.")
                else:
                    st.markdown("### 🔗 생육 지표와 생중량의 상관계수 비교")
                    st.info(
                        "**상관계수(r)**는 -1에서 1 사이의 값으로,  "
                        "1에 가까울수록 함께 증가하고, -1에 가까울수록 함께 감소하며,  "
                        "0에 가까우면 관계가 거의 없다는 뜻이에요."
                    )

                    corr_results = []
                    for p in predictors:
                        x = growth_df[p].dropna()
                        y = growth_df.loc[x.index, "생중량"].dropna()
                        if len(x) >= 2 and len(y) >= 2:
                            common = x.index.intersection(y.index)
                            if len(common) >= 2:
                                r = np.corrcoef(x.loc[common], y.loc[common])[0, 1]
                                corr_results.append((p, r))

                    if corr_results:
                        fig_corr = make_corr_bar(corr_results, "생육 지표 vs 생중량 상관계수 (r)")
                        if fig_corr is not None:
                            st.plotly_chart(fig_corr, width="stretch")

                        st.markdown("---")
                        st.markdown("### 📈 산점도 및 회귀 추세선")
                        st.info(
                            "각 생육 지표와 생중량의 관계를 점과 추세선으로 나타낸 그래프예요.  "
                            "**추세선**은 np.polyfit으로 구한 직선이에요 (px.scatter의 trendline 인자는 statsmodels가 필요해 사용하지 않았어요)."
                        )

                        for p, r in corr_results:
                            common = growth_df[p].dropna().index.intersection(
                                growth_df["생중량"].dropna().index
                            )
                            if len(common) < 2:
                                continue
                            x_s = growth_df.loc[common, p]
                            y_s = growth_df.loc[common, "생중량"]
                            fig_scatter, r_val, (slope, intercept) = make_scatter_with_trend(
                                x_s, y_s, p, "생중량 (g)", color="#2c3e50"
                            )
                            if fig_scatter is not None:
                                st.markdown(f"#### {p} vs 생중량  (상관계수 r = {r_val:.3f})")
                                st.plotly_chart(fig_scatter, width="stretch")

                    # 잎 수 예측 회귀식 및 R²
                    st.markdown("---")
                    st.markdown("### 🔮 잎 수를 이용한 생중량 예측")
                    if "잎 수" in growth_df.columns:
                        common = growth_df["잎 수"].dropna().index.intersection(
                            growth_df["생중량"].dropna().index
                        )
                        if len(common) >= 2:
                            x_l = growth_df.loc[common, "잎 수"]
                            y_l = growth_df.loc[common, "생중량"]
                            slope_l, intercept_l = np.polyfit(x_l, y_l, 1)
                            y_pred = slope_l * x_l + intercept_l
                            ss_res = ((y_l - y_pred) ** 2).sum()
                            ss_tot = ((y_l - y_l.mean()) ** 2).sum()
                            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                            st.success(
                                f"**예측 회귀식:**  생중량(g) = {slope_l:.3f} × 잎 수 + {intercept_l:.3f}\n\n"
                                f"**결정계수 (R²):** {r2:.3f}\n\n"
                                f"R²는 0~1 사이의 값으로, 1에 가까울수록 잎 수로 생중량을 잘 예측할 수 있다는 뜻이에요."
                            )
                        else:
                            st.info("⚠️ 잎 수와 생중량 값이 충분하지 않아 회귀식을 구할 수 없어요.")
                    else:
                        st.info("⚠️ 잎 수 열이 없어 회귀 예측을 할 수 없어요.")

                    # 다운로드 버튼
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
