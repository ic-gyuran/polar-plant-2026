import io
import os
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ── 한글 폰트 (Streamlit CSS) ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""", unsafe_allow_html=True)

HERE = os.path.dirname(os.path.abspath(__file__))

# ── CSV 인코딩 대응 읽기 ────────────────────────────────────────────────────
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

# ── 열 이름 표준화 (KeyError 방지) ──────────────────────────────────────────
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
            parsed[bad] = pd.to_datetime(parts[0], format="%Y%m%d") + \
                pd.to_timedelta(parts[1].astype(int), unit="h")
        df["time"] = parsed
    return df

# ── CSV 다운로드 헬퍼 (한글 깨짐 방지) ──────────────────────────────────────
def download_csv_button(label, df, key):
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label=label,
        data=csv_bytes,
        file_name=f"{key}_data.csv",
        mime="text/csv",
        key=key,
    )

# ── 추세선 헬퍼 ─────────────────────────────────────────────────────────────
def add_trendline(fig, x, y, color="red", dash="dash", name="추세선"):
    clean = pd.concat(
        [pd.Series(x, name="x"), pd.Series(y, name="y")], axis=1
    ).dropna()
    if len(clean) < 2:
        return
    m, b = np.polyfit(clean["x"].values, clean["y"].values, 1)
    xs = np.array([clean["x"].min(), clean["x"].max()])
    fig.add_trace(go.Scatter(
        x=xs, y=m * xs + b,
        mode="lines",
        line=dict(color=color, dash=dash, width=2),
        name=name,
        showlegend=True,
    ))

# ── 산점도 헬퍼 (추세선 포함) ───────────────────────────────────────────────
def scatter_with_trend(df, xcol, ycol, xlabel, ylabel, title):
    n_df = df[[xcol, ycol]].dropna()
    if len(n_df) < 2:
        fig = go.Figure()
        fig.update_layout(
            title=dict(
                text=f"{title} (데이터 부족)",
                font=dict(size=13, color="gray"),
            ),
            font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        )
        return fig
    r = n_df[xcol].corr(n_df[ycol])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=n_df[xcol], y=n_df[ycol],
        mode="markers",
        marker=dict(
            size=9, color="#5B9BD5",
            line=dict(width=1, color="white"),
        ),
        hovertemplate=(
            f"{xlabel}: %{{x:.1f}}<br>{ylabel}: %{{y:.1f}}"
            "<extra></extra>"
        ),
    ))
    add_trendline(fig, n_df[xcol], n_df[ycol], color="orange")
    fig.add_annotation(
        x=0.02, y=0.98,
        xref="paper", yref="paper",
        text=f"상관계수 r = {r:.2f}",
        showarrow=False,
        font=dict(size=13, color="#333"),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="#999",
        borderwidth=1,
    )
    fig.update_layout(
        title=dict(text=title, x=0.02, font=dict(size=14)),
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        margin=dict(l=60, r=40, t=50, b=60),
        width=620,
        height=460,
    )
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
#  메인 앱
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="극지식물 생육 대시보드", layout="wide")

st.title("🌱 우리 학교 극지식물 개체별 생육 차이 분석")

st.sidebar.markdown("---")
st.sidebar.header("📂 데이터 업로드")

st.sidebar.info("""
**CSV 파일을 올려주세요**
- 환경 데이터: time, temperature, humidity (pH, EC 있어도 됨)
- 생육 결과: 개체번호, 잎 수, 지상부 길이, 지하부 길이, 생중량 등
""")

env_file = st.sidebar.file_uploader("🌡️ 환경 CSV", type="csv")
growth_file = st.sidebar.file_uploader("🌿 생육 결과 CSV", type="csv")

st.sidebar.markdown("---")
st.sidebar.subheader("📏 생육 측정 항목 (그래프 표시)")
growth_cols_cfg = [
    ("생중량", "생중량 (g)"),
    ("잎 수", "잎 수 (장)"),
    ("지상부 길이", "지상부 길이 (mm)"),
    ("지하부 길이", "지하부 길이 (mm)"),
]
for key, label in growth_cols_cfg:
    st.sidebar.checkbox(label, True, key=f"gck_{key}")

st.sidebar.markdown("---")
st.sidebar.caption(
    "제작: Streamlit + pandas + Plotly  |  고등학교 통계 교육 대시보드"
)

# ── 데이터 로딩 ──────────────────────────────────────────────────────────────
with st.spinner("CSV를 읽고 표준화 중입니다…"):
    raw_env_df = read_csv(env_file) if env_file else pd.DataFrame()
    raw_growth_df = read_csv(growth_file) if growth_file else pd.DataFrame()
    env_df = std_cols(raw_env_df) if not raw_env_df.empty else pd.DataFrame()
    growth_df = std_cols(raw_growth_df) if not raw_growth_df.empty else pd.DataFrame()

# ── 학교명 추출 (파일명 앞부분) ──────────────────────────────────────────────
def school_name_from_filename(fname):
    if not fname:
        return "미설정"
    base = os.path.splitext(os.path.basename(fname))[0]
    m = re.search(r"[가-힣A-Za-z]+", base)
    return m.group(0) if m else base

school_env = school_name_from_filename(env_file.name) if env_file else ""
school_growth = school_name_from_filename(growth_file.name) if growth_file else ""
school_display = school_env or school_growth or "미설정"

st.sidebar.caption(f"🏫 감지된 학교명: **{school_display}**")

# ═══════════════════════════════════════════════════════════════════════════════
#  탭
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])

# ─────────────────────────────────────────────────────────────────────────────
#  Tab 1 — 📖 실험 개요
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    try:
        st.markdown("### 연구 배경 및 목적")
        st.info("""
**극지식물(Extremophile plants)** 은 강한 햇빛, 제한된 수분, 척박한 토양 등
극한 환경에서 자라는 식물입니다.
이 대시보드에서는 우리 학교 학생들이 직접 기른 극지식물 **개체별 생육 차이**를
통계적으로 분석합니다.

**분석 목표**
1. 개체별 생중량(식물 전체 무게)의 분포와 차이를 평균·표준편차·CV(변동계수)로
   이해한다.
2. 잎 수·지상부 길이와 생중량 사이의 상관관계를 상관계수(r)와 산점도로 살펴본다.
3. 생중량이 가장 큰 개체(상위 25%)와 가장 작은 개체(하위 25%)를 비교해
   생육 차이를 파악한다.

> 📘 **CV(변동계수)**: 표준편차를 평균으로 나눈 값. 단위가 달라도 비교 가능하며,
> 값이 클수록 개체 간 차이가 크다는 뜻입니다.
> 📘 **상관계수 r**: 두 변수가 함께 변하는 정도를 -1 ~ +1 사이로 나타낸 값.
> +1에 가까울수록 함께 증가, -1에 가까울수록 반대로 움직입니다.
""")

        # ── 데이터 요약 표 ───────────────────────────────────────────────
        st.markdown("### 📋 데이터 요약")
        if env_df.empty and growth_df.empty:
            st.info("📂 환경 CSV와 생육 CSV를 모두 올려주세요.")
        else:
            rows = []
            if not env_df.empty:
                n_env = len(env_df)
                t_min = env_df["time"].min() if "time" in env_df.columns else None
                t_max = env_df["time"].max() if "time" in env_df.columns else None
                env_items = [
                    c for c in ["temperature", "humidity", "ph", "ec"]
                    if c in env_df.columns
                ]
                rows.append({
                    "구분": "환경 데이터",
                    "개체(기록) 수": f"{n_env}건",
                    "측정 기간": (
                        f"{t_min.strftime('%Y-%m-%d')} ~ "
                        f"{t_max.strftime('%Y-%m-%d')}"
                        if t_min is not None and t_max is not None
                        else "N/A"
                    ),
                    "측정 항목": ", ".join(env_items) or "없음",
                })
            if not growth_df.empty:
                n_g = len(growth_df)
                g_items = [
                    c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량"]
                    if c in growth_df.columns
                ]
                rows.append({
                    "구분": "생육 결과 데이터",
                    "개체(기록) 수": f"{n_g}개체",
                    "측정 기간": "단일 시점 (개체별 비교)",
                    "측정 항목": ", ".join(g_items) or "없음",
                })
            if rows:
                st.dataframe(
                    pd.DataFrame(rows),
                    width="stretch",
                    use_container_width=False,
                )

        # ── 주요 지표 카드 4개 ────────────────────────────────────────────
        st.markdown("### 🏆 주요 지표")
        c1, c2, c3, c4 = st.columns(4)

        total = len(growth_df) if not growth_df.empty else 0
        c1.metric("총 개체수", f"{total}개체" if total > 0 else "0개체")

        if not env_df.empty and "temperature" in env_df.columns:
            t_mean = env_df["temperature"].dropna().mean()
            c2.metric("평균 온도", f"{t_mean:.1f}°C")
        else:
            c2.metric("평균 온도", "데이터 없음")

        if not env_df.empty and "humidity" in env_df.columns:
            h_mean = env_df["humidity"].dropna().mean()
            c3.metric("평균 습도", f"{h_mean:.1f}%")
        else:
            c3.metric("평균 습도", "데이터 없음")

        if not growth_df.empty and "생중량" in growth_df.columns:
            bw_mean = growth_df["생중량"].dropna().mean()
            c4.metric("평균 생중량", f"{bw_mean:.2f}g")
        else:
            c4.metric("평균 생중량", "데이터 없음")

    except Exception as e:
        st.warning(
            f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}"
        )

# ─────────────────────────────────────────────────────────────────────────────
#  Tab 2 — 🌡️ 환경 데이터
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    try:
        if env_df.empty:
            st.info(
                "🌡️ 환경 CSV를 사이드바에 올려주세요. "
                "환경 데이터 탭이 여기에 표시됩니다."
            )
        else:
            st.markdown("### 📈 환경 요약 (평균 · 최소 · 최대)")
            env_cols = [
                c for c in ["temperature", "humidity", "ph", "ec"]
                if c in env_df.columns
            ]
            if not env_cols:
                st.info(
                    "환경 데이터에 temperature, humidity, ph, ec 열이 "
                    "하나도 없습니다. 다른 파일을 올려보세요."
                )
            else:
                summary_rows = []
                for c in env_cols:
                    s = env_df[c].dropna()
                    if len(s) == 0:
                        continue
                    label = (
                        c.replace("temperature", "온도 (°C)")
                        .replace("humidity", "습도 (%)")
                        .replace("ph", "pH")
                        .replace("ec", "EC (mS/cm)")
                    )
                    summary_rows.append({
                        "측정 항목": label,
                        "평균": round(s.mean(), 2),
                        "최소": round(s.min(), 2),
                        "최대": round(s.max(), 2),
                        "측정 횟수": len(s),
                    })
                if summary_rows:
                    st.dataframe(
                        pd.DataFrame(summary_rows),
                        width="stretch",
                        use_container_width=False,
                    )

            # ── 일별 평균 꺾은선 ────────────────────────────────────────────
            if "time" in env_df.columns:
                env_df["date"] = env_df["time"].dt.date
                st.markdown("### 📅 날짜별 환경 변화 (일별 평균)")

                # 존재하는 열만 agg kwargs에 담음 (None 전달 금지)
                agg_kwargs = {}
                if "temperature" in env_df.columns:
                    agg_kwargs["mean_temp"] = ("temperature", "mean")
                if "humidity" in env_df.columns:
                    agg_kwargs["mean_hum"] = ("humidity", "mean")
                if "ph" in env_df.columns:
                    agg_kwargs["mean_ph"] = ("ph", "mean")
                if "ec" in env_df.columns:
                    agg_kwargs["mean_ec"] = ("ec", "mean")

                if agg_kwargs:
                    daily = env_df.groupby("date").agg(
                        **agg_kwargs
                    ).reset_index()
                    n_traces = len(agg_kwargs)
                    titles = []
                    for ck in agg_kwargs.keys():
                        if ck == "mean_temp":
                            titles.append("온도 (°C)")
                        elif ck == "mean_hum":
                            titles.append("습도 (%)")
                        elif ck == "mean_ph":
                            titles.append("pH")
                        elif ck == "mean_ec":
                            titles.append("EC (mS/cm)")

                    fig_daily = make_subplots(
                        rows=1,
                        cols=n_traces,
                        subplot_titles=titles,
                        vertical_spacing=0.12,
                    )
                    for i, ck in enumerate(agg_kwargs.keys(), start=1):
                        fig_daily.add_trace(
                            go.Scatter(
                                x=daily["date"],
                                y=daily[ck],
                                mode="lines+markers",
                                line=dict(width=2),
                                name=ck.replace("mean_", "")
                                .replace("_", " ")
                                .title(),
                            ),
                            row=1,
                            col=i,
                        )
                    fig_daily.update_layout(
                        title=dict(
                            text="일별 평균 환경 지표",
                            x=0.02,
                            font=dict(size=14),
                        ),
                        font=dict(
                            family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"
                        ),
                        height=340,
                        margin=dict(t=40, b=40, l=40, r=20),
                    )
                    st.plotly_chart(fig_daily, width="stretch")
                else:
                    st.info("일별 평균을 계산할 열이 없습니다.")

            # ── 원본 테이블 + 다운로드 ──────────────────────────────────────
            with st.expander("📄 환경 데이터 원본 테이블 보기"):
                st.dataframe(env_df, width="stretch", use_container_width=False)
                download_csv_button(
                    "⬇️ 환경 데이터 CSV 다운로드",
                    env_df,
                    "env_download",
                )
    except Exception as e:
        st.warning(
            f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}"
        )

# ─────────────────────────────────────────────────────────────────────────────
#  Tab 3 — 📊 생육 결과
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    try:
        if growth_df.empty:
            st.info(
                "🌿 생육 결과 CSV를 사이드바에 올려주세요. "
                "생육 결과 탭이 여기에 표시됩니다."
            )
        else:
            # ── 핵심 결과 카드 ──────────────────────────────────────────────
            st.markdown("### 🥇 핵심 결과")
            bc = growth_df["생중량"].dropna() if "생중량" in growth_df.columns else pd.Series(dtype=float)

            if len(bc) >= 1:
                mean_bw = bc.mean()
                max_idx = bc.idxmax()
                min_idx = bc.idxmin()
                max_row = growth_df.loc[max_idx]
                min_row = growth_df.loc[min_idx]

                c_max, c_mean, c_min = st.columns(3)
                c_max.metric(
                    "📈 최대 생중량 개체",
                    f"{max_row['생중량']:.2f}g",
                    delta=f"개체번호: {max_row['개체번호']}",
                    delta_color="normal",
                )
                c_mean.metric(
                    "📊 평균 생중량",
                    f"{mean_bw:.2f}g",
                    delta=f"개체 수: {len(bc)}개체",
                )
                c_min.metric(
                    "📉 최소 생중량 개체",
                    f"{min_row['생중량']:.2f}g",
                    delta=f"개체번호: {min_row['개체번호']}",
                    delta_color="normal",
                )
            else:
                st.info("생중량 데이터가 없어서 핵심 결과를 표시할 수 없습니다.")

            # ── 개체별 생육 비교 (막대 그래프) ──────────────────────────────
            st.markdown("### 📊 개체별 생육 비교")
            g_cols = ["생중량", "잎 수", "지상부 길이", "지하부 길이"]
            avail_g = [c for c in g_cols if c in growth_df.columns]
            if not avail_g:
                st.info("생육 데이터에 지표가 없어 비교할 수 없습니다.")
            else:
                n_plots = len(avail_g)
                n_rows = (n_plots + 1) // 2
                n_cols = 2
                subplot_titles = []
                for c in avail_g:
                    if c == "생중량":
                        subplot_titles.append("평균 생중량 (g) — ⭐ 가장 중요")
                    elif c == "잎 수":
                        subplot_titles.append("평균 잎 수 (장)")
                    elif c == "지상부 길이":
                        subplot_titles.append("평균 지상부 길이 (mm)")
                    elif c == "지하부 길이":
                        subplot_titles.append("평균 지하부 길이 (mm)")
                # 빈 칸 채우기
                while len(subplot_titles) < n_rows * n_cols:
                    subplot_titles.append("")

                fig_g = make_subplots(
                    rows=n_rows,
                    cols=n_cols,
                    subplot_titles=subplot_titles,
                    vertical_spacing=0.12,
                    horizontal_spacing=0.06,
                    specs=[[{"type": "xy"} for _ in range(n_cols)]
                           for _ in range(n_rows)],
                )
                for i, c in enumerate(avail_g):
                    row_i = i // n_cols + 1
                    col_i = i % n_cols + 1
                    bc_sorted = (
                        growth_df[[c, "개체번호"]]
                        .dropna(subset=[c])
                        .sort_values(c, ascending=False)
                    )
                    fig_g.add_trace(
                        go.Bar(
                            x=bc_sorted.index,
                            y=bc_sorted[c],
                            orientation="h",
                            marker_color="#4CAF50",
                            marker_line_color="white",
                            marker_line_width=0.5,
                        ),
                        row=row_i,
                        col=col_i,
                    )
                fig_g.update_layout(
                    title=dict(
                        text="개체별 평균 지표 비교 (정렬: 값 큰 순 → 작은 순)",
                        x=0.02,
                        font=dict(size=13),
                    ),
                    font=dict(
                        family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"
                    ),
                    height=540,
                    margin=dict(l=120, r=40, t=40, b=40),
                )
                st.plotly_chart(fig_g, width="stretch")

            # ── 생중량 분포 (박스플롯) ──────────────────────────────────────
            if "생중량" in growth_df.columns:
                st.markdown("### 📦 생중량 분포 (박스플롯)")
                bw_clean = growth_df["생중량"].dropna()
                if len(bw_clean) >= 2:
                    q75 = bw_clean.quantile(0.75)
                    q25 = bw_clean.quantile(0.25)
                    fig_box = go.Figure()
                    fig_box.add_trace(go.Box(
                        y=bw_clean,
                        name="생중량 분포",
                        fillcolor="#5B9BD5",
                        line_color="white",
                        boxmean=True,
                    ))
                    fig_box.add_hline(
                        y=q75,
                        line_dash="dash",
                        line_color="orange",
                        annotation_text="상위 25% 경계 (Q3)",
                    )
                    fig_box.add_hline(
                        y=q25,
                        line_dash="dash",
                        line_color="orange",
                        annotation_text="하위 25% 경계 (Q1)",
                    )
                    fig_box.update_layout(
                        title=dict(
                            text="생중량 분포 — 박스플롯 "
                                 "(상자: Q1~Q3, 선: 중앙값)",
                            x=0.02,
                            font=dict(size=13),
                        ),
                        yaxis_title="생중량 (g)",
                        font=dict(
                            family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"
                        ),
                        margin=dict(l=60, r=40, t=50, b=50),
                        height=360,
                    )
                    st.plotly_chart(fig_box, width="stretch")
                else:
                    st.info(
                        "생중량 값이 2개 미만이라 박스플롯을 그릴 수 없습니다."
                    )

            # ── 상관관계 분석 (산점도 2개 + 추세선) ────────────────────────
            st.markdown("### 🔗 상관관계 분석 (산점도 + 추세선)")
            corr_pairs = [
                ("잎 수", "생중량", "잎 수 (장)", "생중량 (g)"),
                ("지상부 길이", "생중량", "지상부 길이 (mm)", "생중량 (g)"),
            ]
            for xcol, ycol, xl, yl in corr_pairs:
                if xcol in growth_df.columns and ycol in growth_df.columns:
                    fig_sc = scatter_with_trend(
                        growth_df, xcol, ycol, xl, yl,
                        f"{xl} vs {yl} 산점도",
                    )
                    st.plotly_chart(fig_sc, width="stretch")
                else:
                    missing = [c for c in [xcol, ycol]
                               if c not in growth_df.columns]
                    st.info(
                        f"📌 '{', '.join(missing)}' 열이 없어 "
                        "이 산점도를 표시할 수 없습니다."
                    )

            # ── 상위 25% vs 하위 25% 비교 ──────────────────────────────────
            if "생중량" in growth_df.columns:
                bw = growth_df["생중량"].dropna()
                if len(bw) >= 2:
                    q25 = bw.quantile(0.25)
                    q75 = bw.quantile(0.75)
                    high = (
                        growth_df[growth_df["생중량"] >= q75]
                        .dropna(subset=["생중량"])
                    )
                    low = (
                        growth_df[growth_df["생중량"] <= q25]
                        .dropna(subset=["생중량"])
                    )
                    if len(high) > 0 and len(low) > 0:
                        st.markdown(
                            "### 🏅 생중량 상위 25% vs 하위 25% 비교"
                        )
                        c_hi, c_lo = st.columns(2)

                        c_hi.markdown(f"""
<div style="background:#e8f5e9; border-radius:8px; padding:10px;">
<strong>🔵 상위 25% (생중량 ≥ {q75:.2f}g)</strong><br>
개체 수: **{len(high)}개체**<br>
평균 생중량: **{high['생중량'].mean():.2f}g**<br>
평균 잎 수: **{high['잎 수'].mean():.1f}장**<br>
평균 지상부 길이: **{high['지상부 길이'].mean():.1f}mm**
</div>
""")

                        c_lo.markdown(f"""
<div style="background:#fff3e0; border-radius:8px; padding:10px;">
<strong>🟠 하위 25% (생중량 ≤ {q25:.2f}g)</strong><br>
개체 수: **{len(low)}개체**<br>
평균 생중량: **{low['생중량'].mean():.2f}g**<br>
평균 잎 수: **{low['잎 수'].mean():.1f}장**<br>
평균 지상부 길이: **{low['지상부 길이'].mean():.1f}mm**
</div>
""")
                    else:
                        st.info("상위 25% 또는 하위 25% 개체가 없습니다.")

            # ── 원본 테이블 + 다운로드 ──────────────────────────────────────
            with st.expander("📄 생육 결과 데이터 원본 테이블 보기"):
                st.dataframe(growth_df, width="stretch", use_container_width=False)
                download_csv_button(
                    "⬇️ 생육 결과 CSV 다운로드",
                    growth_df,
                    "growth_download",
                )
    except Exception as e:
        st.warning(
            f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}"
        )

st.caption("---")
st.caption(
    "📚 통계 용어 해설: 평균(값들의 중심), 표준편차(값들이 평균에서 얼마나 "
    "퍼져 있는지), CV=표준편차÷평균(단위 다른 데이터 비교용), "
    "상관계수 r=-1~+1(두 변수의 함께 변하는 정도)"
)
