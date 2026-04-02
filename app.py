"""Data Quality Agent - Streamlit UI"""

import streamlit as st
import pandas as pd
import json

from engine.schema_detector import detect_schema
from engine.issue_detector import detect_all_issues, ISSUE_TYPES, ISSUE_COLORS, Issue
from engine.risk_scorer import score_issues, RISK_LEVELS, RISK_COLORS
from engine.excel_exporter import export_to_excel

# ── Page Config ────────────────────────────────────────────────────

st.set_page_config(
    page_title="Data Quality Agent",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2F4F4F;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2F4F4F;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #888;
        margin-top: 0.3rem;
    }
    .risk-critical { color: #800020; font-weight: 700; }
    .risk-high { color: #FF0000; font-weight: 700; }
    .risk-medium { color: #DAA520; font-weight: 600; }
    .risk-low { color: #8B8000; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────

st.markdown('<div class="main-header">Data Quality Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Veri kalitesi sorunlarını otomatik tespit edin, kategorize edin ve önceliklendirin.</div>', unsafe_allow_html=True)

# ── Sidebar: File Upload & Rules ───────────────────────────────────

with st.sidebar:
    st.header("Dosya Yükleme")
    uploaded_file = st.file_uploader(
        "CSV veya Excel dosyası yükleyin",
        type=["csv", "xlsx", "xls"],
        help="Maksimum 200MB"
    )

    st.divider()
    st.header("Kullanıcı Tanımlı Kurallar")
    st.caption("Belirli kolonlar için doğrulama kuralları tanımlayabilirsiniz.")

    use_custom_rules = st.toggle("Özel kural tanımla", value=False)
    user_rules = {}

    if use_custom_rules:
        rules_json = st.text_area(
            "Kuralları JSON formatında girin",
            value='{\n  "column_name": {\n    "min_value": 0,\n    "max_value": 100,\n    "regex": "^[A-Z]"\n  }\n}',
            height=200,
            help="Desteklenen alanlar: dtype, regex, min_length, max_length, min_value, max_value, allowed_values"
        )
        try:
            user_rules = json.loads(rules_json)
            st.success("Kurallar geçerli ✓")
        except json.JSONDecodeError as e:
            st.error(f"JSON hatası: {e}")
            user_rules = {}

# ── Main Content ───────────────────────────────────────────────────

if uploaded_file is None:
    st.info("Başlamak için sol panelden bir dosya yükleyin.")

    # Landing page info
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Tespit Edilen Sorun Tipleri")
        for itype, label in ISSUE_TYPES.items():
            color = ISSUE_COLORS[itype]
            st.markdown(
                f'<span style="background-color:#{color}; padding:2px 8px; border-radius:4px; '
                f'margin-right:8px; font-size:0.85rem;">{label}</span>',
                unsafe_allow_html=True,
            )
            st.write("")

    with col2:
        st.markdown("### Risk Seviyeleri")
        risk_desc = {
            "LOW": "Küçük sorun, düşük etki",
            "MEDIUM": "Dikkat gerektirir ama kritik değil",
            "HIGH": "Analiz üzerinde güçlü etki",
            "CRITICAL": "Modelleme öncesi mutlaka düzeltilmeli",
        }
        for level, desc in risk_desc.items():
            color = RISK_COLORS[level]
            st.markdown(
                f'<span style="background-color:#{color}; color:{"#fff" if level in ("HIGH","CRITICAL") else "#000"}; '
                f'padding:3px 10px; border-radius:4px; font-weight:600;">{level}</span> — {desc}',
                unsafe_allow_html=True,
            )
            st.write("")

    st.stop()

# ── Load Data ──────────────────────────────────────────────────────

@st.cache_data
def load_data(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)


try:
    df = load_data(uploaded_file)
except Exception as e:
    st.error(f"Dosya okunamadı: {e}")
    st.stop()

# ── Run Analysis ───────────────────────────────────────────────────

with st.spinner("Analiz ediliyor..."):
    schema = detect_schema(df)
    issues = detect_all_issues(df, schema, user_rules if user_rules else None)
    row_risks = score_issues(issues, df)

# ── Summary Metrics ────────────────────────────────────────────────

st.markdown("---")
st.markdown("### Özet")

row_issues = [i for i in issues if i.row_idx is not None]
col_issues = [i for i in issues if i.row_idx is None]
flagged_rows = len(set(i.row_idx for i in row_issues))

risk_counts = {}
for risk in row_risks.values():
    risk_counts[risk] = risk_counts.get(risk, 0) + 1

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Toplam Satır", f"{len(df):,}")
with c2:
    st.metric("Toplam Kolon", len(df.columns))
with c3:
    st.metric("Sorunlu Satır", f"{flagged_rows:,}")
with c4:
    pct = (flagged_rows / len(df) * 100) if len(df) > 0 else 0
    st.metric("Sorun Oranı", f"%{pct:.1f}")
with c5:
    st.metric("Toplam Sorun", f"{len(issues):,}")

# Risk breakdown
st.markdown("#### 🚨 Risk Dağılımı")
rc1, rc2, rc3, rc4 = st.columns(4)
for col_widget, level in zip([rc1, rc2, rc3, rc4], RISK_LEVELS):
    with col_widget:
        count = risk_counts.get(level, 0)
        color_class = f"risk-{level.lower()}"
        st.markdown(
            f'<div class="metric-card"><div class="metric-value {color_class}">{count}</div>'
            f'<div class="metric-label">{level}</div></div>',
            unsafe_allow_html=True,
        )

# ── Schema Info ────────────────────────────────────────────────────

st.markdown("---")
with st.expander("Tespit Edilen Şema", expanded=False):
    schema_df = pd.DataFrame([
        {"Kolon": col, "Tip": dtype, "Boş Değer %": f"{df[col].isna().mean():.1%}", "Benzersiz Değer": df[col].nunique()}
        for col, dtype in schema.items()
    ])
    st.dataframe(schema_df, use_container_width=True, hide_index=True)

# ── Issue Breakdown ────────────────────────────────────────────────

st.markdown("---")
st.markdown("### Sorun Detayları")

# Group issues by type
issue_groups: dict[str, list[Issue]] = {}
for issue in issues:
    issue_groups.setdefault(issue.issue_type, []).append(issue)

tabs = st.tabs([f"{ISSUE_TYPES[k]} ({len(v)})" for k, v in issue_groups.items()] if issue_groups else ["Sorun Yok"])

if not issue_groups:
    with tabs[0]:
        st.success("Hiçbir veri kalitesi sorunu tespit edilmedi!")
else:
    for tab, (itype, group) in zip(tabs, issue_groups.items()):
        with tab:
            color = ISSUE_COLORS[itype]
            st.markdown(
                f'<span style="background-color:#{color}; padding:4px 12px; border-radius:6px; '
                f'font-weight:600;">{ISSUE_TYPES[itype]}</span>',
                unsafe_allow_html=True,
            )
            st.write("")

            row_level = [i for i in group if i.row_idx is not None]
            col_level = [i for i in group if i.row_idx is None]

            if col_level:
                st.markdown("**Kolon Seviyesi Sorunlar:**")
                for ci in col_level:
                    st.warning(f"**{ci.col}**: {ci.detail}")

            if row_level:
                st.markdown(f"**Satır Seviyesi Sorunlar:** {len(row_level)} adet")
                sample = row_level[:200]
                detail_df = pd.DataFrame([
                    {
                        "Satır": i.row_idx,
                        "Kolon": i.col,
                        "Değer": str(i.value) if i.value is not None else "",
                        "Detay": i.detail,
                        "Risk": getattr(i, "risk", ""),
                    }
                    for i in sample
                ])
                st.dataframe(detail_df, use_container_width=True, hide_index=True)
                if len(row_level) > 200:
                    st.caption(f"İlk 200 sorun gösteriliyor. Toplam: {len(row_level)}")

# ── Column-Level Issues Summary ────────────────────────────────────

if col_issues:
    st.markdown("---")
    st.markdown("### Kolon Seviyesi Sorunlar")
    for ci in col_issues:
        color = ISSUE_COLORS[ci.issue_type]
        st.markdown(
            f'<div style="background-color:#{color}22; border-left:4px solid #{color}; '
            f'padding:8px 12px; margin-bottom:8px; border-radius:4px;">'
            f'<b>{ci.col}</b> — {ISSUE_TYPES[ci.issue_type]}<br/>'
            f'<span style="color:#666;">{ci.detail}</span></div>',
            unsafe_allow_html=True,
        )

# ── Data Preview ───────────────────────────────────────────────────

st.markdown("---")
with st.expander("Veri Önizleme", expanded=False):
    st.dataframe(df.head(100), use_container_width=True)

# ── Excel Export ───────────────────────────────────────────────────

st.markdown("---")
st.markdown("### Excel Çıktısı")

if issues:
    excel_buf = export_to_excel(df, issues, row_risks)
    st.download_button(
        label="Excel Dosyasını İndir",
        data=excel_buf,
        file_name="data_quality_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
        use_container_width=True,
    )
    st.caption("Excel dosyasında sorunlu hücreler renk kodlu, Risk kolonu ise risk seviyesine göre renklendirilmiştir.")
else:
    st.info("Sorun tespit edilmediği için Excel raporu oluşturulmadı.")
