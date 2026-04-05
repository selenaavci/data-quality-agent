"""Data Quality Agent - Streamlit UI"""

import streamlit as st
import pandas as pd

from engine.schema_detector import detect_schema
from engine.issue_detector import detect_all_issues, ISSUE_TYPES, ISSUE_COLORS, Issue
from engine.risk_scorer import score_issues, RISK_LEVELS, RISK_COLORS
from engine.excel_exporter import export_to_excel
from engine.aggregator import aggregate_issues

# ── Page Config ────────────────────────────────────────────────────

st.set_page_config(
    page_title="Data Quality Agent",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (dark mode compatible) ──────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        opacity: 0.7;
        margin-bottom: 1rem;
    }
    .metric-card {
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid rgba(128,128,128,0.3);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.85rem;
        opacity: 0.6;
        margin-top: 0.3rem;
    }
    .risk-kritik { color: #C0392B; font-weight: 700; }
    .risk-yüksek { color: #E74C3C; font-weight: 700; }
    .risk-orta { color: #E67E22; font-weight: 600; }
    .risk-düşük { color: #27AE60; }
    .issue-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 4px;
    }
    .issue-card {
        padding: 8px 12px;
        margin-bottom: 8px;
        border-radius: 4px;
        border-left-width: 4px;
        border-left-style: solid;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar: Reference Info ────────────────────────────────────────

with st.sidebar:
    st.markdown("### Sorun Tipleri")
    for itype, label in ISSUE_TYPES.items():
        color = ISSUE_COLORS[itype]
        st.markdown(
            f'<span class="issue-badge" style="background-color:#{color};">{label}</span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### Risk Seviyeleri")
    risk_desc = {
        "Düşük": "Küçük sorun, düşük etki",
        "Orta": "Dikkat gerektirir ama kritik değil",
        "Yüksek": "Analiz üzerinde güçlü etki",
        "Kritik": "Modelleme öncesi mutlaka düzeltilmeli",
    }
    for level, desc in risk_desc.items():
        color = RISK_COLORS[level]
        text_color = "#1a1a1a" if level in ("Düşük", "Orta") else "#ffffff"
        st.markdown(
            f'<span style="background-color:#{color}; color:{text_color}; '
            f'padding:4px 12px; border-radius:4px; font-weight:600;">{level}</span> — {desc}',
            unsafe_allow_html=True,
        )
        st.write("")

# ── Header + File Upload ──────────────────────────────────────────

st.markdown('<div class="main-header">Data Quality Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Veri kalitesi sorunlarını otomatik tespit edin, kategorize edin ve önceliklendirin.</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "CSV veya Excel dosyası yükleyin",
    type=["csv", "xlsx", "xls"],
)

if uploaded_file is None:
    st.info("Başlamak için bir dosya yükleyin.")
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

# ── Summary Metrics ────────────────────────────────────────────────

st.markdown("---")
st.markdown("### Özet")

sc1, sc2, sc3 = st.columns(3)
with sc1:
    st.metric("Toplam Satır", f"{len(df):,}")
with sc2:
    st.metric("Toplam Kolon", len(df.columns))
with sc3:
    st.metric("Dosya", uploaded_file.name)

# ── Data Preview ───────────────────────────────────────────────────

with st.expander("Veri Önizleme", expanded=True):
    st.dataframe(df.head(100), use_container_width=True)

# ── User-Defined Rules Section ─────────────────────────────────────

st.markdown("---")
st.markdown("### Kural Tanımlama")
st.caption("Belirli kolonlar için doğrulama kuralları ekleyebilirsiniz. Kural eklemeden de analiz çalıştırabilirsiniz.")

RULE_TYPE_OPTIONS = {
    "Minimum Değer": "min_value",
    "Maksimum Değer": "max_value",
    "Minimum Uzunluk": "min_length",
    "Maksimum Uzunluk": "max_length",
    "Veri Tipi": "dtype",
    "Düzenli İfade (Regex)": "regex",
    "İzin Verilen Değerler": "allowed_values",
}

DTYPE_OPTIONS = ["int", "float", "string", "date"]

if "user_rules_list" not in st.session_state:
    st.session_state.user_rules_list = []

# Add rule form
with st.expander("Yeni Kural Ekle", expanded=False):
    rc1, rc2 = st.columns(2)
    with rc1:
        selected_col = st.selectbox("Kolon", options=list(df.columns), key="rule_col")
    with rc2:
        selected_rule_label = st.selectbox("Kural Tipi", options=list(RULE_TYPE_OPTIONS.keys()), key="rule_type")

    selected_rule_key = RULE_TYPE_OPTIONS[selected_rule_label]

    # Dynamic input based on rule type
    rule_value = None
    if selected_rule_key in ("min_value", "max_value"):
        rule_value = st.number_input(
            f"{selected_rule_label} girin",
            value=0.0,
            format="%.2f",
            key="rule_num_val",
        )
    elif selected_rule_key in ("min_length", "max_length"):
        rule_value = st.number_input(
            f"{selected_rule_label} girin",
            value=1,
            min_value=0,
            step=1,
            key="rule_len_val",
        )
    elif selected_rule_key == "dtype":
        rule_value = st.selectbox("Beklenen veri tipi", options=DTYPE_OPTIONS, key="rule_dtype_val")
    elif selected_rule_key == "regex":
        rule_value = st.text_input("Regex deseni", value="", placeholder="^[A-Z].*", key="rule_regex_val")
    elif selected_rule_key == "allowed_values":
        rule_value_str = st.text_input(
            "Virgülle ayırarak yazın",
            value="",
            placeholder="Evet, Hayır, Belirsiz",
            key="rule_allowed_val",
        )
        if rule_value_str:
            rule_value = [v.strip() for v in rule_value_str.split(",") if v.strip()]

    if st.button("Kural Ekle", type="primary"):
        if rule_value is not None and rule_value != "" and rule_value != []:
            st.session_state.user_rules_list.append({
                "col": selected_col,
                "rule_key": selected_rule_key,
                "rule_label": selected_rule_label,
                "value": rule_value,
            })
            st.rerun()
        else:
            st.warning("Bir değer girin.")

# Show existing rules
if st.session_state.user_rules_list:
    st.markdown("**Tanımlanan Kurallar:**")
    for i, rule in enumerate(st.session_state.user_rules_list):
        rc1, rc2 = st.columns([5, 1])
        with rc1:
            display_val = rule["value"]
            if isinstance(display_val, list):
                display_val = ", ".join(display_val)
            st.markdown(f"`{rule['col']}` — **{rule['rule_label']}**: `{display_val}`")
        with rc2:
            if st.button("Sil", key=f"del_rule_{i}"):
                st.session_state.user_rules_list.pop(i)
                st.rerun()

# Convert session rules to engine format
user_rules = {}
for rule in st.session_state.user_rules_list:
    col = rule["col"]
    if col not in user_rules:
        user_rules[col] = {}
    user_rules[col][rule["rule_key"]] = rule["value"]

# ── Key-Based Duplicate Detection ──────────────────────────────────

st.markdown("---")
st.markdown("### Tekrar Kontrolü (Anahtar Bazlı)")
st.caption("Tam satır tekrarları otomatik tespit edilir. Ek olarak belirli kolonlara göre tekrar araması yapabilirsiniz.")

duplicate_keys = st.multiselect(
    "Tekrar kontrolü için anahtar kolonları seçin (opsiyonel)",
    options=list(df.columns),
    default=[],
    key="dup_keys",
)

# ── Run Analysis ───────────────────────────────────────────────────

with st.spinner("Analiz ediliyor..."):
    schema = detect_schema(df)
    issues = detect_all_issues(
        df, schema,
        user_rules if user_rules else None,
        duplicate_keys if duplicate_keys else None,
    )
    row_risks = score_issues(issues, df)
    summary = aggregate_issues(issues, row_risks, len(df))

# ── Analysis Results ───────────────────────────────────────────────

st.markdown("---")
st.markdown("### Analiz Sonuçları")

row_issues = [i for i in issues if i.row_idx is not None]
col_issues = [i for i in issues if i.row_idx is None]
flagged_rows = len(set(i.row_idx for i in row_issues))

risk_counts = {}
for risk in row_risks.values():
    risk_counts[risk] = risk_counts.get(risk, 0) + 1

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Sorunlu Satır", f"{flagged_rows:,}")
with c2:
    pct = (flagged_rows / len(df) * 100) if len(df) > 0 else 0
    st.metric("Sorun Oranı", f"%{pct:.1f}")
with c3:
    st.metric("Toplam Sorun", f"{len(issues):,}")

# Risk breakdown
st.markdown("#### Risk Dağılımı")
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

# ── Aggregated Breakdown ───────────────────────────────────────────

agg_col1, agg_col2 = st.columns(2)
with agg_col1:
    st.markdown("**Sorun Tipine Göre Dağılım**")
    if summary.issues_by_type:
        type_df = pd.DataFrame([
            {"Sorun Tipi": ISSUE_TYPES.get(k, k), "Adet": v}
            for k, v in summary.issues_by_type.items()
        ])
        st.dataframe(type_df, use_container_width=True, hide_index=True)

with agg_col2:
    st.markdown("**En Sorunlu Kolonlar (Top 10)**")
    if summary.top_risky_columns:
        risky_df = pd.DataFrame([
            {"Kolon": col, "Sorun Sayısı": cnt, "En Yüksek Risk": risk}
            for col, cnt, risk in summary.top_risky_columns
        ])
        st.dataframe(risky_df, use_container_width=True, hide_index=True)

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
                f'<span class="issue-badge" style="background-color:#{color};">'
                f'{ISSUE_TYPES[itype]}</span>',
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
                        "Risk": i.risk or "",
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
            f'<div class="issue-card" style="border-left-color:#{color}; '
            f'background-color:#{color}22;">'
            f'<b>{ci.col}</b> — {ISSUE_TYPES[ci.issue_type]}<br/>'
            f'<span style="opacity:0.7;">{ci.detail}</span></div>',
            unsafe_allow_html=True,
        )

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
