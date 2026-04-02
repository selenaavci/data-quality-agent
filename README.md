# 🧪 Data Quality Agent

## 📌 Project Purpose

Data Quality Agent is designed to automatically detect, categorize, and prioritize data quality issues before any downstream analysis or modeling steps.  
It minimizes manual data inspection by highlighting only problematic records and assigning risk levels to each issue.  
The agent serves as a foundational layer within the AI Hub, ensuring that all analytical processes operate on clean, reliable, and interpretable data.

---

## ⚙️ Core Capabilities

The agent analyzes uploaded datasets (CSV / Excel) and detects multiple types of data quality issues at both column and row levels.

### 🔍 Detected Problem Types

- **Missing Values**
  - Null, NaN, empty strings
  - Highlighted as **Yellow**

- **Format Issues**
  - Invalid data types (e.g. text in numeric fields)
  - Date parsing failures
  - Regex mismatches (if rule provided)
  - Highlighted as **Orange**

- **Data Type Drift**
  - Mixed data types within a column (e.g. int + string)
  - Silent schema inconsistencies
  - Highlighted as **Dark Orange**

- **Semantic Inconsistency**
  - Different representations of the same value
  - Example: "İstanbul", "IST", "Ist."
  - Highlighted as **Purple**

- **Duplicate Records**
  - Full row duplicates
  - Optional key-based duplicates
  - Highlighted as **Blue**

- **Range Violations**
  - Values outside logical or business-defined ranges
  - Example: negative age, percentage > 100
  - Highlighted as **Teal**

- **Sparse Columns / High Missing Columns (Column-Level Flag)**
  - Columns with very high missing ratios
  - Low usability features
  - Highlighted as **Pink**

- **Meaningless / Low-Quality Features**
  - Constant columns
  - ID-like or non-informative columns
  - High cardinality categorical columns
  - Highlighted as **Light Pink**

---

## 🚨 Risk Scoring System

Each detected issue is assigned a **risk level** to help prioritize manual review.

| Risk Level | Meaning |
|-----------|--------|
| LOW       | Minor issue, low impact |
| MEDIUM    | Needs attention but not critical |
| HIGH      | Strong impact on analysis |
| CRITICAL  | Must be fixed before any modeling |

### 🎨 Risk Color Encoding (Excel Export)

| Risk Level | Color        |
|------------|-------------|
| LOW        | Beige       |
| MEDIUM     | Light Yellow (non-neon) |
| HIGH       | Red         |
| CRITICAL   | Bordeaux    |

> ⚠️ Note: Risk colors are strictly separated from issue type colors.

---

## 🧠 Analysis Workflow

1. **File Upload**
   - User uploads CSV or Excel file
   - File is parsed into a DataFrame

2. **Schema Detection**
   - Columns are classified:
     - Numerical
     - Categorical
     - Date
     - Text

3. **Column Quality Analysis**
   - Missing ratio
   - Unique value ratio
   - Data type consistency
   - Pattern detection

4. **Rule-Based Validation**
   - Built-in validation rules are applied
   - Optional user-defined rules are included

5. **Issue Detection Engine**
   - Row-level and column-level issues are detected
   - Each issue is categorized and labeled

6. **Risk Assignment**
   - Each issue is assigned a risk level
   - Enables prioritization of problematic data

7. **Visualization**
   - Issues are grouped and color-coded
   - Risk levels are visually distinguished

8. **Excel Export**
   - Highlighted rows based on issue types
   - Additional **Risk column** included
   - Risk column is color-coded based on severity

---

## 🎨 Issue Color Encoding

| Issue Type                  | Color       |
|----------------------------|------------|
| Missing Values             | Yellow     |
| Format Issues              | Orange     |
| Data Type Drift            | Dark Orange|
| Semantic Inconsistency     | Purple     |
| Duplicate Records          | Blue       |
| Range Violations           | Teal       |
| Sparse Columns             | Pink       |
| Meaningless Features       | Light Pink |

---

## 🧩 User-Defined Rules

Users can optionally define validation rules for specific columns:

- Expected data type (int, float, date, string)
- Regex pattern (email, phone number, etc.)
- Min / Max length
- Allowed value constraints
- Range limits

These rules are integrated into the validation engine and affect both issue detection and risk scoring.

---

## 📤 Output

The agent generates an **Excel file** with:

- Highlighted problematic rows based on issue types
- A dedicated **Risk column**
- Risk-based color coding
- Structured format for efficient manual review

👉 Users only focus on flagged rows instead of reviewing the entire dataset.

---

## 💡 Example Use Case

A user uploads a dataset with 50,000 records.

- 3,500 rows are flagged with issues
- Each issue is categorized and assigned a risk level
- High and critical issues are immediately visible
- Manual effort is reduced by focusing only on problematic rows

---

## 🏗️ System Architecture

- **Frontend**: Streamlit UI
- **Backend**: Python (Pandas-based processing)
- **Validation Engine**: Rule-based quality checks
- **Scoring Engine**: Risk-based prioritization system
- **Export Engine**: Excel generation with styled formatting (openpyxl)

---

## 🚀 Position in AI Hub

Data Quality Agent acts as a **pre-processing gatekeeper** for:

- Anomaly Detection Agent
- Clustering / Segmentation Agent
- Future ML pipelines

Ensures all downstream systems operate on **validated, high-quality data**.

---

## 🔮 Future Enhancements (Phase 2)

- Automated data cleaning suggestions
- Smart missing value imputation
- Column-level quality scoring dashboards
- Pipeline integration with other agents
- Optional LLM-based semantic validation

---

## 🧭 Key Value

Instead of:
> “Review the entire dataset manually”

This agent enables:
> “Focus only on high-risk, problematic data”

It transforms data quality control into a **prioritized, structured, and efficient workflow**.
