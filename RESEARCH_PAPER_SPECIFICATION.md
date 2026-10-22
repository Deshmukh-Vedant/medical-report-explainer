# RESEARCH PAPER TECHNICAL SPECIFICATION & SYSTEM DOSSIER
## Project: AI Medical Report Explainer & Machine Learning Clinical Risk Classifier

> **Document Purpose**: This comprehensive technical and clinical dossier contains complete specifications, mathematical formulations, architecture diagrams, datasets provenance, machine learning metrics, security controls, and experimental evaluations for the Medical Report Explainer system. It is structured to directly serve as reference material for writing an academic research paper (IEEE / ACM / Springer / JAMIA format).

---

## 1. Title & Abstract Formulation

- **Suggested Working Titles**:
  1. *A Hybrid Machine Learning and Large Language Model Framework for Automated Medical Report Simplification and Multi-Tiered Clinical Risk Stratification*
  2. *MedExplain AI: Bridging Health Literacy Gaps via Regex-Anchored Parameter Normalization, Random Forest Risk Classification, and Gemini-Powered Plain-Language Synthesis*
  3. *An End-to-End Fault-Tolerant Clinical Decision Support System for Laboratory Report Ingestion, Parameter Extraction, and Risk Stratification*

- **Abstract Summary**:
  Routine laboratory reports (such as Complete Blood Counts, Comprehensive Metabolic Panels, and Lipid Profiles) are critical diagnostic instruments, yet their technical nomenclature and reference intervals frequently induce confusion or medical anxiety among lay patients. We propose an end-to-end, privacy-conscious full-stack healthcare framework that ingests heterogeneous digital and scanned PDF reports, robustly extracts clinical parameters using a dual-engine fallback (digital stream and optical character recognition), maps and normalizes measurements across clinical scales, classifies clinical risk into three tiers (*Low*, *Moderate*, *High*) using a trained Random Forest pipeline with missing-value imputation, and delivers simple-language explanations using Google's Gemini 2.5 Flash Large Language Model (with deterministic offline fallbacks). Evaluated across 12,000 clinically grounded synthetic patient profiles and verified via automated test suites, our ML pipeline achieves 99.34% cross-validation accuracy ($F_1 = 0.98$ macro average) while enforcing strict boundaries between automated decision support and licensed medical consultation.

---

## 2. Problem Statement & Research Objectives

1. **The Patient Health Literacy Gap**: Over 35% of adult patients struggle to interpret basic diagnostic markers (e.g., ALT, Creatinine, HbA1c), leading to either ungrounded panic or failure to seek necessary follow-up care.
2. **Heterogeneity of Report Formats**: Medical laboratories lack unified layout standards. Reports arrive as searchable digital PDFs, multi-column tables, scanned image-only documents, or hybrid files with non-standard font encodings.
3. **Black-Box AI vs. Rule-Based Rigor**: Pure LLM interpretations often hallucinate clinical values or provide vague disclaimers. Conversely, pure rule-based flags lack natural-language clarity. Our hybrid system fuses **deterministic regex parameter extraction**, **scikit-learn risk prediction**, and **LLM contextualization**.
4. **Resilience & Availability in Clinical Systems**: Online cloud services can fail or rate-limit. The system must incorporate deterministic offline analytical fallbacks so users are never left without insights.

---

## 3. High-Level System Architecture

```
                                      +-----------------------------------------+
                                      |            Client Web Browser           |
                                      |  (Bootstrap 5.3, Vanilla JS, Chart.js)  |
                                      +-------------------+---------------------+
                                                          |
                                                    HTTP / HTTPS
                                                  (JWT / REST API)
                                                          |
                                                          v
                                      +-----------------------------------------+
                                      |          FastAPI Backend Gateway        |
                                      |  (Rate Limiting, CORS, Static Serving)  |
                                      +-----+---------------+-------------+-----+
                                            |               |             |
                         +------------------+               |             +------------------+
                         |                                  |                                |
                         v                                  v                                v
       +---------------------------------+  +-------------------------------+  +-----------------------------+
       |   Document Processing Pipeline  |  |    Authentication & Profile   |  |   Database & Data Models    |
       |  - PyMuPDF Encryption Precheck  |  |  - Bcrypt / Argon2 Hashing    |  |  - SQLite + SQLAlchemy ORM  |
       |  - pdfplumber digital stream    |  |  - Single-Use HMAC-SHA256     |  |  - Users & Reports Tables   |
       |  - PyMuPDF vector text fallback |  |    Password Reset Tokens      |  |  - Persisted ML & AI JSON   |
       |  - PyTesseract OCR (Scanned)    |  |  - Async SMTP / Console Mail  |  +-----------------------------+
       +----------------+----------------+  +-------------------------------+
                        |
                        v
       +---------------------------------+
       |    Clinical Parameter Extractor |
       |  - Proximity Lookahead Window   |
       |  - Scale Normalization (10^3/uL)|
       |  - Plausible Sanity Bounds      |
       +----------------+----------------+
                        |
            +-----------+-----------+
            |                       |
            v                       v
+-----------------------+ +-----------------------------------+
|  ML Risk Predictor    | |  AI Report Explainer (Gemini LLM) |
|  - Median Imputation  | |  - Gemini 2.5 Flash API           |
|  - Random Forest      | |  - Deterministic Offline Fallback |
|    Classifier         | |  - Bilingual English/Marathi      |
|  - Confidence Scoring | |  - Red Flags & Emergency Alerts   |
+-----------+-----------+ +-----------------+-----------------+
            |                               |
            +---------------+---------------+
                            |
                            v
             +------------------------------+
             | Combined Patient Dashboard   |
             |  - Health Score Ring         |
             |  - 3-Tier Risk Badge         |
             |  - Interactive ML Calculators|
             |  - Web Speech Read-Aloud     |
             +------------------------------+
```

---

## 4. Clinical Parameter Extraction & Normalization Engine

The system focuses on 10 clinically representative biomarkers spanning haematology, renal function, hepatic function, metabolism, and lipidology:

| Parameter | Clinical Domain | Standard Units | Medically Plausible Sanity Range | Scaling / Normalization Logic |
|---|---|---|---|---|
| **Hemoglobin** (`hemoglobin`) | Haematology | $\text{g/dL}$ | $3.0 - 25.0$ | Direct regex extraction; detects $\text{g/dL}$ or $\text{g/L}$ |
| **White Blood Cells (WBC)** (`wbc`) | Immune / Haematology | $10^3/\mu\text{L}$ | $0.5 - 100.0$ | If raw $\text{value} > 200$, scaled as $\text{raw} / 1000$ (e.g. $6,800 \rightarrow 6.8$) |
| **Platelets** (`platelets`) | Coagulation | $10^3/\mu\text{L}$ | $10.0 - 1,500.0$ | If raw $\text{value} > 2,000$, scaled as $\text{raw} / 1000$ (e.g. $245,000 \rightarrow 245.0$) |
| **Fasting Blood Glucose** (`glucose`) | Metabolism / Diabetes | $\text{mg/dL}$ | $20.0 - 800.0$ | Normalized to $\text{mg/dL}$ |
| **Serum Creatinine** (`creatinine`) | Renal Function | $\text{mg/dL}$ | $0.1 - 30.0$ | Direct regex detection within clinical range |
| **Alanine Transaminase (ALT/SGPT)** (`alt`) | Hepatic Function | $\text{U/L}$ | $1.0 - 5,000.0$ | Direct enzyme activity extraction |
| **Aspartate Transaminase (AST/SGOT)** (`ast`) | Hepatic Function | $\text{U/L}$ | $1.0 - 5,000.0$ | Direct enzyme activity extraction |
| **Total Protein** (`total_protein`) | Nutritional / Hepatic | $\text{g/dL}$ | $2.0 - 15.0$ | Direct plasma concentration extraction |
| **Glycated Hemoglobin (HbA1c)** (`hba1c`) | Glycaemic Control | $\%$ | $2.0 - 20.0$ | Preserves percentage scale |
| **Total Cholesterol** (`cholesterol`) | Cardiovascular / Lipid | $\text{mg/dL}$ | $30.0 - 700.0$ | Normalized to $\text{mg/dL}$ |

### Algorithmic Extraction Details:
1. **Target Anchor Matching**: Case-insensitive regular expressions match biomarker aliases (e.g., `\bwbc\b`, `\bwhite\s+blood\s+(?:cell|count)\b`, `\bleukocyte\b`).
2. **Proximity Lookahead Window**: Scans up to 80 characters immediately succeeding the regex match end.
3. **Single-Line Boundary Priority**: Isolates the current line (`after_text.split('\n')[0]`) to avoid bleeding values across rows in multi-line lab tables.
4. **Pre-Sanity Unit Transformation**: Dynamic re-scaling occurs *prior* to boundary rejection, preventing valid scaled values from being misidentified as out-of-bounds.

---

## 5. Machine Learning Model Architecture & Training Methodology

### 5.1 Dataset Construction & Provenance
Because real-world Electronic Health Record (EHR) datasets are subject to strict HIPAA/GDPR constraints, a clinically calibrated synthetic dataset of **$N = 12,000$ patient instances** was algorithmically generated based on recognized clinical reference distributions:
- **Haematological norms**: World Health Organization (WHO) adult reference ranges.
- **Diabetes & Glycaemic thresholds**: American Diabetes Association (ADA) diagnostic standards.
- **Lipid thresholds**: National Cholesterol Education Program (NCEP) Adult Treatment Panel III (ATP III).
- **Simulated Missingness**: A uniform 10% dropout ($P(\text{NaN}) = 0.10$) was injected across all features to simulate OCR extraction limitations in imperfect documents.

### 5.2 Ground-Truth Clinical Risk Logic
Ground truth risk classifications $y \in \{0, 1, 2\}$ correspond to:
- **0: Low Risk**: All clinical parameters reside within normal physiological ranges.
- **1: Moderate Risk**: Borderline abnormalities (e.g., Glucose $110-300\text{ mg/dL}$, HbA1c $5.7-9.0\%$, Total Cholesterol $200-350\text{ mg/dL}$, Hemoglobin $7-11\text{ g/dL}$).
- **2: High Risk**: Severe clinical anomalies or critical alerts (e.g., Glucose $> 300$ or $< 40\text{ mg/dL}$, HbA1c $> 9.0\%$, Creatinine $> 5.0\text{ mg/dL}$, Hemoglobin $< 7.0\text{ g/dL}$, ALT/AST $> 200\text{ U/L}$).

### 5.3 Pipeline Design
```python
Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])
```
- **Imputation**: `SimpleImputer` replaces missing parameters with the median of training distributions, ensuring inference never fails when reports omit specific assays.
- **Model Choice**: `RandomForestClassifier` was chosen for its non-linear decision surface, resilience against collinearity, robust handling of missing features, and inherent resistance to overfitting on high-dimensional clinical markers.

### 5.4 Empirical Evaluation & Measured Results
- **Stratified 5-Fold Cross-Validation Accuracy**:
  $$\text{Mean CV Accuracy} = 99.34\% \pm 0.16\%$$
- **Test Set Performance ($n = 2,400$, 20% holdout)**:

| Risk Class | Precision | Recall | $F_1$-Score | Support |
|---|---|---|---|---|
| **Low (0)** | 0.99 | 0.97 | 0.98 | 136 |
| **Moderate (1)** | 1.00 | 0.99 | 1.00 | 1,990 |
| **High (2)** | 0.95 | 0.99 | 0.97 | 274 |
| **Overall Accuracy** | — | — | **0.99** | **2,400** |
| **Macro Average** | **0.98** | **0.98** | **0.98** | 2,400 |
| **Weighted Average** | **0.99** | **0.99** | **0.99** | 2,400 |

- **Confidence Calibrations**: Inference returns the posterior class probability:
  $$\hat{y} = \arg\max_k P(y=k \mid \mathbf{x}), \quad \text{Confidence} = \max_k P(y=k \mid \mathbf{x})$$

---

## 6. Document Ingestion, Fault-Tolerant OCR, and Preprocessing

Medical reports arrive as scanned or digitally rendered PDFs. The ingestion pipeline executes a 4-stage tiered workflow:

1. **Pre-Flight Structural Validation**:
   - MIME type and extension validation (`application/pdf`, `.pdf`).
   - Size limitation verification ($\le 20\text{ MB}$).
   - **Encryption & Password Guard**: Opens with PyMuPDF (`fitz.open()`) to check `doc.is_encrypted` and `doc.needs_pass`. Protected files immediately return HTTP 422 with actionable guidance.
2. **Primary Digital Extraction (`pdfplumber`)**:
   - Extracts character vectors and layout boxes for native digital PDFs.
3. **Secondary Vector Stream Fallback (`PyMuPDF`)**:
   - For pages where `pdfplumber` yields fewer than 20 characters (due to non-standard CID encodings), direct PyMuPDF text stream parsing is attempted without external dependencies.
4. **Tertiary Optical Character Recognition (`PyTesseract`)**:
   - For image-based scanned reports, pages are rendered at $2.0\times$ zoom factor into memory-buffered PNG images and processed via Tesseract OCR engine.
   - If Tesseract is unavailable on the local host, the pipeline gracefully falls back to structured placeholder processing without crashing.

---

## 7. Natural Language Generation & Clinical Explanation Architecture

### 7.1 Primary LLM Synthesis (Gemini 2.5 Flash)
- **Model**: `gemini-1.5-flash` / `gemini-2.5-flash` via `google-generativeai`.
- **System Prompt Framing**: Constrains the model to output strict valid JSON matching the `ReportAnalysis` Pydantic schema:
  - `patient_summary` (Plain English breakdown for Grade-6 reading level)
  - `overall_health_summary` (Concise clinical overview)
  - `detected_tests` (Array of test name, normal range, patient value, status: Green/Yellow/Red, plain-language meaning, possible causes, lifestyle/diet advice)
  - `health_score` (Numeric score $0-100$)
  - `risk_level` (`Low`, `Moderate`, `High`)
  - `diet_advice`, `exercise_advice`
  - `doctor_recommendation`
  - `red_flags`, `emergency_signs`
- **Output Sanitization**: Robust regex cleaning strips Markdown backtick fences (````json ... ````) and performs salvage parsing for truncated blocks.

### 7.2 Deterministic Offline Fallback Parser
To maintain strict system availability during network partitions or quota exhaustion, an offline rule-based parser automatically activates:
- Scans for standard laboratory markers via local regex.
- Categorizes values according to established clinical reference intervals.
- Dynamically calculates the Health Score:
  $$\text{Score} = \max\left(45, \; 90 - 12 \times N_{\text{abnormal}}\right)$$
- Assigns deterministic risk tiers:
  $$N_{\text{abnormal}} \ge 3 \implies \text{High}, \quad N_{\text{abnormal}} \ge 1 \implies \text{Moderate}, \quad \text{Otherwise} \implies \text{Low}$$

### 7.3 Accessibility & Cross-Lingual Features
- **Bilingual Translation**: Supports instantaneous English $\leftrightarrow$ Marathi translations via LLM translation prompts.
- **Auditory Synthesis**: Utilizes the browser-native W3C Web Speech API (`SpeechSynthesisUtterance`) to read explanations aloud for visually impaired or low-literacy users.

---

## 8. Security, Privacy, and Authentication Engineering

- **Password Storage**: Passwords hashed using standard one-way derivation functions (`passlib` with `bcrypt`).
- **Cryptographic Password Reset Flow**:
  - Reset tokens generated using 48-byte URL-safe cryptographic entropy: `secrets.token_urlsafe(48)`.
  - Database stores only the HMAC-SHA256 digest: $\text{hash} = \text{SHA256}(\text{token}_{\text{raw}})$.
  - Time-to-Live (TTL) strictly enforced at 30 minutes.
  - **Single-Use Invalidation**: `reset_token_hash` and `reset_token_expires` are immediately nulled upon successful password update, preventing replay attacks.
  - Constant generic response messages prevent account enumeration across `/forgot-password`.
- **Session Integrity**: Stateless JSON Web Tokens (`python-jose`) with HS256 algorithm and configurable expiration.
- **Access Control & RBAC**: Strict user-level record filtering (`Report.user_id == current_user.id`) enforced on all report queries, preventing Insecure Direct Object References (IDOR).
- **Rate Limiting**: Integrated `slowapi` rate limiter guarding authentication endpoints against brute-force attacks.

---

## 9. Comprehensive Automated Test Matrix

The test suite consists of 16 end-to-end integration and unit tests passing with **100% success rate**:

| Test Identifier | Category | Target Component | Description / Assertion |
|---|---|---|---|
| `test_build_fallback_analysis` | AI Engine | `routes.analysis` | Validates deterministic fallback generation when AI is offline |
| `test_model_is_available` | Machine Learning | `ml.predictor` | Asserts serialized model artifact exists and unpickles cleanly |
| `test_extract_lab_params` | Extraction | `ml.extractor` | Verifies regex extraction and unit scaling on realistic reports |
| `test_sanity_check_rejects` | Data Cleaning | `ml.extractor` | Rejects biologically impossible values (e.g. Glucose = $5$ or $9,500\text{ mg/dL}$) |
| `test_predict_normal_profile` | Machine Learning | `ml.predictor` | Asserts normal panel classifies as *Low Risk* with high confidence |
| `test_predict_high_risk` | Machine Learning | `ml.predictor` | Asserts critical markers classify as *High Risk* with missing param handling |
| `test_01_register_user` | Auth / Security | `routes.auth` | Verifies user creation and issuance of JWT token |
| `test_02_login_and_get_profile` | Auth / Security | `routes.auth` | Tests credential authentication and `/profile` data extraction |
| `test_03_update_profile_name` | Profile Mgmt | `routes.profile` | Tests safe display name modification via `PUT /profile` |
| `test_04_change_password` | Auth / Security | `routes.profile` | Verifies old password check, hashing update, and new credential login |
| `test_05_forgot_and_reset` | Auth / Security | `routes.auth` | Full cycle of forgot password, HMAC verification, and single-use revocation |
| `test_01_upload_valid_pdf` | Upload Pipeline | `routes.upload` | Validates multi-part PDF ingest, filesystem isolation, and Report row creation |
| `test_02_upload_rejects_non_pdf`| Upload Pipeline | `routes.upload` | Asserts `.exe` / non-PDF payloads return HTTP 400 |
| `test_03_upload_rejects_empty` | Upload Pipeline | `routes.upload` | Asserts 0-byte files return HTTP 400 |
| `test_04_analyze_uploaded_report`| Integration | `routes.analysis` | End-to-end pipeline: PDF $\rightarrow$ text $\rightarrow$ ML inference $\rightarrow$ AI analysis |
| `test_05_view_analyzed_history` | Persistence | `routes.history` | Verifies reports and ML risk tags are retrievable from user timeline |

---

## 10. Clinical Limitations & Ethical Considerations

1. **Non-Diagnostic Decision Support**: The framework is explicitly engineered as an educational aid and triage mechanism. It does not generate formal medical diagnoses or prescriptions.
2. **Synthetic Data Constraints**: While calibrated against WHO/ADA literature, the Random Forest model was trained on parameterized synthetic distributions. Clinical translation requires IRB-approved retraining on validated multi-center cohorts (e.g., MIMIC-IV).
3. **Legal Disclaimer Guardrails**: The system embeds conspicuous clinical disclaimers across all UI dashboards, exported PDF reports, and API payloads to prevent misplaced clinical reliance.

---

## 11. Technology Stack Summary Table

| Component | Framework / Library | Specification / Version |
|---|---|---|
| **Backend Framework** | FastAPI | Python 3.11 / 3.12, ASGI architecture |
| **Server Engine** | Uvicorn | ASGI Server with worker concurrency |
| **Machine Learning** | Scikit-Learn | Random Forest, SimpleImputer, 5-Fold Stratified CV |
| **Scientific Computing** | NumPy / SciPy | Feature arrays, distribution modeling |
| **Artifact Serialization**| Joblib | Model pipeline persistence |
| **AI Language Model** | Google Gemini 2.5 Flash | Structured JSON output via `google-generativeai` |
| **PDF Extraction** | PyMuPDF (Fitz) & pdfplumber | Hybrid vector stream & layout extraction |
| **OCR Fallback Engine** | PyTesseract (Tesseract OCR) | Optical Character Recognition for scanned assets |
| **Database ORM** | SQLAlchemy 2.0 | Declarative ORM with SQLite backend |
| **Security & Auth** | Passlib, Bcrypt, Python-Jose | Argon2/Bcrypt hashing, HMAC-SHA256, JWT (HS256) |
| **Email Subsystem** | aiosmtplib | Asynchronous SMTP delivery with console fallback |
| **Frontend UI** | HTML5, CSS3, JavaScript | Responsive Glassmorphism, Bootstrap 5.3, Chart.js |
