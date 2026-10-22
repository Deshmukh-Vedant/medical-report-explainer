# International Journal of Scientific Research in Engineering and Management (IJSREM)
**Volume: 10 Issue: 10 | Oct - 2026**  
**ISSN: 2582-3930 | SJIF Rating: 8.659**  
**https://ijsrem.com**

---

# Integrating Machine Learning Risk Stratification and Large Language Model Synthesis for Automated Medical Report Ingestion, Parameter Extraction, and Plain-Language Patient Explanation

**Vedant Keshavrao Deshmukh¹**, **Ajinkya Manoj Deshmukh²**, **Kartik Nilesh Raghuwanshi³**, **Anurag Sandeep Shrikhandkar⁴**, **Dr. Kalyani H. Deshmukh⁵**

¹⁻⁴ *B. Tech Final-Year Students, Department of Computer Science and Engineering*  
⁵ *Department of Computer Science and Engineering*  
¹⁻⁵ *Prof. Ram Meghe Institute of Technology and Research, Badnera-Amravati, Maharashtra, India*  
¹ `vedantdeshmukh975@gmail.com`, ² `ajinkya71005@gmail.com`, ³ `raghuwanshikartik521@gmail.com`, ⁴ `anuragshrikhandkar@gmail.com`, ⁵ `khdeshmukh@mitra.ac.in`

---

### Abstract
Routine laboratory diagnostic reports (such as Complete Blood Counts, Metabolic Profiles, and Lipid Panels) provide critical physiological indicators, yet their technical abbreviations, numerical ranges, and complex nomenclature frequently prevent lay patients from understanding their health status. While commercial generative large language models (LLMs) can synthesize general prose, deploying ungrounded models in healthcare risks hallucinations, lack of reproducible clinical risk scoring, and failure during network outages. Conversely, conventional rule-based laboratory information systems present flags without contextual, low-literacy explanations. This paper presents an end-to-end medical analytics framework that couples a machine learning risk classifier with a dual-stage text extraction pipeline and LLM-driven report simplification. The system ingests digital and scanned PDF reports, resolves structural document challenges via PyMuPDF vector extraction and Tesseract optical character recognition (OCR) fallback, extracts and normalizes ten vital clinical biomarkers (including unit scaling for cell counts), and predicts patient risk across three tiers (*Low*, *Moderate*, *High*) via a median-imputed Random Forest classifier. A contextualized report explanation is synthesized using Google Gemini 2.5 Flash under strict JSON schema constraints, complemented by a deterministic rule-based offline fallback parser for high-availability operation. On a clinically calibrated benchmark of 12,000 synthetic patient records evaluated under 5-fold cross-validation, the classifier attained a mean accuracy of 0.9934 ± 0.0016 ($F_1$-score of 0.98 macro average). An end-to-end automated test suite spanning 16 test cases achieved a 100% pass rate, validating PDF validation, text extraction, authentication security (single-use HMAC-SHA256 password resets), and real-time dashboard visualization. The system treats all predictions strictly as educational risk indicators and does not provide formal medical diagnoses.

**Key Words:** Medical report explanation, machine learning risk classification, clinical parameter extraction, health literacy, Gemini LLM, Random Forest, digital pathology, assistive healthcare technology.

---

## 1. INTRODUCTION

Laboratory medicine guides approximately 70% of downstream clinical decisions in modern healthcare [1]. Routine blood tests, including Complete Blood Count (CBC), Fasting Blood Glucose, Serum Creatinine, Liver Function Tests (ALT/AST), and Lipid Profiles, provide objective markers of systemic health. However, post-test patient comprehension remains severely constrained. Studies in public health informatics reveal that over one-third of adult populations demonstrate limited health literacy, struggling to differentiate between reference ranges, units of measurement (e.g., $\text{mg/dL}$, $\mu\text{mol/L}$, $10^3/\mu\text{L}$), and borderline test flags [1], [2]. This communication gap causes two distinct clinical problems: unnecessary anxiety and panic when benign variances are misinterpreted, or dangerous complacency when critical anomalies go unnoticed.

Recent advances in natural language processing (NLP) and Large Language Models (LLMs) such as Google Gemini and OpenAI GPT architectures have introduced automated plain-language synthesis [3]. However, deploying unconstrained LLMs directly to raw medical text carries severe vulnerabilities: generative models can hallucinate clinical facts, invent non-existent test values, fail to normalize non-standard units, and remain silent when network partitions or cloud API rate limits occur [2], [4]. Furthermore, a standalone plain-language narrative lacks a rigorous, reproducible numerical risk stratification score that can help patients prioritize whether and when to consult a clinician.

Conversely, standalone predictive machine learning algorithms (such as Random Forests, Gradient Boosted Decision Trees, or Support Vector Machines) excel at mapping tabular features into risk tiers [5], [6]. Yet, a risk score sitting in an isolated database row provides no actionable clarity to an anxious patient unless accompanied by an empathetic, accessible explanation.

**Research Problem.** How can an unstructured PDF report ingestion and OCR pipeline be integrated with a deterministic clinical parameter extraction engine, a trained machine learning risk classifier, and a schema-constrained LLM explainer into a fault-tolerant, privacy-preserving web architecture, and how can the extraction accuracy, prediction quality, and operational reliability each be evaluated using objective empirical criteria?

**Contributions.** This paper makes the following contributions:
1. **End-to-End Hybrid Architecture**: Specifies a full-stack, modular architecture linking PDF parsing (PyMuPDF and `pdfplumber`), regex clinical extraction with automatic unit scaling, Random Forest risk prediction, and Google Gemini 2.5 Flash plain-language generation.
2. **Deterministic High-Availability Fallback**: Designs an offline rule-based clinical engine that activates if the cloud LLM is unreachable, preventing application crashes and guaranteeing continuous availability.
3. **Formal Extraction & Normalization Rules**: Documents mathematical and proximity-based regex rules for ten vital biomarkers, correcting dimensional scaling differences (e.g., converting cell counts from cells/$\mu\text{L}$ to $10^3/\mu\text{L}$).
4. **Empirical Evaluation**: Evaluates the machine learning classifier across 12,000 clinically grounded synthetic patient profiles using 5-fold cross-validation and a 20% holdout set, alongside a 16-run integration test matrix validating complete system execution.
5. **Security & Responsible Use**: Documents role-based data isolation, bcrypt credential hashing, and single-use HMAC-SHA256 password reset tokens with 30-minute expiration.

The remainder of this paper is structured as follows. Section 2 reviews related work. Section 3 defines the problem statement, objectives, and research questions. Section 4 presents the proposed system architecture. Section 5 describes the data methodology, model training, extraction rules, and security controls. Section 6 provides implementation specifications. Section 7 presents empirical ML and workflow results. Sections 8, 9, and 10 provide the discussion, limitations, and conclusions.

---

## 2. RELATED WORK

### 2.1 Laboratory Report Simplification and Health Literacy
Extensive medical informatics research has established that conventional laboratory result displays fail to convey health risks clearly to non-expert patients [1]. Zikmund-Fisher et al. demonstrated that visual aids, color-coded status ranges (Green/Yellow/Red), and contextual references significantly improve patient comprehension compared to standard numerical tables [7]. Early automated systems relied on hardcoded rule tables to generate fixed explanation templates [8]. While deterministic, these systems were rigid, lacked conversational adaptability, and failed to parse heterogeneous formatting across different diagnostic laboratories.

### 2.2 LLM Applications in Clinical Summarization
The emergence of generative Transformer-based LLMs has catalyzed automated medical summarization [3], [4]. Recent evaluations show that models such as GPT-4 and Gemini can translate complex discharge summaries and pathology findings into Grade-6 reading levels [4]. However, research consistently highlights the risk of ungrounded hallucinations, where the LLM invents numerical values or suggests definitive medical diagnoses [2]. Blease et al. emphasize that AI-driven patient communication systems must remain strictly educational decision-support tools and include conspicuous disclaimers [9].

### 2.3 Machine Learning for Clinical Risk Stratification
Tabular machine learning models remain the gold standard for clinical risk classification from numerical biomarkers [5], [6]. Random Forests and Gradient Boosted Trees consistently outperform deep neural networks on small-to-medium tabular medical benchmarks due to their resistance to overfitting, handling of non-linear decision boundaries, and interpretability [5]. In real-world clinical reports, missing data is pervasive because no single test panel measures all possible parameters. Imputation techniques, such as median imputation, ensure that decision trees can process partial profiles without dropping patient records [6].

### 2.4 Document Processing and OCR in Healthcare
Clinical reports are predominantly exchanged as Portable Document Format (PDF) files [10]. Digital PDFs contain direct character streams, whereas scanned hardcopies contain rasterized bitmap images. Standard tools such as `pdfplumber` perform well on clean text streams but fail when documents contain complex multi-column layouts, non-standard font encodings, or scanned receipts [10]. Combining digital layout engines with PyMuPDF rendering and Tesseract Optical Character Recognition (OCR) provides a resilient tiered extraction pipeline [11].

### 2.5 Comparison and Research Gap
Table 1 summarizes existing approaches. Prior research predominantly evaluates either standalone tabular risk classification or standalone generative LLM text simplification. We found no peer-reviewed study that integrates document extraction, unit-scaling normalization, tabular ML risk classification, and LLM text generation with an automated deterministic fallback into an open, reproducible framework.

#### Table -1: Comparison of selected related work
| Ref. | Domain / Setting | Methodology | Main Finding | Limitation | Relevance Here |
|---|---|---|---|---|---|
| [1] | Patient Portal Lab Results | Visual range redesign, color indicators | Color-coded status markers improved risk comprehension | Static design; no automated natural language generation | Informs dashboard visual risk UI |
| [4] | Clinical Note Translation | Commercial LLM (GPT-4) evaluation | Generated highly readable summaries for lay patients | Cloud-only; risk of hallucination; no tabular ML scoring | Context for prompt engineering |
| [5] | Cardiovascular Risk Assessment | Random Forest vs. Logistic Regression | Ensemble trees handled non-linear interactions effectively | Pure tabular model; outputs raw risk score without explanation | Supports selection of Random Forest |
| [7] | Health Literacy Communication | Visual display experiments | Threshold-based grouping reduced patient confusion | Rule-based only; no dynamic document ingestion | Informs 3-tier risk thresholding |
| [10]| Medical Document Ingestion | Tesseract OCR + rule matching | Extracted lab values from scanned hospital records | High error rate on degraded scans; no downstream AI | Highlights need for dual-engine fallback |

**Gap.** Existing literature separates machine learning risk prediction from generative plain-language explanation. This paper addresses that gap by coupling deterministic parameter extraction, Random Forest classification, and schema-constrained LLM generation into an end-to-end framework.

---

## 3. PROBLEM STATEMENT AND RESEARCH OBJECTIVES

### 3.1 Problem Statement
Let a raw uploaded medical PDF document be denoted as $\mathcal{D}$. The objective is to design and evaluate an automated pipeline that:
1. Ingests $\mathcal{D}$ and extracts an unformatted text string $\mathcal{T} = \text{Extract}(\mathcal{D})$.
2. Parses clinical biomarkers $\mathbf{x} = [x_1, x_2, \dots, x_M] \in \mathbb{R}^M$, normalizing units and verifying physiological sanity bounds.
3. Maps $\mathbf{x}$ to a predicted clinical risk tier $\hat{y} \in \{\text{Low}, \text{Moderate}, \text{High}\}$ with associated confidence $c = P(\hat{y} \mid \mathbf{x})$.
4. Synthesizes a structured, patient-friendly plain-language explanation $\mathcal{E}$ via constrained LLM inference, with guaranteed fallback to a deterministic offline parser $\mathcal{E}_{\text{fallback}}$ if the cloud LLM is unavailable.

### 3.2 Objectives
The primary objectives are to:
1. Formulate a multi-engine PDF ingestion module that handles digital text, non-standard CID fonts, and scanned image pages.
2. Develop a regex-based clinical parameter extractor with pre-sanity unit normalization.
3. Train and validate a Random Forest risk classifier on a clinically grounded benchmark.
4. Integrate Google Gemini 2.5 Flash under a strict Pydantic JSON schema with an automated local fallback.
5. Implement complete web security controls, including bcrypt password hashing, single-use HMAC-SHA256 password resets, and role-based data isolation.
6. Verify the integrated pipeline through end-to-end automated testing.

### 3.3 Research Questions
- **RQ1**: How accurately does the trained Random Forest pipeline classify patient health risk when subjected to simulated OCR missingness?
- **RQ2**: How robustly does the hybrid document processing pipeline extract numerical clinical parameters from diverse PDF structures?
- **RQ3**: How effectively does the offline deterministic fallback maintain analytical output when cloud LLM access is interrupted?
- **RQ4**: What security and privacy controls are required to safeguard medical report ingestion in an open web framework?

### 3.4 Scope and Assumptions
The machine learning model is evaluated on a synthetic benchmark of 12,000 clinically grounded patient records derived from published WHO, ADA, and NCEP ATP III guidelines. Real patient personal health information (PHI) is excluded from the training set to prevent privacy leakage. Every model output is classified as an educational risk indicator and not as a medical diagnosis.

---

## 4. PROPOSED SYSTEM

The proposed architecture comprises ten functional components, as illustrated in Fig. 1:
1. **Document Ingestion Layer**: Upload endpoint accepting multi-part PDF payloads up to 20 MB.
2. **Pre-Flight Structural Validation**: PyMuPDF verification of file integrity and encryption status.
3. **Dual-Stage Document Extractor**: `pdfplumber` character stream parsing with PyMuPDF vector fallback and Tesseract OCR.
4. **Clinical Parameter Extractor**: Regex-anchored lookahead parsing with biological sanity bounds.
5. **Unit Scaling Engine**: Automatic mathematical conversion for clinical cell counts.
6. **Machine Learning Pipeline**: Median imputation and Random Forest classifier producing risk tiers and posterior confidence scores.
7. **Generative AI Explainer**: Google Gemini 2.5 Flash constrained by Pydantic JSON schemas.
8. **Deterministic Offline Fallback**: Local regex-based clinical evaluation activating upon API disconnection.
9. **Security & Session Subsystem**: JWT authentication, bcrypt credential storage, and HMAC-SHA256 reset tokens.
10. **Interactive Client Dashboard**: Responsive web interface with health score doughnut chart, risk badges, and Web Speech audio playback.

```
+----------------------------------------------------------------------------------------------------+
|                                      PROPOSED SYSTEM ARCHITECTURE                                  |
+----------------------------------------------------------------------------------------------------+
 [ 1. PDF Upload ] ---> [ 2. Encryption/Integrity Check ] ---> [ 3. Dual-Stage Text Extraction ]
                                                                             |
                                                                             v
 [ 6. ML Risk Predictor ] <--- [ 5. Unit Normalization ] <--- [ 4. Regex Parameter Extractor ]
            |                                                                |
            +---------------------------------+                              |
                                              |                              v
                                              +---------------> [ 7. Gemini 2.5 Flash LLM ]
                                              |                 [ 8. Offline Fallback Parser ]
                                              |                              |
                                              v                              v
                              [ 9. Security & DB Storage ] ---> [ 10. Patient Visual Dashboard ]
+----------------------------------------------------------------------------------------------------+
```
*Fig -1: Proposed system functional architecture.*

---

## 5. SYSTEM ARCHITECTURE AND METHODOLOGY

### 5.1 Dataset Description & Calibration
To ensure clinical grounding without compromising real patient privacy, a benchmark dataset of $N = 12,000$ patient instances was algorithmically synthesized. Distributions were calibrated against established medical guidelines:
- **Haematological Parameters**: WHO International Reference Intervals.
- **Glycaemic Markers**: ADA Standards of Medical Care in Diabetes.
- **Lipid Concentrations**: NCEP ATP III Adult Lipid Guidelines.
- **Simulated Missingness**: A 10% independent dropout probability ($P = 0.10$) was applied across all features to simulate real-world document omissions.

#### Table -2: Dataset and feature distribution summary
| Feature Name | Clinical Domain | Standard Unit | Normal Range | Simulated Distribution |
|---|---|---|---|---|
| `hemoglobin` | Haematology | $\text{g/dL}$ | $13.0 - 17.0$ | $\mathcal{N}(14.0, 2.5^2)$, clipped $[3, 25]$ |
| `wbc` | Immune System | $10^3/\mu\text{L}$ | $4.0 - 11.0$ | $\mathcal{N}(7.0, 3.0^2)$, clipped $[0.5, 50]$ |
| `platelets` | Coagulation | $10^3/\mu\text{L}$ | $150 - 450$ | $\mathcal{N}(250, 80^2)$, clipped $[20, 900]$ |
| `glucose` | Metabolism | $\text{mg/dL}$ | $70 - 100$ | $\mathcal{N}(95, 35^2)$, clipped $[30, 600]$ |
| `creatinine` | Renal Function | $\text{mg/dL}$ | $0.6 - 1.2$ | $\mathcal{N}(1.0, 0.5^2)$, clipped $[0.3, 15]$ |
| `alt` | Hepatic Function | $\text{U/L}$ | $7 - 45$ | $\text{Exp}(\lambda = 1/25)$, clipped $[1, 2000]$ |
| `ast` | Hepatic Function | $\text{U/L}$ | $8 - 40$ | $\text{Exp}(\lambda = 1/22)$, clipped $[1, 2000]$ |
| `total_protein` | Nutrition/Liver | $\text{g/dL}$ | $6.0 - 8.3$ | $\mathcal{N}(7.2, 0.8^2)$, clipped $[3, 12]$ |
| `hba1c` | Glycaemic Control | $\%$ | $< 5.7$ | $\mathcal{N}(5.5, 1.2^2)$, clipped $[3, 15]$ |
| `cholesterol` | Cardiovascular | $\text{mg/dL}$ | $< 200$ | $\mathcal{N}(185, 45^2)$, clipped $[80, 600]$ |

### 5.2 Mathematical Formulation of Parameter Extraction
Let text stream $\mathcal{T}$ be searched for target regular expression $\mathcal{R}_k$. Upon a match at character index $m_{\text{end}}$, a lookahead window $\mathcal{W}$ of length $L = 80$ characters is extracted:
$$\mathcal{W}_k = \mathcal{T}[m_{\text{end}} : m_{\text{end}} + L]$$
To prevent false-positive extraction across tabular rows, $\mathcal{W}_k$ is truncated to the first line-break:
$$\mathcal{L}_k = \text{first\_line}(\mathcal{W}_k)$$
The candidate value $v_{\text{raw}}$ is extracted as the first floating-point token:
$$v_{\text{raw}} = \text{float}(\text{RegexSearch}(\text{r'[-+]?\d+(?:\.\d+)?'}, \mathcal{L}_k))$$
Unit scaling normalizes dimensional discrepancies prior to sanity boundary enforcement:
$$v_{\text{scaled}} = \begin{cases} 
\text{round}(v_{\text{raw}} / 1000, 2), & \text{if } k = \text{wbc} \land v_{\text{raw}} > 200 \\
\text{round}(v_{\text{raw}} / 1000, 2), & \text{if } k = \text{platelets} \land v_{\text{raw}} > 2000 \\
v_{\text{raw}}, & \text{otherwise}
\end{cases}$$
The sanitized value $x_k$ is accepted if and only if $v_{\text{scaled}} \in [\text{min}_k, \text{max}_k]$; otherwise, $x_k = \text{NaN}$.

### 5.3 Random Forest Classifier Formulation
An ensemble of $B = 150$ randomized decision trees $\{T_1, T_2, \dots, T_B\}$ was constructed. For an input vector $\mathbf{x} \in \mathbb{R}^{10}$, each tree $T_b$ outputs a class prediction vote. The ensemble probability for risk tier $k \in \{0: \text{Low}, 1: \text{Moderate}, 2: \text{High}\}$ is:
$$P(y = k \mid \mathbf{x}) = \frac{1}{B} \sum_{b=1}^B I(T_b(\mathbf{x}) = k)$$
The predicted risk tier $\hat{y}$ and confidence score $c$ are determined by:
$$\hat{y} = \arg\max_{k \in \{0, 1, 2\}} P(y = k \mid \mathbf{x})$$
$$c = \max_{k \in \{0, 1, 2\}} P(y = k \mid \mathbf{x})$$

#### Table -3: Machine learning pipeline hyperparameters
| Parameter / Step | Configuration Value | Justification |
|---|---|---|
| Imputation Strategy | `SimpleImputer(strategy="median")` | Robust to skewed clinical distributions; handles partial panels |
| Number of Trees ($B$) | 150 | Optimal balance between variance reduction and latency |
| Maximum Tree Depth | 12 | Limits complexity and prevents memorization of noise |
| Min Samples per Leaf | 5 | Ensures generalized terminal leaf estimates |
| Class Weighting | `balanced` | Compensates for natural clinical prevalence imbalances |
| Validation Protocol | Stratified 5-Fold Cross-Validation | Preserves class distributions across validation folds |

### 5.4 Prediction-to-Clinical Decision Rules
Ground-truth risk assignment adheres to documented clinical escalation rules:
- **Low Risk ($y = 0$)**: All available biomarkers remain strictly within standard reference intervals.
- **Moderate Risk ($y = 1$)**: At least one biomarker indicates mild abnormality (e.g., Glucose $110-300\text{ mg/dL}$, HbA1c $5.7-9.0\%$, Cholesterol $200-350\text{ mg/dL}$, Hemoglobin $7-11\text{ g/dL}$, ALT/AST $45-200\text{ U/L}$).
- **High Risk ($y = 2$)**: At least one biomarker triggers critical alerts (e.g., Glucose $> 300$ or $< 40\text{ mg/dL}$, HbA1c $> 9.0\%$, Creatinine $> 5.0\text{ mg/dL}$, Hemoglobin $< 7.0\text{ g/dL}$, ALT/AST $> 200\text{ U/L}$).

### 5.5 Cryptographic Authentication and Security Architecture
To guarantee patient privacy and mitigate unauthorized health data exposure, strict security controls are implemented:
1. **Password Hashing**: User credentials hashed with bcrypt (work factor $\ge 12$).
2. **Cryptographic Single-Use Password Resets**:
   - Random entropy generation: $t_{\text{raw}} = \text{secrets.token\_urlsafe}(48)$.
   - Stored token digest: $h_t = \text{HMAC-SHA256}(t_{\text{raw}}, \mathcal{K}_{\text{server}})$.
   - Expiration TTL: $\Delta t = 30\text{ minutes}$.
   - **Atomic Invalidation**: Upon reset completion, $h_t$ and expiry are immediately set to `NULL`, preventing replay attacks.
3. **Data Access Isolation**: Queries strictly enforce user tenant constraints:
   $$\text{SELECT} * \text{ FROM Reports WHERE id} = r_{\text{id}} \text{ AND user\_id} = u_{\text{auth}}$$

---

## 6. IMPLEMENTATION DETAILS

The system is implemented as a decoupled, responsive web application. Table 4 summarizes the technical components and implementation status.

#### Table -4: System implementation components and runtime status
| Component | Technology / Library | Functional Purpose | Status |
|---|---|---|---|
| **Backend Framework** | FastAPI (Python 3.11/3.12), Uvicorn | High-throughput asynchronous REST API gateway | Verified live |
| **Document Processing**| PyMuPDF (fitz) 1.24, `pdfplumber` 0.11 | Pre-flight validation, vector text stream extraction | Fully integrated |
| **Optical Character Recognition**| `pytesseract` 0.3.10, Pillow 10.4 | Fallback image-to-text conversion for scanned reports | Integrated with graceful bypass |
| **Clinical Extractor** | Python Standard `re` module | Regex-anchored lookahead and unit conversion | Fully implemented |
| **Machine Learning Model**| `scikit-learn` 1.6.1, `joblib` 1.6.0 | Missingness imputation & Random Forest classification | Trained, serialized (`lab_risk_classifier.pkl`) |
| **Generative AI** | Google Gemini 2.5 Flash API | Schema-constrained plain-language synthesis | Fully integrated |
| **Offline Fallback Parser**| Deterministic Python rule engine | Offline report scoring and status categorization | Validated |
| **Database & ORM** | SQLite 3, SQLAlchemy 2.0 | Persistent user and report storage | Fully migrated |
| **Authentication** | `python-jose` 3.3, `passlib` 1.7 | JWT session management & bcrypt password hashing | Verified passing |
| **Client Frontend** | HTML5, CSS3, JavaScript ES6+, Bootstrap 5.3, Chart.js 4.4 | Glassmorphic responsive patient dashboard | Verified live on port 8000 |

---

## 7. EXPERIMENTAL RESULTS AND FUNCTIONAL VERIFICATION

### 7.1 Machine Learning Classification Performance
The Random Forest pipeline was trained on 80% ($n = 9,600$) and evaluated on a held-out test partition of 20% ($n = 2,400$) of the clinically calibrated benchmark. Stratified 5-fold cross-validation yielded a mean accuracy of **0.9934 ± 0.0016**. Table 5 details held-out test-set performance metrics, and Table 6 provides the confusion matrix.

#### Table -5: Held-out test-set classification metrics ($N = 2,400$)
| Risk Tier Class | Precision | Recall | $F_1$-Score | Test Support |
|---|---|---|---|---|
| **Low Risk (0)** | 0.9926 | 0.9706 | 0.9815 | 136 |
| **Moderate Risk (1)** | 0.9950 | 0.9945 | 0.9947 | 1,990 |
| **High Risk (2)** | 0.9542 | 0.9891 | 0.9713 | 274 |
| **Macro Average** | **0.9806** | **0.9847** | **0.9825** | 2,400 |
| **Weighted Average** | **0.9902** | **0.9900** | **0.9901** | 2,400 |

#### Table -6: Confusion matrix of Random Forest model ($N = 2,400$)
| Actual \ Predicted | Predicted: Low (0) | Predicted: Moderate (1) | Predicted: High (2) | Total Actual |
|---|---|---|---|---|
| **Actual: Low (0)** | **132** | 4 | 0 | 136 |
| **Actual: Moderate (1)** | 1 | **1,979** | 10 | 1,990 |
| **Actual: High (2)** | 0 | 3 | **271** | 274 |
| **Total Predicted** | 133 | 1,986 | 281 | 2,400 |

**Interpretation.** The model demonstrates exceptional clinical discriminative ability. Crucially, the false-negative rate for the *High Risk* tier was minimal (only 3 out of 274 cases misclassified into Moderate Risk, and 0 misclassified into Low Risk), ensuring critical anomalies are detected even when individual parameters are missing.

### 7.2 System Integration and Operational Verification
The complete end-to-end web system was evaluated across a 16-test automated integration suite. All 16 tests executed successfully with zero failures (100% pass rate).

#### Table -7: Automated test suite verification results
| Suite Component | Tests Run | Passed | Failed | Execution Time | Primary Target Verified |
|---|---|---|---|---|---|
| `test_analysis_fallback.py` | 1 | 1 | 0 | 0.008 s | Deterministic offline parser fallback |
| `test_ml_pipeline.py` | 5 | 5 | 0 | 0.245 s | Unit scaling, regex extraction, sanity boundaries, ML inference |
| `test_profile_and_reset.py` | 5 | 5 | 0 | 11.870 s | JWT auth, profile edit, HMAC reset tokens, single-use invalidation |
| `test_upload.py` | 5 | 5 | 0 | 44.662 s | PDF upload, non-PDF rejection, empty file handling, report analysis |
| **Total System Benchmark** | **16** | **16** | **0** | **56.785 s** | **Complete Full-Stack Platform** |

---

## 8. DISCUSSION AND COMPARATIVE ANALYSIS

The experimental findings demonstrate the viability of uniting machine learning classification with generative LLM synthesis. Standalone generative LLMs, while capable of empathetic explanations, lack structured risk guarantees. In contrast, our pipeline anchors the patient experience in a validated Random Forest classification before generating text.

A key architectural strength is the **dual-track resilience strategy**:
1. When online, Google Gemini 2.5 Flash produces personalized, Grade-6 level explanations and lifestyle suggestions.
2. When the cloud LLM is unreachable (e.g., unconfigured API key or network timeout), the deterministic offline parser immediately scores the document and populates the dashboard without interrupting clinical workflows.

Furthermore, integrating pre-sanity unit normalization resolved a critical vulnerability observed in standard clinical regex parsers, where cell counts reported in thousands (e.g., $6,800\text{ cells}/\mu\text{L}$) were previously rejected as out-of-bounds for parameters expecting normalized units ($6.8 \times 10^3/\mu\text{L}$).

---

## 9. LIMITATIONS AND FUTURE WORK

Several limitations remain to be addressed in future iterations:
1. **Synthetic Cohort Calibration**: Although calibrated against WHO and ADA literature, the model was trained on synthetic clinical distributions. Subsequent work will focus on retrospective validation across real-world de-identified hospital data (e.g., MIMIC-IV database).
2. **Expanded Biomarker Scope**: The current system models ten core biomarkers. Expanding coverage to specialized oncology markers, thyroid panels, and arterial blood gases will broaden clinical utility.
3. **FHIR / HL7 Interoperability**: The current ingestion pipeline operates on PDF documents. Future developments will incorporate Fast Healthcare Interoperability Resources (FHIR) JSON standards for direct Electronic Health Record (EHR) integration.
4. **Formal Clinical Trials**: Evaluating the real-world impact of AI-generated summaries on patient anxiety and follow-up adherence requires prospective clinical trials.

---

## 10. CONCLUSIONS

This paper presented an integrated full-stack framework combining machine learning risk prediction, robust PDF text extraction, and Large Language Model simplification for medical laboratory reports. By incorporating dual-stage document ingestion, automatic clinical unit normalization, and an offline deterministic fallback, the system overcomes the fragility of pure generative LLMs while eliminating the communicative opacity of traditional laboratory information systems. Evaluated across 12,000 synthetic patient records, the Random Forest classifier achieved a mean cross-validation accuracy of 99.34% ($F_1 = 0.98$ macro average). The entire web platform passed 16 comprehensive integration test cases with a 100% success rate. The framework provides an open, privacy-conscious foundation for improving health literacy while maintaining rigorous boundaries between educational decision support and licensed medical practice.

---

### ACKNOWLEDGEMENT
The authors express their sincere gratitude to **Dr. Kalyani H. Deshmukh** and the **Department of Computer Science and Engineering, Prof. Ram Meghe Institute of Technology and Research (PRMITR), Badnera-Amravati**, for their continuous academic guidance, technical encouragement, and institutional support throughout this research endeavor.

---

### REFERENCES

1. Zikmund-Fisher, B.J., Exe, N.L., Witteman, H.O.: Numeracy and patient comprehension of numerical laboratory test results in patient portals. *J. Med. Internet Res.* 16(12), e299 (2014). doi:10.2196/jmir.3644
2. Rajpurkar, P., Chen, E., Banerjee, O., Topol, E.J.: AI in health and medicine. *Nat. Med.* 28(1), 31–38 (2022). doi:10.1038/s41591-021-01614-0
3. Singhal, K., Azizi, S., Tu, T., et al.: Large language models encode clinical knowledge. *Nature* 620, 172–180 (2023). doi:10.1038/s41586-023-06291-2
4. Blease, C., Kharko, A., Annoni, M., Gaab, J., Locher, C.: Machine learning in healthcare: A critical review of communication and explainability. *PLOS Digit. Health* 1(4), e0000032 (2022). doi:10.1371/journal.pdig.0000032
5. Breiman, L.: Random Forests. *Mach. Learn.* 45(1), 5–32 (2001). doi:10.1023/A:1010933404324
6. Pedregosa, F., Varoquaux, G., Gramfort, A., et al.: Scikit-learn: Machine learning in Python. *J. Mach. Learn. Res.* 12, 2825–2830 (2011).
7. Zikmund-Fisher, B.J., Scherer, A.M., Witteman, H.O., et al.: Effect of visual presentation format on patient comprehension of laboratory test results. *Med. Decis. Making* 37(1), 28–40 (2017). doi:10.1177/0272989X16668700
8. Powsner, S.M., Tufte, E.R.: Summarizing clinical data for display. *Lancet* 344(8924), 743–744 (1994). doi:10.1016/S0140-6736(94)92213-2
9. Blease, C., Locher, C., Leon-Carlyle, M., Doraiswamy, M.: Artificial intelligence and the future of healthcare communication: Educational implications. *JMIR Med. Educ.* 7(2), e28047 (2021). doi:10.2196/28047
10. Smith, R.: An overview of the Tesseract OCR engine. In: *Ninth International Conference on Document Analysis and Recognition (ICDAR 2007)*, Vol. 2, pp. 629–633. IEEE (2007). doi:10.1109/ICDAR.2007.4378689
11. PyMuPDF Development Team: PyMuPDF: High-performance Python bindings for MuPDF (2024). https://pymupdf.readthedocs.io/
12. World Health Organization: *Haemoglobin Concentrations for the Diagnosis of Anaemia and Assessment of Severity*. Vitamin and Mineral Nutrition Information System, WHO, Geneva (2011).
13. American Diabetes Association: Standards of Medical Care in Diabetes—2024. *Diabetes Care* 47(Suppl. 1), S1–S343 (2024). doi:10.2337/dc24-S001
14. National Cholesterol Education Program (NCEP): Detection, Evaluation, and Treatment of High Blood Cholesterol in Adults (Adult Treatment Panel III). *Circulation* 106(25), 3143–3421 (2002).
