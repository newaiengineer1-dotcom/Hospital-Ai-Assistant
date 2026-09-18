# Al-Shifa Hospital – Synthetic Document Dataset (for RAG Development)

## ⚠️ Important Note
These are **synthetic, fictional documents** created for building and testing a
Retrieval-Augmented Generation (RAG) application. They do **not** contain any
real patient data, real hospital records, or real institutional policy. Use
this as a structural template — replace the PDFs with your actual (properly
authorized and de-identified) hospital documents when moving to production.

## Folder Structure
```
AlShifa_RAG_Dataset/
├── Admissions/
│   ├── ADM-001_Patient_Admission_Policy.pdf
│   ├── ADM-002_Discharge_Procedure_SOP.pdf
│   └── ADM-003_Insurance_Verification_Guidelines.pdf
├── Departments/
│   ├── DEPT-001_Directory_and_Services_Overview.pdf
│   └── DEPT-002_Interdepartmental_Referral_Protocol.pdf
├── Emergency/
│   ├── ER-001_Triage_Protocol.pdf
│   ├── ER-002_Mass_Casualty_Incident_Plan.pdf
│   └── ER-003_Trauma_Activation_Criteria.pdf
└── Patient_Safety/
    ├── PS-001_Incident_Reporting_Policy.pdf
    ├── PS-002_Medication_Safety_and_Five_Rights.pdf
    ├── PS-003_Fall_Prevention_Program.pdf
    └── PS-004_Hand_Hygiene_and_Infection_Control.pdf
```

Each PDF includes structured metadata (document code, department, version,
effective date) plus multiple sections and, where relevant, a reference
table — good test cases for chunking strategies that need to handle headers,
tables, and metadata separately from body text.

## Suggested RAG Pipeline

1. **Ingestion**: Walk the folder tree; tag each chunk's metadata with
   `department` (from the parent folder name) and `doc_code` (parsed from the
   filename prefix, e.g. `ADM-001`).
2. **Parsing**: Use `pypdf` or `pdfplumber` to extract text; extract tables
   separately with `pdfplumber.page.extract_tables()` since they carry
   distinct semantic value (e.g., timelines, safeguard matrices).
3. **Chunking**: Recommend ~300–500 token chunks with section-heading-aware
   splitting (split on the bold "N. Section Name" headers) rather than pure
   fixed-length chunking, to preserve policy coherence.
4. **Embedding**: Store `department`, `doc_code`, `title`, and `version` as
   metadata alongside each vector so you can filter retrieval by department
   (e.g., only search "Emergency" docs for ER-related queries).
5. **Retrieval**: Use hybrid search (BM25 + dense vectors) since these are
   policy documents with specific defined terms (e.g., "ESI Category 2",
   "Morse Fall Scale") that benefit from exact-term matching.
6. **Generation**: Prompt the LLM to cite `doc_code` and section number when
   answering, so responses are traceable back to a specific policy clause.

## Next Steps
- Swap in real, properly de-identified/authorized hospital documents.
- Add access-control metadata if some departments' documents should not be
  retrievable by all users (e.g., role-based filtering in your vector store).
- Consider versioning: keep older policy versions in an `archive/` subfolder
  per department so the RAG system can distinguish "current" vs "historical"
  policy when asked.
