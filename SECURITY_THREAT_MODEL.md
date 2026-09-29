# VideoMind-AI: Security Threat Model & Hardening Specification

## 1. Threat Modeling Methodology (STRIDE)
VideoMind-AI enforces zero-trust data ingestion, processing, and multi-tenant resource access evaluated against the STRIDE threat categorization.

| STRIDE Threat | Potential Attack Vector | Applied Platform Defense | Defense Layer |
|---|---|---|---|
| **Spoofing** | Forged JWT tokens, unauthorized student impersonating admin | HMAC-SHA256 signature verification, strict role checks (`require_role`), token expiry enforcement | `app/security/auth.py` |
| **Tampering** | Modifying another tenant's video/job records or knowledge base | Resource ownership checks (`verify_resource_ownership`), database user_id scoping | `app/api/main.py`, `app/database` |
| **Repudiation** | Denying an account deletion or administrative quota override | Structured audit logging, versioned knowledge commits with author signatures | `app/knowledge/versioning.py`, `app/security/governance.py` |
| **Information Disclosure** | Data egress of learner PII, leaking cloud API keys in prompts | Automated regex PII redaction (`PIIRedactionEngine`), `LOCAL_ONLY` privacy mode | `app/privacy/engine.py`, `app/security/policy.py` |
| **Denial of Service** | Submitting bomb archive payloads, hammering costly LLM routes | Token-bucket rate limiting (`InMemoryRateLimiter`), quota limits (`QuotaManager`), resource guard | `app/admin/quotas.py`, `app/pipeline/orchestrator.py` |
| **Elevation of Privilege** | Agent executing arbitrary system commands or subshell commands | Tool sandbox permission gating (`ToolExecutionSandbox`) with hard block on `exec`/`eval`/`subprocess` | `app/security/tools.py` |

---

## 2. In-Depth Defense Mechanisms

### 2.1 SSRF Security Layer (`app/security/sanitizer.py`)
- Resolves DNS targets before socket binding.
- Rejects loopback addresses (`127.0.0.1`, `::1`), link-local IPs (`169.254.0.0/16`), private RFC 1918 subnets (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`), and AWS/GCP cloud metadata endpoints (`169.254.169.254`).
- Strictly enforces HTTPS/HTTP schemes.

### 2.2 Prompt Injection Fencing (`app/security/prompt_guard.py`)
- Spoken transcripts, OCR text, and user question inputs are categorized as *untrusted external input*.
- Content is wrapped within XML/Markdown boundary fences (`<<<UNTRUSTED_CONTENT>>> ... <<</UNTRUSTED_CONTENT>>>`).
- Known jailbreak directives (*"ignore all previous instructions"*, *"system override"*) are scrubbed or sanitized prior to model context assembly.

### 2.3 Safe Tool Execution Sandbox (`app/security/tools.py`)
- Agents cannot directly invoke arbitrary Python callables or OS utilities.
- Every tool call requires explicit capability registration with defined parameter types.
- System calls (`os.system`, `subprocess`, `shutil.rmtree`, `open`, `eval`) are blocked at registration time.

### 2.4 Enterprise Data Governance & GDPR Compliance (`app/security/governance.py`)
- **Right-to-Portability**: Exports all video records, jobs, study cards, and personal memories as clean JSON packages.
- **Right-to-Erasure**: Cascading deletion that removes database records, storage files, vector indexes, and personal learning memories cleanly across all subsystems.
