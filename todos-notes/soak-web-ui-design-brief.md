# Soak Web UI -- Design Brief & Implementation Plan

## Overview

A Django-based web application for running soak qualitative analysis pipelines. Users can manage projects, upload documents, configure and run pipelines, track costs, and compare analysis results.

**Key Principles:**
- Progressive disclosure -- powerful features accessible but not overwhelming
- HTMX for reactive UI without heavy JavaScript
- Celery for async pipeline execution
- BYOK (Bring Your Own Key) model with optional credit purchase
- Bootstrap 5 for styling (consistent with struckdown UI)

---

## Version Scope

### Version 1 (MVP)
- **Pipelines**: Pre-built system pipelines only; users configure context variables
- **Classification**: Separate pipeline type where users can edit the full struckdown template
- **Users**: Individual accounts only (no teams/organisations)
- **Storage**: Text content stored in database; original files discarded after text extraction
- **Billing**: BYOK + optional credit purchase with 5% markup

### Version 2 (Future)
- Guided visual pipeline builder
- Classification results create document sets for further analysis
- Team/organisation features
- More advanced pipeline customisation

---

## 1. User Authentication & Billing Model

### Authentication
- Django's built-in auth with email-based login
- Optional social auth (Google, GitHub) via django-allauth
- API key storage encrypted in database (using django-cryptography or similar)

### Billing Tiers

| Tier | Model | Description |
|------|-------|-------------|
| **Free/BYOK** | User provides own API key | No platform charges, user pays LLM provider directly |
| **Credits** | Pre-purchased credits | 5% markup on LLM costs, platform handles API calls |
| **Enterprise** | Custom | Dedicated resources, SLA, volume discounts |

### Credit System
- Users purchase credits in advance (Stripe integration)
- Credits debited in real-time during pipeline execution
- Cost tracking: prompt_tokens + completion_tokens priced per model
- Transparent cost display before and after runs
- Low balance warnings, auto-pause on zero credits

---

## 2. Data Model (Django ORM)

```
Organisation (optional, for teams)
├── id, name, created_at
├── billing_tier, credit_balance
└── members (M2M to User with role)

User
├── id, email, password_hash
├── api_key_encrypted (for BYOK)
├── api_base_url (custom endpoint support)
├── organisation_id (FK, optional)
├── credit_balance (for individual accounts)
└── preferences (JSON)

Project
├── id, name, description
├── user_id (FK) / organisation_id (FK)
├── created_at, updated_at
├── is_archived
└── settings (JSON -- default models, etc.)

Document
├── id, project_id (FK)
├── filename (original name for display)
├── content (TextField -- extracted text, original file discarded)
├── metadata (JSON -- extracted from spreadsheets, headers, etc.)
├── uploaded_at
├── content_hash (for deduplication)
└── word_count, char_count

DocumentSet (subsets of documents for analysis)
├── id, project_id (FK)
├── name, description
├── created_at
└── documents (M2M to Document)

PipelineTemplate
├── id, name, description
├── yaml_content (TextField -- full .soak file)
├── is_system (bool -- built-in vs user-created)
├── user_id (FK, null for system pipelines)
├── created_at, updated_at
├── default_context_schema (JSON -- describes expected variables)
└── category (e.g., "thematic", "classification", "coverage")

Run
├── id, project_id (FK)
├── pipeline_template_id (FK)
├── document_set_id (FK, optional -- null = all project docs)
├── user_id (FK)
├── name, description
├── status (pending, running, completed, failed, cancelled)
├── context_overrides (JSON -- user-provided context vars)
├── config_overrides (JSON -- model, temperature, etc.)
├── started_at, completed_at
├── celery_task_id
├── error_message (TextField, null)
├── export_folder_path
└── result_summary (JSON -- themes count, codes count, etc.)

RunCost
├── id, run_id (FK)
├── total_cost, fresh_cost
├── prompt_tokens, completion_tokens
├── fresh_count, cached_count
├── by_node (JSON -- per-node breakdown)
├── billing_tier_at_time
└── credit_deducted

RunNode (for detailed tracking)
├── id, run_id (FK)
├── node_name, node_type
├── execution_order
├── status (pending, running, completed, failed, skipped)
├── started_at, completed_at
├── output_preview (JSON -- truncated result)
├── cost, prompt_tokens, completion_tokens
└── error_message

QualitativeResult (extracted analysis)
├── id, run_id (FK, OneToOne)
├── themes (JSON array)
├── codes (JSON array)
├── narrative (TextField)
├── analysis_name
└── details (JSON -- full result dump)

Comparison
├── id, project_id (FK)
├── name, description
├── runs (M2M to Run)
├── created_at
├── config (JSON -- threshold, method, etc.)
├── result_html (TextField -- rendered comparison)
└── status (pending, completed, failed)
```

---

## 3. Key User Flows

### 3.1 Project Setup Flow
```
1. Create Project → Name, description
2. Upload Documents → Drag-drop zone, ZIP support, progress bar
3. Review Documents → Table view with metadata, word counts
4. Create Document Sets → Optional subsets for different analyses
```

### 3.2 Pipeline Selection & Configuration Flow
```
1. Select Pipeline → Gallery of available pipelines (system + custom)
2. Preview Pipeline → DAG visualization, description, required variables
3. Configure Context Variables → Form generated from pipeline schema
   - Text inputs for prompts (research_question, persona, etc.)
   - Dropdowns for model selection
   - Sliders for numeric params (temperature, chunk_size)
4. Select Documents → Choose document set or specific files
5. Cost Estimate → Approximate cost based on document size + model
6. Launch Run → Async execution via Celery
```

### 3.3 Run Monitoring Flow
```
1. Run Dashboard → List of runs with status badges
2. Live Progress → HTMX polling for node-by-node progress
   - Show current node, completion percentage
   - Real-time cost accumulation
   - Stream node outputs as they complete
3. Completion → Notification (browser, optional email)
4. View Results → Embedded HTML report (pipeline.html or simple.html)
```

### 3.4 Comparison Flow
```
1. Select Runs → Multi-select from completed runs
2. Configure Comparison → Threshold, embedding settings
3. Generate → Async comparison via Celery
4. View → Embedded comparison.html with all visualizations
```

### 3.5 Classification Pipeline Flow (Special Case)
```
1. Select "Classification" pipeline type
2. Edit Classification Template → Full struckdown editor with syntax highlighting
   - Pre-populated with sensible default template
   - Users define classification criteria, labels, output format
   - Syntax validation before saving
3. Select Documents → Choose document set
4. Run Classification → Async execution
5. View Results → Classification results table with per-document labels
   (Version 2: Results can create new document sets for filtered analysis)
```

---

## 4. UI Screens (Wireframes)

### 4.1 Dashboard (Home)
```
┌─────────────────────────────────────────────────────────────────┐
│ [Logo] Soak                    [Credits: $12.45] [User ▼]      │
├───────┬─────────────────────────────────────────────────────────┤
│       │                                                         │
│ Nav   │  Recent Projects                    [+ New Project]     │
│       │  ┌──────────┐ ┌──────────┐ ┌──────────┐                │
│ Dash  │  │ CFS Study│ │ Interview│ │ Survey   │                │
│ Proj  │  │ 45 docs  │ │ 12 docs  │ │ 200 rows │                │
│ Runs  │  │ 8 runs   │ │ 3 runs   │ │ 1 run    │                │
│ Cred  │  └──────────┘ └──────────┘ └──────────┘                │
│       │                                                         │
│       │  Recent Runs                                            │
│       │  ┌─────────────────────────────────────────────────┐   │
│       │  │ ● CFS Thematic Analysis    Completed  $0.42  ⋮  │   │
│       │  │ ○ Interview Classification Running... $0.12  ⋮  │   │
│       │  │ ● Survey Themes            Completed  $1.23  ⋮  │   │
│       │  └─────────────────────────────────────────────────┘   │
└───────┴─────────────────────────────────────────────────────────┘
```

### 4.2 Project View
```
┌─────────────────────────────────────────────────────────────────┐
│ ← Projects    CFS Study                    [Settings] [Archive] │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ [Documents] [Runs] [Comparisons]                               │
│ ─────────────────────────────────────────                      │
│                                                                 │
│ Documents (45)                      [+ Upload] [+ Create Set]   │
│ ┌───────────────────────────────────────────────────────────┐  │
│ │ □ Filename          Words    Uploaded      Set            │  │
│ │ ☑ transcript_01.txt 2,450    Jan 15       All, CFS-only   │  │
│ │ ☑ transcript_02.txt 3,120    Jan 15       All, CFS-only   │  │
│ │ □ transcript_03.txt 1,890    Jan 16       All             │  │
│ │ ...                                                        │  │
│ └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│ [Bulk: Add to Set ▼] [Delete Selected]                         │
│                                                                 │
│ Document Sets                                                   │
│ ┌──────────────────┐ ┌──────────────────┐                      │
│ │ All Documents    │ │ CFS-only         │ [+ New Set]          │
│ │ 45 documents     │ │ 28 documents     │                      │
│ └──────────────────┘ └──────────────────┘                      │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 New Run -- Pipeline Selection
```
┌─────────────────────────────────────────────────────────────────┐
│ ← CFS Study    New Analysis Run                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Step 1: Select Pipeline                                        │
│ ────────────────────────                                       │
│                                                                 │
│ System Pipelines                                               │
│ ┌──────────────────────────────────────────────────────────┐   │
│ │ [●] Thematic Analysis (Zero-Shot)                        │   │
│ │     Identify themes and codes from qualitative data      │   │
│ │     Nodes: Split → Map → Reduce                          │   │
│ │                                                          │   │
│ │ [ ] Thematic Analysis (Pre-Filtered)                     │   │
│ │     Filter excerpts before theme extraction              │   │
│ │                                                          │   │
│ │ [ ] Classification                                       │   │
│ │     Classify documents using multiple models             │   │
│ └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│ Custom Pipelines                               [+ Create New]   │
│ ┌──────────────────────────────────────────────────────────┐   │
│ │ (No custom pipelines yet)                                │   │
│ └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│                                          [Cancel] [Next →]      │
└─────────────────────────────────────────────────────────────────┘
```

### 4.4 New Run -- Configuration (Offcanvas Pattern)
```
┌─────────────────────────────────────────────────────────────────┐
│ ← CFS Study    Configure: Thematic Analysis                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────┬────────────────┐  │
│  │                                         │                │  │
│  │  Pipeline Preview                       │ Configuration  │  │
│  │  ┌─────────────────────────────────┐   │ ─────────────  │  │
│  │  │                                 │   │                │  │
│  │  │    [documents]                  │   │ Research       │  │
│  │  │         │                       │   │ Question:      │  │
│  │  │         ▼                       │   │ ┌────────────┐ │  │
│  │  │      [split]                    │   │ │What themes │ │  │
│  │  │         │                       │   │ │emerge from │ │  │
│  │  │         ▼                       │   │ │this data?  │ │  │
│  │  │       [map]                     │   │ └────────────┘ │  │
│  │  │         │                       │   │                │  │
│  │  │         ▼                       │   │ Persona:       │  │
│  │  │     [reduce]                    │   │ ┌────────────┐ │  │
│  │  │                                 │   │ │Experienced │ │  │
│  │  │  (Mermaid DAG diagram)          │   │ │researcher  │ │  │
│  │  │                                 │   │ └────────────┘ │  │
│  │  └─────────────────────────────────┘   │                │  │
│  │                                         │ ▼ Advanced     │  │
│  │  Documents: CFS-only (28 files)        │ ─────────────  │  │
│  │  [Change ▼]                            │ Model: gpt-4o  │  │
│  │                                         │ Temp: 0.7  ═══│  │
│  │  Estimated Cost: ~$0.35-0.50           │ Chunk: 20000   │  │
│  │                                         │                │  │
│  └─────────────────────────────────────────┴────────────────┘  │
│                                                                 │
│                                   [← Back] [Run Analysis →]     │
└─────────────────────────────────────────────────────────────────┘
```

### 4.5 Run Progress (HTMX Live Updates)
```
┌─────────────────────────────────────────────────────────────────┐
│ ← CFS Study    Run: Thematic Analysis #4                [Stop] │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Status: Running                              Cost: $0.28       │
│  ════════════════════════════░░░░░░░░░░░  65%                  │
│                                                                 │
│  Nodes                                                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ ✓ split      Completed    0.0s     $0.00                 │  │
│  │ ● map        Running...   45.2s    $0.24    [12/28 docs] │  │
│  │ ○ reduce     Pending      -        -                     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Live Output (map)                                   [Expand]   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Processing: transcript_12.txt                            │  │
│  │ Codes found: 4                                           │  │
│  │ - "pacing_strategies" (3 quotes)                         │  │
│  │ - "medical_dismissal" (2 quotes)                         │  │
│  │ ...                                                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.6 Results View (Embedded HTML)
```
┌─────────────────────────────────────────────────────────────────┐
│ ← CFS Study    Run: Thematic Analysis #4         [↓] [Compare] │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ [Overview] [Themes] [Codes] [Pipeline Details] [Raw JSON]      │
│ ─────────────────────────────────────────────────────────────  │
│                                                                 │
│ ┌───────────────────────────────────────────────────────────┐  │
│ │                                                           │  │
│ │  (Embedded simple.html or pipeline.html content)          │  │
│ │                                                           │  │
│ │  Themes (8)                                               │  │
│ │  ┌─────────────────────────────────────────────────────┐  │  │
│ │  │ 1. Pacing and Energy Management                     │  │  │
│ │  │    Patients describe strategies for managing...     │  │  │
│ │  │    Codes: pacing_strategies, rest_periods, ...      │  │  │
│ │  │    ▼ Supporting quotes (12)                         │  │  │
│ │  └─────────────────────────────────────────────────────┘  │  │
│ │                                                           │  │
│ │  ┌─────────────────────────────────────────────────────┐  │  │
│ │  │ 2. Medical System Challenges                        │  │  │
│ │  │    ...                                              │  │  │
│ │  └─────────────────────────────────────────────────────┘  │  │
│ │                                                           │  │
│ └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│ Run Info: 28 docs | $0.42 | 2m 34s | gpt-4o | Jan 20, 2026    │
└─────────────────────────────────────────────────────────────────┘
```

### 4.7 Comparison View
```
┌─────────────────────────────────────────────────────────────────┐
│ ← CFS Study    Comparison: CFS vs COVID Themes                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Comparing: [Run #3: CFS] ↔ [Run #4: COVID]          [Settings] │
│ ─────────────────────────────────────────────────────────────  │
│                                                                 │
│ ┌───────────────────────────────────────────────────────────┐  │
│ │                                                           │  │
│ │  (Embedded comparison.html with all visualizations)       │  │
│ │                                                           │  │
│ │  - Similarity heatmaps                                    │  │
│ │  - Optimal transport Sankey diagrams                      │  │
│ │  - Hit rate / fidelity metrics                            │  │
│ │  - Best matches tables                                    │  │
│ │                                                           │  │
│ └───────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.8 Classification Template Editor
```
┌─────────────────────────────────────────────────────────────────┐
│ ← CFS Study    Classification: Edit Template                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────┬────────────────┐  │
│  │                                         │                │  │
│  │  Template Editor (Struckdown)           │ Preview        │  │
│  │  ┌─────────────────────────────────┐   │ ─────────────  │  │
│  │  │ Classify the following text:    │   │                │  │
│  │  │                                 │   │ Labels:        │  │
│  │  │ {{input}}                       │   │ - CFS          │  │
│  │  │                                 │   │ - Long COVID   │  │
│  │  │ Based on the content, what      │   │ - Unclear      │  │
│  │  │ diagnosis does this patient     │   │                │  │
│  │  │ describe?                       │   │ Output fields: │  │
│  │  │                                 │   │ - diagnosis    │  │
│  │  │ [[pick:diagnosis|CFS,COVID,     │   │ - confidence   │  │
│  │  │   Unclear]]                     │   │                │  │
│  │  │                                 │   │ Detected       │  │
│  │  │ Confidence (1-5):               │   │ syntax: valid  │  │
│  │  │ [[int:confidence]]              │   │                │  │
│  │  │                                 │   │                │  │
│  │  │ (syntax highlighting active)    │   │                │  │
│  │  └─────────────────────────────────┘   │                │  │
│  │                                         │                │  │
│  │  [Reset to Default] [Validate]         │ [Help]         │  │
│  └─────────────────────────────────────────┴────────────────┘  │
│                                                                 │
│                              [← Back] [Save & Configure Run →]  │
└─────────────────────────────────────────────────────────────────┘
```

**Editor Features** (inspired by struckdown playground):
- Syntax highlighting for struckdown: `[[slot]]`, `{{variable}}`, `{% jinja %}`
- Real-time validation with error highlighting
- Help panel (offcanvas) with syntax reference
- Preview panel showing detected output fields
- Reset to default template option

### 4.9 Credits / Billing Page
```
┌─────────────────────────────────────────────────────────────────┐
│ [Logo] Soak                    [Credits: $12.45] [User ▼]      │
├───────┬─────────────────────────────────────────────────────────┤
│       │                                                         │
│ Nav   │  Credits & Billing                                      │
│       │  ────────────────                                       │
│ Dash  │                                                         │
│ Proj  │  Current Balance: $12.45                [+ Add Credits] │
│ Runs  │                                                         │
│ Cred  │  ┌────────────────────────────────────────────────────┐│
│       │  │ Add Credits                                        ││
│       │  │ ○ $10   ○ $25   ○ $50   ○ $100   ○ Custom: [___]  ││
│       │  │                                    [Purchase]      ││
│       │  └────────────────────────────────────────────────────┘│
│       │                                                         │
│       │  Or: Use Your Own API Key              [Configure BYOK]│
│       │                                                         │
│       │  Usage History                                          │
│       │  ┌────────────────────────────────────────────────────┐│
│       │  │ Date       Run                   Cost    Balance   ││
│       │  │ Jan 20     CFS Thematic #4       $0.42   $12.45    ││
│       │  │ Jan 19     Interview Class       $0.18   $12.87    ││
│       │  │ Jan 19     Credit Purchase       +$15.00 $13.05    ││
│       │  └────────────────────────────────────────────────────┘│
└───────┴─────────────────────────────────────────────────────────┘
```

---

## 5. Technical Architecture

### 5.1 Django Project Structure
```
soak_web/
├── config/                 # Django settings, URLs, WSGI
│   ├── settings/
│   │   ├── base.py
│   │   ├── development.py
│   │   └── production.py
│   ├── urls.py
│   └── celery.py
├── apps/
│   ├── accounts/          # User auth, API keys, billing
│   ├── projects/          # Projects, documents, document sets
│   ├── pipelines/         # Pipeline templates, configuration
│   ├── runs/              # Run execution, monitoring, results
│   ├── comparisons/       # Comparison generation
│   └── billing/           # Credits, payments, usage tracking
├── templates/
│   ├── base.html
│   ├── components/        # Reusable HTMX partials
│   └── [app]/             # App-specific templates
├── static/
│   ├── css/
│   │   └── main.css       # Custom styles (Bootstrap base)
│   └── js/
│       └── app.js         # Minimal JS (HTMX does most work)
└── manage.py
```

### 5.2 Key Dependencies
```
# Core
Django>=5.0
celery[redis]
django-htmx
django-allauth

# Database
psycopg[binary]           # PostgreSQL
django-storages           # S3 for file storage (production)

# Security
django-cryptography       # Encrypted fields for API keys

# Payments
stripe

# Soak Integration
soak                      # The pipeline library itself
struckdown                # For LLM interactions

# UI
django-widget-tweaks      # Form styling
whitenoise                # Static files
```

### 5.3 Celery Task Architecture
```python
# runs/tasks.py

@shared_task(bind=True, max_retries=3)
def execute_pipeline(self, run_id: int):
    """Execute a soak pipeline asynchronously."""
    run = Run.objects.get(id=run_id)
    run.status = 'running'
    run.celery_task_id = self.request.id
    run.started_at = timezone.now()
    run.save()

    try:
        # Load pipeline template
        pipeline = load_template_bundle(run.pipeline_template.yaml_content)

        # Apply user config overrides
        pipeline.default_context.update(run.context_overrides)
        # ... configure documents, model, etc.

        # Execute with progress callbacks
        analysis, errors = asyncio.run(
            pipeline.run(progress_callback=lambda p: update_run_progress(run_id, p))
        )

        # Save results
        run.status = 'completed'
        # ... save QualitativeResult, costs, etc.

    except Exception as e:
        run.status = 'failed'
        run.error_message = str(e)

    finally:
        run.completed_at = timezone.now()
        run.save()
```

### 5.4 HTMX Patterns

**Progress Polling:**
```html
<!-- runs/progress.html -->
<div hx-get="{% url 'runs:progress' run.id %}"
     hx-trigger="every 2s [run.status == 'running']"
     hx-swap="outerHTML">
    {% include "runs/partials/progress_bar.html" %}
</div>
```

**Form Submission:**
```html
<!-- projects/upload.html -->
<form hx-post="{% url 'projects:upload_documents' project.id %}"
      hx-encoding="multipart/form-data"
      hx-target="#document-list"
      hx-swap="beforeend">
    <input type="file" name="files" multiple>
    <button type="submit">Upload</button>
</form>
```

**Offcanvas Configuration Panel:**
```html
<!-- runs/configure.html -->
<button hx-get="{% url 'runs:config_panel' pipeline.id %}"
        hx-target="#config-offcanvas-body"
        data-bs-toggle="offcanvas"
        data-bs-target="#config-offcanvas">
    Configure
</button>

<div class="offcanvas offcanvas-end" id="config-offcanvas">
    <div class="offcanvas-header">
        <h5>Configuration</h5>
    </div>
    <div class="offcanvas-body" id="config-offcanvas-body">
        <!-- HTMX loads config form here -->
    </div>
</div>
```

---

## 6. Security Considerations

1. **API Key Storage**: AES-256 encryption at rest, never logged or exposed in UI
2. **Input Validation**: Sanitize all user-provided context variables
3. **Rate Limiting**: django-ratelimit on API endpoints
4. **CSRF**: Django's built-in protection + HTMX headers
5. **File Upload**: Virus scanning, size limits, type validation
6. **Sandbox**: Pipeline execution in isolated environment (consider docker/subprocess)
7. **Credit Deduction**: Transactional, atomic operations to prevent double-spend

---

## 7. Remaining Design Decisions

These can be decided during implementation:

1. **Notifications**: Email notifications for long-running jobs, or browser-only? (Suggest: browser-only for V1)

2. **Export Formats**: What export formats beyond HTML reports? CSV, XLSX, JSON download? (Suggest: JSON + CSV for V1)

3. **Comparison Scope**: Compare runs across projects or only within same project? (Suggest: same project only for V1)

4. **Syntax Highlighting Library**: CodeMirror 5 (like struckdown) or Monaco Editor or simpler overlay approach?

---

## 8. Implementation Phases (Updated)

### Phase 1: Core MVP (Intern Focus)
**Goal**: End-to-end working system with UI mocks for advanced features

**Backend Priority**:
1. Django project setup with Celery + Redis + PostgreSQL
2. User auth (email-based, django-allauth)
3. Project CRUD
4. Document upload → text extraction → store in DB (discard files)
5. Pipeline execution via Celery (system templates only)
6. Results storage and HTML embedding
7. BYOK API key configuration

**Frontend Priority** (Bootstrap 5 + HTMX):
1. Base template with navigation
2. Dashboard with project cards
3. Project view with document table
4. Pipeline selection and configuration forms
5. Run progress with HTMX polling
6. Results view with embedded HTML

**UI Mocks** (static HTML, no backend):
- Classification template editor
- Comparison selection/view
- Credits/billing page

### Phase 2: Classification & Comparison
- Classification pipeline with struckdown template editor
- Comparison feature (select runs, generate, view)
- Document sets (create, assign documents)

### Phase 3: Billing
- Credit purchase (Stripe integration)
- Cost tracking and display
- Usage history
- Credit balance management

### Phase 4: Polish & Advanced
- Email notifications
- Export formats (CSV, JSON download)
- Performance optimisation
- Error handling improvements

---

## 9. Verification Plan

### Development Environment
```bash
# Terminal 1: Django
uv run python manage.py runserver

# Terminal 2: Celery worker
uv run celery -A config worker -l info

# Terminal 3: Redis (via docker or local)
docker run -p 6379:6379 redis:alpine
```

### Test Scenarios

1. **User Registration & Auth**
   - Register with email
   - Login/logout
   - Configure BYOK API key

2. **Document Management**
   - Create project
   - Upload .txt files (verify text stored, file discarded)
   - Upload .docx/.pdf (verify extraction)
   - Upload .csv/.xlsx (verify row expansion)
   - View document list with word counts

3. **Pipeline Execution**
   - Select system pipeline (e.g., "Thematic Analysis")
   - Configure context variables
   - Launch run
   - Verify HTMX progress polling updates UI
   - Verify node-by-node status changes
   - Verify cost accumulates
   - Verify results displayed on completion

4. **Results & Comparison** (Phase 2)
   - View embedded HTML results
   - Select two completed runs
   - Generate comparison
   - Verify comparison visualisations render

---

## 10. Implementation Notes for Intern

### Getting Started
1. Create new Django project: `django-admin startproject config .`
2. Create apps: `python manage.py startapp accounts`, etc.
3. Set up PostgreSQL database locally
4. Install soak as editable: `uv pip install -e /path/to/soaking`

### Key Files to Study in Soak
- `soak/cli.py` -- how pipelines are loaded and run
- `soak/specs.py` -- `load_template_bundle()` function
- `soak/models/pipeline.py` -- `QualitativeAnalysisPipeline.run()` and `to_html()`
- `soak/document_utils.py` -- text extraction from various file types

### HTMX Tips
```html
<!-- Polling pattern for run progress -->
<div id="run-status"
     hx-get="/runs/{{ run.id }}/status/"
     hx-trigger="every 2s"
     hx-swap="innerHTML">
    {% include "runs/partials/status.html" %}
</div>

<!-- Stop polling when complete -->
{% if run.status == 'completed' or run.status == 'failed' %}
<div hx-swap-oob="true" id="run-status">
    <!-- No more hx-trigger, polling stops -->
    {% include "runs/partials/status.html" %}
</div>
{% endif %}
```

### Celery Task Pattern
```python
from celery import shared_task
from django.utils import timezone

@shared_task(bind=True)
def execute_pipeline_task(self, run_id: int):
    from .models import Run, RunNode
    run = Run.objects.get(id=run_id)

    try:
        run.status = 'running'
        run.celery_task_id = self.request.id
        run.started_at = timezone.now()
        run.save()

        # Load and execute pipeline
        # (See soak CLI for reference)

        run.status = 'completed'
    except Exception as e:
        run.status = 'failed'
        run.error_message = str(e)
    finally:
        run.completed_at = timezone.now()
        run.save()
```

### Text Extraction
```python
from soak.document_utils import extract_text

def handle_upload(uploaded_file):
    # Save temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
        for chunk in uploaded_file.chunks():
            tmp.write(chunk)
        tmp_path = tmp.name

    # Extract text
    content = extract_text(tmp_path)

    # Handle spreadsheets (returns list of dicts)
    if isinstance(content, list):
        # Create one Document per row
        for i, row in enumerate(content):
            Document.objects.create(
                project=project,
                filename=f"{uploaded_file.name}__row_{i}",
                content=str(row),  # or specific column
                metadata=row,
            )
    else:
        # Single document
        Document.objects.create(
            project=project,
            filename=uploaded_file.name,
            content=content,
        )

    # Delete temp file
    os.unlink(tmp_path)
```

---

*Plan prepared for implementation. Intern should start with Phase 1 backend priority items, building the Django project structure and basic models first.*
