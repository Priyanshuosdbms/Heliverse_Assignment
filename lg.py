Here is a comprehensive, production-grade master prompt designed to be fed into any advanced AI code generator (like Claude, GPT, or an LLM coding assistant). It explicitly structures your tech stack (Next.js + FastAPI) and leaves absolutely no stones unturned based on your exact planning notes.

Copy and paste the text below directly into your AI tool of choice:

---

# Master System Prompt: Industrial Redmine Ticket Checker & Split-View Analyzer

## 1. Project Overview & Objective

You are an expert enterprise systems architect and full-stack software engineer. Your goal is to generate a fully functioning, production-ready workspace application called **"H.D. (Planning & Design)"**.
The app is a specialized **Redmine Ticket Checker** that provides users with a side-by-side workspace within Google Chrome. It displays a master controls dashboard on one side and a specific Redmine ticket instance on the other. It captures real-time modifications, evaluates updates against semantic tags, and leverages an LLM to generate intelligent summaries and structural task tracking.

### Core Tech Stack

* **Frontend:** Next.js (App Router, TypeScript, Tailwind CSS, shadcn/ui, Lucide Icons)
* **Split-Panel Mechanics:** `react-resizable-panels` (or a cross-origin window/extension bridge if frames are blocked)
* **Backend:** FastAPI (Python 3.11+, Pydantic v2, SQLAlchemy/SQLModel, Async/Await)
* **Database:** PostgreSQL (for saving evaluation history, config states, and instance tokens)
* **LLM Integration:** LangChain / OpenAI or Anthropic API SDKs mapped within FastAPI

---

## 2. Comprehensive Architectural & Step-by-Step Workflow

### Step 1: The Master Home Page (Next.js Dashboard)

* **UI Components:** Build a sleek, minimalist home dashboard layout.
* **The Core Trigger:** Provide a central component titled **"Redmine Ticket Summary & Checker"**. This component lists active tickets fetched via the FastAPI backend from the Redmine REST API.
* **Multi-Instance Logic:** When a user clicks a ticket, the system must cross-reference available Redmine deployment URLs. If multiple Redmine instances exist/are mapped to that project context, intercept the click and display an elegant shadcn/ui Modal/Dialog. The modal must prompt the user: *"Multiple Redmine instances detected. Please select the correct workspace to launch."*
* **Routing Action:** Once selected, capture the specific ticket ID and target instance base URL, then transition the user into the Workspace Shell view.

### Step 2: The Split Workspace Window Shell

* **Visual Layout:** Replicate a clean native split-view dashboard (simulating Chrome's side-by-side browser look). The left side must render the **"Redmine Checker Controls"** (Our custom Next.js UI). The right side must load the **Target Redmine Ticket**.
* **Technical Implementation Requirement:** 1.  *Primary Path:* Attempt an `<iframe>` implementation using `react-resizable-panels`. To prevent `X-Frame-Options: SAMEORIGIN` blockages typical of enterprise Redmine setups, the FastAPI backend must act as a streaming reverse-proxy or rewrite headers *OR* provide a fallback button that triggers a perfectly positioned, programmatic side-by-side browser window placement via standard `window.open(..., left, width)` configurations.
2.  *State Preservation:* Ensure that split divider width percentages are cached in `localStorage` so a user’s layout preference persists on page refresh.

### Step 3: Real-Time Data Capture & Action Pipeline

* **The Action Button:** On the left side ("Redmine Checker" dashboard), place a prominent, high-visibility action button labeled **"Capture & Analyze Ticket Updates"**.
* **The Capture Event:** When the user initiates a ticket update or clicks this button, the Next.js frontend sends an async request to FastAPI. FastAPI uses the Redmine API token to scrape the live text payload, historical journals, and comments of that explicit ticket ID.

---

## 3. Deep-Dive Backend Features & Functional Scope (FastAPI & LLM)

You must write explicit FastAPI route endpoints (`/api/v1/tickets/...`) to execute the following core evaluation criteria:

### Feature A: Evaluation & Tag Validation

* Parse the captured ticket markdown text and journals.
* Compare updates against project-specific "Relevant Tags" (e.g., `#bugfix`, `#feature-dev`, `#deployment`, `#code-review`).
* Evaluate whether the changes logged in the ticket structurally and logically align with the tags assigned to it. Flag anomalies if a tag states one thing but text logs describe another.

### Feature B: Automated Strategic Summarization

Send the cleaned text data to an LLM provider using a highly engineered, deterministic system prompt. The LLM output must be parsed cleanly into a structured JSON schema using Pydantic, returning exactly two sections:

1. **What has been done so far:** A highly organized, chronological, and structural breakdown of historical progress ("planned form").
2. **Current Track & Impediments:** A focused summary of what the team is currently working on next, along with an explicit list of highlighted blockers or issues currently faced by the engineering team.

### Feature C: Historical Archiving & Context Management

* **Historying Tool:** Build a dedicated PostgreSQL database table (`ticket_history`) that logs every automated check, storing the ticket ID, timestamp, the captured text snapshot, and the resulting LLM evaluation payload. This allows users to view previous "snapshots" of their ticket health over time.
* **Context Window Mitigation (Crucial Constraint):** Long-running Redmine tickets contain thousands of lines of messy HTML/Markdown strings. Implement a rigorous data-cleaning utility in FastAPI before feeding data to the LLM:
* Strip out redundant CSS, user signatures, heavy markup tags, and repeating boilerplate navigation links.
* Implement a token counter tool (`tiktoken`). If the token payload exceeds standard model safety limits, execute an iterative summary truncation loop or a semantic chunk-merging approach to prevent LLM context overflows.



---

## 4. Technical Deliverables & Code Layout Request

Please generate the file structures and complete code blocks for this application following this layout:

1. **FastAPI Backend (`main.py`, `schemas.py`, `analyzer.py`):**
* Async endpoints for fetching Redmine tickets, managing multi-tenant instance settings, and processing LLM analysis pipelines.
* Clean error boundaries for missing Redmine API keys or token expirations.


2. **Next.js Frontend (`/app/dashboard/page.tsx`, `/components/SplitWorkspace.tsx`):**
* Tailwind implementation of the side-by-side draggable panels.
* Dialog boxes for handling multiple ticket instances.
* Interactive states for the main analyzer button showing loading frames and rendering the final dual-section structured summary outputs cleanly.



*Begin generating the code step-by-step, starting with the FastAPI architecture.*
