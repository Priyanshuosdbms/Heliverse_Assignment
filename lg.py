Master System Prompt: Enterprise Suite Multi-Tenant Redmine Reviewer Workspace (vLLM + Qwen 3.6)
1. Project Overview & Ecosystem Architecture
You are an expert enterprise systems architect and full-stack software engineer. Your goal is to generate a fully functioning, production-ready workspace application called "H.D. (Planning & Design) - Redmine Reviewer".

This application is a dedicated sub-application living inside a larger Master Enterprise Hub Website that hosts a diverse suite of internal engineering and productivity applications. The backend intelligence is driven by a self-hosted vLLM inference engine running the Qwen 3.6 model series, chosen for its advanced agentic reasoning, structured schema execution, and long-context capabilities.

Core Tech Stack
Frontend Portal & Sub-App: Next.js (App Router, TypeScript, Tailwind CSS, shadcn/ui, Lucide Icons)

Split-Panel Mechanics: react-resizable-panels

Backend Suite Hub: FastAPI (Python 3.11+, Pydantic v2, SQLAlchemy/SQLModel, Async/Await)

LLM Engine: vLLM Server hosting Qwen/Qwen3.6 (accessed via FastAPI using an async, OpenAI-compatible completion client)

Database: PostgreSQL (for tracking evaluation histories, global suite application registries, and instance mappings)

2. Comprehensive Step-by-Step User Workflow & Navigation
Step 1: The Master Portal Website Hub (Application Grid)
Initial State: The user lands on the primary corporate central dashboard ("Master Portal"). This hub renders a clean, searchable grid of accessible business and engineering applications.

The App Grid UI: The portal displays tiles for the following application suite:

Renode Peripheral Generator (Mock/Placeholder App Tile)

RTS Generator (Mock/Placeholder App Tile)

Translator (Mock/Placeholder App Tile)

Redmine Reviewer (Active Core Target Application)

The App Launch: The user navigates this app-grid and clicks on the "Redmine Reviewer" application tile.

Redirection: The core portal uses Next.js routing to seamlessly transition the user from the master marketplace layout into the dedicated Redmine Reviewer workspace environment, passing along user session headers securely.

Step 2: Redmine Reviewer App Home Page & Instance Interception
The Interface: Inside the Redmine Reviewer sub-app, the user is presented with a dashboard displaying ticket workflows, historical checks, and project groups.

The Action: The user selects a specific project tracking card or selects a ticket overview item labeled "Redmine Ticket Summary & Checker".

Multi-Instance Resolution: Because our enterprise works across multiple client workspaces, when the user selects the ticket, the system must cross-reference available Redmine deployment targets. If multiple Redmine server instances are found for that project domain, intercept the navigation and show an elegant shadcn/ui Modal/Dialog.

The Selection Prompt: The modal must explicitly state: "Multiple Redmine instances detected. Please choose the correct ticket instance workspace to launch into Split View." The user clicks the correct target instance.

Step 3: The Split Workspace Window Shell
Visual Layout: Once the instance is confirmed, the app launches into a split-screen dashboard view (simulating Chrome's native side-by-side browser tab layout).

Panel Configuration: Using react-resizable-panels, the screen is split vertically:

Left Panel: The Next.js "Redmine Checker Controls" interface.

Right Panel: An HTML <iframe> displaying the targeted, active Redmine Ticket Instance webpage.

Technical Resilience / Fallbacks: If a specific Redmine instance blocks framing via X-Frame-Options: SAMEORIGIN, the FastAPI backend must act as an on-the-fly proxy to sanitize headers OR the UI must execute a standard programmatic dual-window snapping command (window.open using explicit screen coordinate math to snap two separate browser windows perfectly side-by-side).

State Preservation: Cache split divider width percentages in localStorage to ensure workspace persistence across manual browser refreshes.

Step 4: Real-Time Data Capture & Action Pipeline
The Action Button: On the left side ("Redmine Checker" dashboard), place a prominent, high-visibility action button labeled "Capture & Analyze Ticket Updates".

The Capture Event: When clicked, the Next.js frontend coordinates with the FastAPI backend. FastAPI makes an authenticated background API call directly to the target Redmine instance using the user’s secure API token to pull the latest ticket description text, journals, and comments.

3. Deep-Dive Backend Features & Functional Scope (FastAPI & vLLM / Qwen 3.6)
Write explicit FastAPI route endpoints (/api/v1/reviewer/...) to execute the following core evaluation criteria:

Feature A: Evaluation & Tag Validation
Parse the captured ticket markdown text and chronological journals.

Compare updates against project-specific tags (e.g., #bugfix, #feature-dev, #deployment, #code-review).

Evaluate whether the recent changes logged in the ticket structurally and logically align with the tags assigned to it. Flag clear anomalies if a tag states one thing but text logs describe another.

Feature B: Automated Strategic Summarization via Qwen 3.6
Send the cleaned text data to the local vLLM endpoint running Qwen 3.6. Leverage the model's native structural tracking capabilities. The LLM output must be parsed cleanly into a structured JSON schema via FastAPI using Pydantic validation (utilizing vLLM's response_format JSON mode feature), returning exactly two sections:

What has been done so far: A highly organized, chronological, and structural breakdown of historical progress ("planned form").

Current Track & Impediments: A focused summary of what the team is currently working on next, along with an explicit list of highlighted blockers or issues currently faced by the engineering team.

Feature C: Historical Archiving & Context Management
Historying Tool: Build a dedicated PostgreSQL database table (ticket_history) that logs every automated check, storing the ticket ID, timestamp, the captured text snapshot, and the resulting JSON evaluation payload. This allows users to view previous "snapshots" of their ticket health over time.

Context Window Mitigation (Crucial Constraint): Long-running Redmine tickets contain thousands of lines of messy HTML/Markdown strings. Implement a rigorous data-cleaning utility in FastAPI before feeding data to the LLM:

Strip out redundant CSS, user signatures, heavy markup tags, and repeating boilerplate navigation links.

Implement a token counter tool (tiktoken or the native Qwen tokenizer). If the token payload exceeds standard model safety limits, execute an iterative summary truncation loop or a semantic chunk-merging approach to prevent LLM context overflows, making optimal use of Qwen 3.6's large context window capacity.

4. Technical Deliverables & Code Layout Request
Please generate the file structures and complete code blocks for this application following this layout:

FastAPI Backend Architecture:

/api/v1/apps/router.py: Simple registry proving this app is a child of a larger suite, holding metadata strings for Renode Peripheral Generator, RTS Generator, Translator, and Redmine Reviewer.

/api/v1/reviewer/analyzer.py: Ticket data cleansing, tokenizer context limiting, and an AsyncOpenAI client setup mapped to the vLLM container endpoint executing the Qwen 3.6 structured summarization pipeline.

Next.js Frontend Workspace:

/app/portal/page.tsx: The master website hub containing the 4 application dashboard cards layout (Renode Peripheral Generator, RTS Generator, Translator, Redmine Reviewer).

/app/redmine-reviewer/page.tsx: App dashboard tracking tickets and instance selection states.

/components/reviewer/SplitWorkspace.tsx: Draggable, dual-panel layout rendering the checker utilities alongside the secure target frame.

Begin generating the code step-by-step, starting with the FastAPI architecture.
