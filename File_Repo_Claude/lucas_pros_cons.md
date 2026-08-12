# lucasastorian/llmwiki — Pros & Cons (with evidence from the actual code)

> Written with your vLLM + Qwen3.6 setup specifically in mind, since this repo's
> architecture makes that question more complicated than a "point it at a custom
> endpoint" setting.

---

## TL;DR

This tool **does not call an LLM to write your wiki at all** — that job is entirely
delegated to whatever MCP client you connect (Claude Code, Claude Desktop, Claude
Cowork, or Codex). There is no config screen for "use my own vLLM/Qwen3.6 endpoint for
wiki generation," because wiki generation isn't something this codebase does. If your
goal is specifically "make Qwen3.6 write my wiki," this repo is very likely the wrong
tool — see §2.1 for the full explanation and what would actually need to be true for it
to work.

---

## 1. Pros

### 1.1 More permissive license than nashsu/llm_wiki
**Evidence:** `LICENSE` file + README badge — Apache 2.0, vs. GPLv3 for `nashsu/llm_wiki`.
**Example:** you can fork, modify, and embed this into a closed-source internal tool
without being obligated to release your changes — meaningfully lower friction if you
ever wanted to build a commercial product on top of it.

### 1.2 True local-first mode with a real privacy boundary
**Evidence:** `cmd_serve()`'s comment: *"Local mode is intentionally loopback-only: the
API listens on `127.0.0.1` and does not support LAN or remote binding."*
**Example:** unlike a lot of "local" tools that quietly still phone home, the network
binding itself is hard-restricted at the code level — there's no accidental-LAN-exposure
footgun here for your own documents.

### 1.3 Highlights and comments travel with the source, not lost in a side panel
**Evidence:** `services/highlight_chunks.py`, `services/highlight_merge.py`, and MCP
tools `comments.py`/`references.py` merge Chrome-extension-captured highlights/margin
notes into the same content the connected AI reads.
**Example:** if you highlight a paragraph in a PDF and leave a note "this contradicts
what X said," Claude sees both the source text *and* your annotation together when
building the wiki page — not just the raw document.

### 1.4 Visualizations are first-class wiki content
**Evidence:** README explicitly lists *"Visualizations — Charts and other
visualizations, including SVGs and Mermaid diagrams"* as a feature, and the guide
architecture gives the connected AI free rein to write arbitrary markdown/content into
pages.
**Example:** because the "writer" is a full agentic LLM (Claude) rather than a
constrained JSON-block parser (as in `nashsu/llm_wiki`'s `parseFileBlocks()`), it can
embed a Mermaid diagram inline in a concept page without needing this codebase to have
explicit support for that content type — it's just markdown Claude decided to write.

### 1.5 Self-maintenance is a documented first-class workflow, not a manual re-run
**Evidence:** README's "Make it self-maintaining" section, recommending Claude Code
Routines or Desktop scheduled tasks running a fixed nightly prompt.
**Example:** you set this up once, and (per the design intent) your wiki keeps
absorbing new sources/clips without you needing to remember to run `ingest` — a genuine
difference from ingest-on-demand tools.

### 1.6 Extension captures both webpages and PDFs, with inline annotation
**Evidence:** README: *"Clip webpages and PDFs as you read, highlight key sections, and
leave comments that Claude can see over MCP."*
**Example:** broader capture surface than a typical single-purpose web clipper — you
get PDF clipping in the same tool as webpage clipping, with your own thinking attached.

---

## 2. Cons (with concrete failure examples)

### 2.1 No native vLLM/Ollama/custom-endpoint support for wiki generation — this is a hard architectural mismatch with your setup
**Evidence:** exhaustive search of the codebase for `ollama`, `vllm`, or a generic
`base_url`/custom-endpoint config found **nothing**. The only model references anywhere
in the code are: Mistral (OCR only, `api/services/ocr.py`), Voyage AI (embeddings only,
optional), and Cloudflare Workers AI with a hardcoded model (`quiz_grader.py`, quiz
grading only). None of these touch actual wiki-page generation.

**What this means concretely for you:**
The entire "read sources, decide what pages to write, synthesize, cross-link" job
happens **inside whatever MCP client you connect** — Claude Code, Claude Desktop, Claude
Cowork, or Codex, as explicitly named in the README. None of those four clients, as
shipped, let you swap their underlying model for a self-hosted Qwen3.6 instance — Claude
Code/Desktop/Cowork run Anthropic's models, Codex runs OpenAI's. There is no setting
inside `llmwiki` itself to change this, because the model choice isn't `llmwiki`'s to
make — it belongs to the MCP client.

**The only theoretical path to using Qwen3.6 here** would be running a *different*,
custom-built MCP client/agent loop that (a) speaks the MCP protocol to connect to
`llmwiki`'s tools, and (b) is itself backed by your vLLM server instead of Claude/Codex.
That's a meaningfully larger build than "change a setting" — you'd be writing your own
agent harness that reads `GUIDE_TEXT`, calls the `search`/`read`/`create`/`edit` tools
in a loop, and manages its own tool-calling logic against Qwen3.6's function-calling
format. `nashsu/llm_wiki`, by contrast, gets you this for free via its Custom provider
setting.

### 2.2 PDF OCR quality has the same "good tier is gated" pattern as nashsu's MinerU
**Evidence:** README: *"Optional: ... a `MISTRAL_API_KEY` for higher-quality PDF OCR."*
Without it, extraction falls back to `pdf_extract.py`'s local `opendataloader-pdf` path.
**Example:** identical failure mode to what you hit with `nashsu/llm_wiki` — a scanned
or complex-layout PDF run through the baseline local extractor alone may come out thin
or scrambled, and you'd need to add a Mistral API key (an external, paid dependency) to
get the better path, even though everything else about this tool is meant to run
locally.

### 2.3 Several advertised capabilities are hosted-mode-only
**Evidence:** `mcp/tools/ingest.py`'s `add_source_from_url` docstring says *"(hosted
mode only)"* explicitly; `.env.example` shows S3/converter-service config that's only
relevant when `AWS_ACCESS_KEY_ID`/`S3_BUCKET` are set for hosted uploads.
**Example:** "pull in this arXiv paper by URL" — a feature the README highlights — won't
work if you're running fully local; you'd need to manually download the PDF yourself
first and drop it in your workspace folder instead.

### 2.4 Setup has more moving parts than a single binary/AppImage
**Evidence:** README's install steps: Python 3.11+, Node.js 20+, a virtualenv, two
separate `pip install` targets (`api/requirements.txt`, `mcp/requirements.txt`), `npm
install` for the web app, and optionally LibreOffice for Word/PowerPoint extraction.
**Example:** compared to `nashsu/llm_wiki`'s single downloadable `.deb`/`.AppImage`,
this is a genuine multi-runtime local dev setup — closer to standing up a small web app
stack than installing a desktop application, which raises the bar if you just want
something running quickly (this is also exactly the kind of setup that tends to hit the
same category of dependency friction you ran into earlier with `libwebkit2gtk`, just in
Python/Node package form instead of system libraries).

### 2.5 Wiki quality is entirely dependent on the connected agent's own judgment and session behavior, with no deterministic safety net
**Evidence:** unlike `nashsu/llm_wiki`'s `updateWikiIndexDeterministically()` (plain
code, not LLM-generated) and `isSafeIngestPath()` (hard-coded path validation),
`llmwiki`'s mutation tools (`create`/`edit`/`append`/`delete`) are called directly by
the connected agent with no equivalent deterministic guard code visible in `mcp/tools/`.
**Example:** if Claude (or Codex) misreads the Guide's instructions in a given session —
say, decides to restructure the whole Concepts taxonomy against the documented
convention — there's no structural lint pass or deterministic index rebuild forcing it
back in line the way there is in `nashsu/llm_wiki`; you're relying on the agent
following `GUIDE_TEXT` correctly every session, with `lint.py`/`quiz_lint.py` as
tools the agent has to *choose* to invoke rather than something that runs automatically.

### 2.6 Quiz grading has a hardcoded model choice you can't change
**Evidence:** `quiz_grader.py`: `GRADER_MODEL = "@cf/google/gemma-4-26b-a4b-it"`, called
via Cloudflare Workers AI, with no config option shown to swap it.
**Example:** if you wanted the quiz feature graded by Qwen3.6 too for consistency, that
specific piece is not swappable without editing the source directly — a small but real
inconsistency in an otherwise "bring your own model" adjacent design.

---

## 3. Bottom line for your use case

Given you're specifically trying to run **vLLM + Qwen3.6** as the model doing the work:
`nashsu/llm_wiki`'s Custom-provider setting is a first-class, supported path.
`lucasastorian/llmwiki` has **no equivalent path** — its entire value proposition
assumes you're bringing a Claude or Codex subscription as the "brain," with this repo
providing the workspace and tools around it. Worth treating this as a different category
of tool (an MCP-based knowledge workspace for an existing agentic coding assistant)
rather than a like-for-like alternative to `nashsu/llm_wiki` for your setup.
