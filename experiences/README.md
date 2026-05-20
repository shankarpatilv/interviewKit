# Experience Documents

Experience documents are private markdown notes that InterviewKit turns into
retrieval chunks. Richer notes produce stronger interview answers because the
retriever can find concrete details and the model can cite real source files
instead of guessing.

## What To Write

Create one markdown file per project, role, incident, launch, or meaningful
work story. Use names that are safe to keep local, such as:

```text
experiences/payments-reliability.md
experiences/search-ranking-launch.md
```

Start from `_template.md` and fill in enough detail for a STAR answer:

- context and constraints
- your specific role
- technical decisions and alternatives
- metrics and impact
- conflicts, ambiguity, or tradeoffs
- lessons learned
- interview themes the story supports

## Why Detail Matters

InterviewKit stores each section as searchable context. A note that says
"improved reliability" is hard to use. A note that says what failed, what you
changed, why you chose that approach, and how reliability improved gives the
STAR synthesizer enough evidence to draft a grounded answer with citations.

## Privacy

Filled-in experience files are intentionally ignored by Git. Only
`_template.md` and this README should be committed from this directory. Do not
commit real personal experience documents, confidential company information,
API keys, customer data, or private interview notes.
