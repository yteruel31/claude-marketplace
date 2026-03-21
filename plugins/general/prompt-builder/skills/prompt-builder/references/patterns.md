# Anti-Pattern Reference

20 patterns that waste tokens and cause re-prompts. Read this when fixing a bad prompt or diagnosing why a prompt underperforms.

---

## Task Patterns

| # | Pattern | Bad | Fixed |
|---|---------|-----|-------|
| 1 | **Vague task verb** | "help me with my code" | "Refactor `getUserData()` to use async/await and handle null returns" |
| 2 | **Two tasks in one** | "explain AND rewrite this function" | Split: explain first (Prompt 1), rewrite second (Prompt 2) |
| 3 | **No success criteria** | "make it better" | "Done when the function passes existing tests and handles null input without throwing" |
| 4 | **Over-permissive agent** | "do whatever it takes" | Explicit allowed actions list + explicit forbidden actions list |
| 5 | **Emotional description** | "it's totally broken, fix everything" | "Throws TypeError on line 43 when `user` is null" |
| 6 | **Build-everything** | "build my entire app" | Break into Prompt 1 (scaffold), Prompt 2 (core feature), Prompt 3 (polish) |
| 7 | **Implicit reference** | "now add the other thing we discussed" | Always restate the full task — never reference "the thing" |

---

## Context Patterns

| # | Pattern | Bad | Fixed |
|---|---------|-----|-------|
| 8 | **Assumed prior knowledge** | "continue where we left off" | Include Memory Block with all prior decisions |
| 9 | **No project context** | "write a cover letter" | "PM role at B2B fintech, 2yr SWE experience, shipped 3 features as tech lead" |
| 10 | **Forgotten stack** | New prompt contradicts earlier tech choice | Always include Memory Block with established stack |
| 11 | **Hallucination invite** | "what do experts say about X?" | "Cite only sources you are certain of. If uncertain, say [uncertain]." |

---

## Format Patterns

| # | Pattern | Bad | Fixed |
|---|---------|-----|-------|
| 12 | **No output format** | "explain this concept" | "3 bullet points, each under 20 words, with a one-sentence summary at top" |
| 13 | **Implicit length** | "write a summary" | "Write a summary in exactly 3 sentences" |
| 14 | **No role for complex task** | (no persona) | "You are a senior backend engineer specializing in Node.js and PostgreSQL" |
| 15 | **Vague aesthetics** | "make it look professional" | "Monochrome palette, 16px base font, 24px line height, no decorative elements" |

---

## Scope Patterns

| # | Pattern | Bad | Fixed |
|---|---------|-----|-------|
| 16 | **No scope boundary** | "fix my app" | "Fix only the login form validation in `src/auth.js`. Touch nothing else." |
| 17 | **No stop condition for agents** | "build the whole feature" | Explicit stop conditions + checkpoint output after each step |
| 18 | **Entire codebase as context** | Full repo pasted every prompt | Scope to only the relevant function and file |

---

## Reasoning Patterns

| # | Pattern | Bad | Fixed |
|---|---------|-----|-------|
| 19 | **No CoT for logic task** | "which approach is better?" | "Think through both approaches step by step before recommending" |
| 20 | **CoT on reasoning models** | "think step by step" sent to o3/o4 | REMOVE IT — reasoning models think internally, CoT degrades output |
