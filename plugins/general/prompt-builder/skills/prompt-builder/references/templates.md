# Prompt Templates Reference

Read the relevant template when the user's task type matches. Do not load all templates — only the one you need.

---

## Template A — RTF

*Role, Task, Format. Use for fast one-shot tasks where the request is clear and simple.*

```
Role: [One sentence defining who the AI is]
Task: [Precise verb + what to produce]
Format: [Exact output format and length]
```

**Example:**
```
Role: You are a senior technical writer.
Task: Write a one-paragraph description of what a REST API is.
Format: Plain prose, 3 sentences maximum, no jargon, suitable for a non-technical audience.
```

---

## Template B — CO-STAR

*Context, Objective, Style, Tone, Audience, Response. Use for professional documents, business writing, reports, and marketing content.*

```
Context: [Background the AI needs to understand the situation]
Objective: [Exact goal — what success looks like]
Style: [Writing style: formal / conversational / technical / narrative]
Tone: [Emotional register: authoritative / empathetic / urgent / neutral]
Audience: [Who reads this — their knowledge level and expectations]
Response: [Format, length, and structure of the output]
```

**Example:**
```
Context: I am a founder pitching a B2B SaaS tool that automates expense reporting for mid-size companies.
Objective: Write a cold email that gets a reply from a CFO.
Style: Direct and conversational, not salesy.
Tone: Confident but not pushy.
Audience: CFO at a 200-person company, busy, skeptical of vendor emails.
Response: 5 sentences max. Subject line included. No bullet points.
```

---

## Template C — RISEN

*Role, Instructions, Steps, End Goal, Narrowing. Use for complex multi-step projects requiring a clear sequence.*

```
Role: [Expert identity the AI should adopt]
Instructions: [Overall task in plain terms]
Steps:
  1. [First action]
  2. [Second action]
  3. [Continue as needed]
End Goal: [What the final output must achieve]
Narrowing: [Constraints, scope limits, what to exclude]
```

**Example:**
```
Role: You are a product manager with 10 years of experience in mobile apps.
Instructions: Write a product requirements document for a habit tracking feature.
Steps:
  1. Define the problem statement in one paragraph
  2. List user stories in "As a [user], I want [goal] so that [reason]" format
  3. Define acceptance criteria for each story
  4. List out-of-scope items explicitly
End Goal: A PRD that an engineering team can begin sprint planning from immediately.
Narrowing: No technical implementation details. No wireframes. Under 600 words total.
```

---

## Template D — Chain of Thought

*Use for logic-heavy tasks, math, debugging, and multi-factor analysis. ONLY for standard models (Claude, Gemini). NEVER for o3, o4-mini, DeepSeek-R1.*

```
[Task statement]

Before answering, think through this carefully:
<thinking>
1. What is the actual problem being asked?
2. What constraints must the solution respect?
3. What are the possible approaches?
4. Which approach is best and why?
</thinking>

Give your final answer after the thinking block.
```

**When to use:**
- Debugging where the cause is not obvious
- Comparing two technical approaches
- Any math or calculation
- Analysis where a wrong first impression is likely

**When NOT to use:**
- Reasoning-native models (they think internally — adding CoT hurts)
- Simple tasks where the answer is clear
- Creative tasks (CoT can kill natural voice)

---

## Template E — Few-Shot

*Use when the output format is easier to show than describe. 2-5 examples is the sweet spot.*

```
[Task instruction]

Here are examples of the exact format needed:

<examples>
  <example>
    <input>[example input 1]</input>
    <output>[example output 1]</output>
  </example>
  <example>
    <input>[example input 2]</input>
    <output>[example output 2]</output>
  </example>
</examples>

Now apply this exact pattern to: [actual input]
```

**Rules:**
- 2-5 examples. More rarely helps and wastes tokens.
- Include edge cases, not just easy cases.
- Use XML tags for Claude. For Gemini/Perplexity, use markdown headers or numbered blocks.
- If the user has been re-prompting for the same format issue twice, switch to few-shot.

---

## Template F — Agentic (ReAct + Stop Conditions)

*Use for Claude Code, autonomous agents, and any tool that runs commands or edits files.*

```
## Role
[Expert identity with specific domain]

## Current State
[What exists right now — files, project state, installed tools]

## Target State
[Exactly what should exist when done]

## Allowed Actions
- [Action 1]
- [Action 2]

## Forbidden Actions
- Do NOT [dangerous action 1]
- Do NOT [dangerous action 2]

## Steps
1. [First action]
2. [Second action]
3. [Continue as needed]

## Stop Conditions
Done when:
- [Binary criterion 1]
- [Binary criterion 2]

## Human Review Triggers
Stop and ask before:
- Deleting any file
- Adding any new dependency
- Running any command that affects production
```

**Example:**
```
## Role
Senior Node.js backend engineer specializing in REST APIs.

## Current State
Empty Express project. `src/app.js` exists with basic server setup. PostgreSQL running locally.

## Target State
`POST /api/users` endpoint in `src/routes/users.js` with input validation, bcrypt password hashing, and PostgreSQL insert.

## Allowed Actions
- Create files in `src/`
- Install packages via npm
- Run tests

## Forbidden Actions
- Do NOT modify `package.json` scripts
- Do NOT add authentication middleware (separate task)
- Do NOT touch `.env` or any config file

## Steps
1. Create `src/routes/users.js` with POST handler
2. Add input validation (email format, password min 8 chars)
3. Hash password with bcrypt (10 rounds)
4. Insert into `users` table
5. Return 201 with user ID (no password in response)

## Stop Conditions
Done when:
- POST /api/users returns 201 for valid input
- Returns 400 with error details for invalid input
- Password is never returned in any response

## Human Review Triggers
Stop and ask before:
- Adding any npm package not listed above
- Creating database migrations
```
