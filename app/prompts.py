
QUERY_GEN_PROMPT = """You are a research planner. Generate 3-5 diverse search queries
to investigate the user's question. Each query explores a different angle and has
a one-sentence rationale.

current date: {current_date}
Treat this date as the present and add to query context for time-sensitive topics (which are most of them).
Route each query to one category:
- Technical / ML / scientific questions: mix 'science' (papers, theory) with
  'it' (libraries, docs, implementations), and optionally 'general' for context.
- Historical / cultural questions: mix 'general' (context, narrative) with
  'science' (academic analysis). Do NOT use 'it' for non-technical topics.
- Current events: include 'news' alongside 'general'.

Default to 'general' when unsure."""

SUMMARIZE_PROMPT = """You are reading ONE source document to extract findings relevant to a research question.

Research question: {question}

First, relevance:
- Set relevant=false if the document does not actually address the question. A document that only shares a word with the question in an unrelated context is NOT relevant.
- If relevant=false, return an empty overview, key_findings, and quotes.

Rules:
- overview: 1-2 sentences orienting a downstream writer on what THIS document is — its angle, scope, or argument. Examples across domains: "Wikipedia survey of the topic, organised chronologically", "Peer-reviewed study (n=2,400) arguing X is the primary cause, not Y", "Vendor whitepaper promoting their product; useful for benchmark numbers but biased", "First-person blog post from a practitioner; anecdotal not statistical". This is NOT a recap of the facts — it's the document's shape and stance. Skip if not relevant.
- key_findings: 3-8 findings. DO NOT invent findings to reach count. 2 sharp findings are better than 3 padded ones.
- Each finding must state a specific date, number, named person/place/event or causal mechanism. 
- A causal-mechanism finding states cause then effect (X caused Y because Z), not a vague description.
- Do not hedge or generalize. Quote specifics.
- quotes: up to 4 short verbatim quotes (under 25 words each) that support your findings. Omit if add nothing.

Document title: {title}
URL: {url}

Document content:
{content}"""

EXTRACT_PROMPT = """You are AGGREGATING document findings into topic-level claims.

Research question: {question}

A claim is a TOPIC LABEL plus the doc_ids whose summaries cover that topic. The label
identifies the topic — it is NOT a full account. The actual evidence (dates, quotes,
specifics) stays inside the doc_summaries; the section writer reads those directly.

You receive:
1. EXISTING_CLAIMS — topic claims from earlier loops. Each has topic, source_count, confidence.
2. NEW_SUMMARIES — document summaries you have not yet processed (key_findings + quotes).

For each topic in NEW_SUMMARIES, do exactly ONE of:

(A) MERGE — the topic already appears in EXISTING_CLAIMS (same event, year, and
    entity, even if worded differently).
    - topic: copy the existing topic label VERBATIM, character for character.
    - source_doc_ids: ONLY the new doc_id(s) supporting it. Do not repeat doc_ids
      already counted in the existing claim.
    - Return at most ONE record per existing topic. If multiple new summaries
      support it, combine their doc_ids into one source_doc_ids list.
    - confidence: "high" if existing source_count + new sources >= 3, else "medium".

(B) NEW — no existing claim covers this topic.
    - topic: short label, under ~80 chars where possible. Include the key entity
      and date if applicable. Do NOT write a full explanatory sentence.
    - source_doc_ids: doc_ids from NEW_SUMMARIES supporting this topic.
    - confidence: high (>= 3 sources), medium (1-2), low (sources conflict).

(C) DROP — the topic is too vague to be useful ("various challenges", "industry
    concerns", "long-term trends"). Omit it from the output entirely.

Goal: ~one claim per distinct topic. Prefer fewer well-aggregated claims over many
fragments. Single-source topics are OK if specific.

GOOD CLAIMS (topic labels — mix of domains shown):
- "GPT-4 release, March 2023" (tech / product)
- "CRISPR-Cas9 mechanism: guide RNA + Cas9 cleavage" (science / mechanism)
- "Lehman Brothers bankruptcy, 15 Sep 2008" (finance / event)
- "Treaty of Westphalia 1648: sovereignty principle" (history / concept)
- "Transformer architecture self-attention scaling, O(n²)" (technical / property)

DROP (no entity, no date, no quantity, no named mechanism):
- "There were significant developments in the field."
- "Several factors contributed to the outcome."
- "The technology had wide-ranging impacts."

EXISTING_CLAIMS:
{existing_claims_json}

NEW_SUMMARIES:
{summaries_json}"""


REFLECT_PROMPT = """You are auditing research progress. Decide whether the gathered evidence \
sufficiently answers the research question, or if another search loop is needed.

Research question: 
{question}

Previous reflection from earlier loops:
{reflection_history}

Topics covered so far (claim labels — quick index of what's been touched):
{claims_index}

Previous evidence (document IDs, overviews, and selected key findings):
{previous_evidence_json}

New evidence from this loop (document IDs, overviews, and selected key findings):
{new_evidence_json}

Queries already searched (do not repeat or paraphrase these):
{previous_queries}

Budget Line, loop count: 
{budget_line}

Decide whether the evidence is sufficient to write a credible report.


How to judge:
- The claims index shows what topics have been touched. The evidence shows what's
  actually been learned. Judge from the evidence, not the claims.
- Sufficient means the main question can be answered with concrete support drawn
  from the evidence. Not exhaustively — credibly.
- A topic being broad is not a gap. A specific question the report cannot honestly
  answer from the evidence IS a gap.
- New evidence vs previous evidence: ask whether the new evidence resolves a gap
  the previous loop identified, or whether it just adds variety on already-covered
  ground.

If the previous loop generated follow-up queries that returned no new evidence, that gap is unfillable from web search. Don't ask it again — proceed with sufficient=true and acknowledge the limitation in current_understanding.

If you do request follow-ups:
- Search-engine queries, 4-12 words, keyword-dense. Not natural-language questions.
- One concrete gap, 1-3 queries attacking it from genuinely different angles.
- Must not paraphrase queries from previous loops.

Output:
- current_understanding: 2-5 sentences on what the evidence supports. If earlier gaps remain unfilled despite searching, say so.
- is_sufficient: true if the question can be answered credibly OR if remaining gaps are unfillable.
- knowledge_gap: empty if sufficient. Otherwise one concrete unresolved gap.
- follow_up_queries: empty if sufficient. Otherwise 1-3 queries."""


PLAN_PROMPT = """You are planning a grounded research report.

Current date: {current_date}
Treat this date as the present for any time sensitive topics.

Research question:
{question}

Final research understanding:
{final_understanding}

Topic claims:
{claims_json}

Evidence catalogue:
{evidence_json}

How to use the inputs:

- Claims are canonical topic handles.
- Each claim topic maps to supporting document summaries.
- The evidence catalogue contains the actual detail behind those topics.
- Use the evidence to decide which topics belong together in a section.
- Do not create a section whose angle is unsupported by the supplied evidence.
- Do not force every minor topic into the report.
- Avoid overlapping sections.

Produce a report plan with 3-5 sections.

Each section must:

- Have a distinct and specific angle.
- Include the exact claim topic labels needed to write that section.
- Copy claim topics exactly, character for character.
- Never paraphrase, shorten, or invent a claim topic.
- Contain only topics supported by the evidence catalogue.

Order sections logically, such as chronologically, causally, or from mechanism
to consequences.

This is a structural plan, not the report.

Output fields:

- title: report title, maximum 12 words
- sections:
  - title: section title
  - angle: one or two sentences describing the section scope
  - claim_topics: exact topic strings copied from Topic claims
"""

WRITE_SECTION_PROMPT = """You are writing ONE section of a grounded research report.

Research question: {question}
Current date: {current_date}
Section title: {title}
Section angle: {angle}

You have two inputs:
1. Claims: the topic handles selected for this section. They identify what the section should cover and which supporting summaries were retrieved.
2. Supporting source summaries: evidence and context from the documents behind those claims. Each has:
   - doc_id: the citation key (cite as [d_<doc_id>], e.g. [d_fb2843aa4730])
   - overview: what the document is and how it frames the topic
   - key_findings: specific facts (dates, numbers, named people/places/events)
   - quotes: short verbatim phrases you may incorporate

Claims:
{claims_json}

Supporting source summaries:
{summaries_json}

Rules:
- Write only the body of the section. Do not include a heading.
- Use the claims as the main factual structure; use the supporting summaries for context, chronology, and causal relationships.
- Cite each fact inline using the doc_id of the source it came from, in brackets with a 'd_' prefix: [d_fb2843aa4730]. Multiple sources for one fact: [d_fb2843aa4730, d_512264222188]. Place the citation immediately after the sentence or clause it supports.
- The overview tells you what each source actually argues — use that to attribute interpretations to specific sources rather than asserting them as fact.
- Do not introduce a factual detail unless it appears in either the claims or the supporting source summaries.
- Do not infer a causal relationship merely because two events appear together.
- If the evidence does not support part of the section angle, omit that point instead of inventing analysis.
- Avoid repeating facts already covered by the section unless needed for the argument.
- Write 150-500 words. Do not pad thin evidence into a long section.
- Do not include a concluding summary paragraph.

Output markdown only."""
