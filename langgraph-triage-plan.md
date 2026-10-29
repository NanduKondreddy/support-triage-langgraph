# Plan: Convert the Support Triage Agent to LangGraph

Goal: turn the existing Multi-Domain Support Triage Agent into a LangGraph project that runs, is tested, and has real numbers in its README. Only after it works, add LangGraph and LangChain to the resume.

Time needed: one to two weeks of focused evenings.

## Rules to follow

- Keep the old pipeline working. Do not delete it.
- Do not add LangGraph or LangChain to the resume until the graph runs and the results match.
- Every number in the README must come from a run you did.
- Use a new branch called `langgraph-version`.

## Step 0: Check what you have (half a day)

1. Run this in the repo and note every file that matches:
   `grep -ri "langgraph\|langchain\|anthropic\|faiss" .`
2. Open the `code/` folder. Write down the function that does each stage: safety, resilience, intent, retrieval, confidence, grounding, escalation.
3. Run the old pipeline once and save the output file as `baseline_output.csv`. This is your proof later.
4. Find where `requirements.txt` is. The README says to install from the repo root. Move the file there or fix the README.

Done when: you have a list of stage functions and a saved baseline output.

## Step 1: Install and learn the basics (one day)

1. Create a virtual environment and install: `pip install langgraph langchain langchain-core pytest`
2. Build a tiny graph with three nodes and one conditional edge. Run it. This is only to learn the pattern.
3. Learn four ideas: state, node, edge, conditional edge.

Done when: your toy graph runs and routes to different nodes based on a value.

## Step 2: Design the state (half a day)

Create one state object that every node reads and updates. Suggested fields:

- `ticket_text`
- `domain`
- `safety_flag`
- `intent`
- `retrieved_chunks`
- `confidence`
- `grounded` (true or false)
- `decision` (answer or escalate)
- `response`

Done when: the state is written as a typed class in one file.

## Step 3: Turn each stage into a node (two to three days)

Make one node per stage. Each node takes the state and returns only the fields it changes. Reuse your existing code inside the nodes. Do not rewrite the logic.

| Node | Wraps your existing | Writes to state |
| --- | --- | --- |
| safety | PII and prompt injection filter | `safety_flag` |
| resilience | empty or gibberish input check | `safety_flag` |
| intent | intent mapping | `intent`, `domain` |
| retrieve | BM25 plus embedding retriever | `retrieved_chunks`, `confidence` |
| ground | n-gram and semantic grounding check | `grounded` |
| answer | response builder | `response`, `decision` |
| escalate | hand off to a human | `response`, `decision` |

Wrap your retriever as a LangChain retriever so the LangChain part is real, not only a name.

Done when: each node runs alone on a sample ticket.

## Step 4: Wire the graph (one day)

1. Add edges in this order: safety, resilience, intent, retrieve, ground.
2. Add conditional edges:
   - If `safety_flag` is true, go straight to escalate.
   - After retrieve, if `confidence` is below 0.40, go to escalate.
   - After ground, if `grounded` is true go to answer, otherwise go to escalate.
3. Compile the graph and draw it with `graph.get_graph().draw_mermaid()`. Save the diagram for the README.

Done when: the graph runs one ticket from start to finish.

## Step 5: Compare with the old pipeline (one day)

1. Run the graph on the full ticket file and save `graph_output.csv`.
2. Compare it with `baseline_output.csv` row by row.
3. Count matching rows and list every difference. Fix differences caused by your wiring. Explain any that remain.
4. Run it twice and confirm the outputs are identical, since your README claims this.

Done when: you can write one honest sentence such as "X of Y tickets match the original pipeline."

## Step 6: Tests (one day)

Write pytest tests for:

- each node on one normal input and one bad input
- the three routing rules (safety, low confidence, failed grounding)
- one full run on three sample tickets

Done when: `pytest` passes.

## Step 7: Evaluation script (one day)

1. Make a small labelled set of 30 to 50 tickets with the correct decision (answer or escalate) for each.
2. Write `evaluate.py` that reports:
   - retrieval hit rate
   - escalation precision and recall
   - how many answers pass the grounding check
3. Put the results table in the README with the date you ran it.

Done when: the numbers come from running the script, not from memory.

## Step 8: API and packaging (one day)

1. Add a FastAPI endpoint that accepts a ticket and returns the decision and response.
2. Add a Dockerfile.
3. Add a sample request and sample response to the README.

Done when: `docker run` starts the service and one curl request returns a result.

## Step 9: README (half a day)

Include:

- one paragraph on what it does
- the graph diagram
- how to install and run
- the results table from Step 7
- the match result from Step 5
- limits, written honestly

## Step 10: Update the resume (after everything above works)

1. Add `LangGraph` and `LangChain` to the skills line.
2. Add or change one bullet on the triage project, using your real numbers. Example shape (fill in your own numbers):
   "Rebuilt the triage pipeline as a LangGraph state graph with conditional routing to human escalation. Matched X of Y original decisions and reached Z% escalation precision on a labelled set of N tickets. Added pytest tests and a Docker service."
3. Push the branch, merge to main, and check the repo link opens.

## Optional extra: image segmentation (two days)

Only if you want that JD line. Run Segment Anything or a U-Net on a few images in a notebook and save the outputs in a small repo with a README. Add "image segmentation" to the resume only after this exists.

## Final checklist

- [ ] Baseline output saved before any change
- [ ] Graph runs end to end
- [ ] Output compared with baseline, result written down
- [ ] Two runs give identical output
- [ ] pytest passes
- [ ] Evaluation numbers come from the script
- [ ] Docker service starts
- [ ] README has diagram, results and limits
- [ ] Resume updated only with what is proven
- [ ] You can explain every node and every routing rule without notes

## Questions to be ready for in an interview

- Why a graph and not a plain function chain?
- What is in the state and who writes each field?
- How does a conditional edge decide to escalate?
- How did you prove the new version matches the old one?
- What does not work yet?
