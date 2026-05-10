# Architecture Diagram

The Trust & Safety research pipeline combines Agentic Eval traces, ACPA memory
management, and Gemma 4 safety reasoning.

```mermaid
flowchart TD
    subgraph Kaggle["Kaggle Runtime"]
        D1["AgentEval Dataset<br/>/kaggle/input/agent-eval-scenarios"]
        C1["Config Files<br/>configs/app.toml<br/>configs/secrets.toml"]
    end

    subgraph Ingestion["Ingestion Layer"]
        L1["Dataset Loader<br/>CSV / JSONL / Parquet"]
        N1["Record Normalizer<br/>prompt, response, trace, label"]
        B1["Context Builder<br/>chunk + score + cite"]
    end

    subgraph ACPA["Adaptive Context Pruning Algorithm"]
        M1["ContextElement Cache"]
        T1["AccessPatternTracker<br/>LFU cache hits"]
        S1["Retention Scoring<br/>alpha*frequency + beta*importance<br/>+ gamma*recency + delta*dependency"]
        E1["Competitive Eviction<br/>remove cold bottom ratio"]
        P1["Pinned Dependency Graph<br/>citations never evicted"]
    end

    subgraph Gemma["Gemma 4 Safety Agents"]
        G1["Gemma 4 Safety Analyst<br/>initial risk analysis"]
        G2["Gemma 4 Adjudicator<br/>grounded final JSON"]
    end

    subgraph Outputs["Research Artifacts"]
        O1["Trust & Safety Findings<br/>risk, category, evidence, mitigations"]
        O2["ACPA Telemetry<br/>retained, evicted, pinned"]
        O3["results.jsonl"]
        O4["benchmark CSV/Markdown<br/>ACPA vs baselines"]
    end

    D1 --> L1 --> N1 --> B1 --> M1
    C1 --> G1
    C1 --> G2
    M1 --> G1
    G1 --> T1
    T1 --> M1
    M1 --> P1 --> S1 --> E1
    E1 --> G2
    G1 --> G2
    G2 --> O1
    E1 --> O2
    O1 --> O3
    O2 --> O3
    B1 --> O4
    E1 --> O4
```

## Data flow

1. The Kaggle notebook mounts the AgentEval dataset under
   `/kaggle/input/agent-eval-scenarios`.
2. The loader discovers supported tabular files and normalizes each row into an
   `AgenticEvalRecord`.
3. The context builder splits prompts, trajectories, tool calls, responses, and
   labels into `ContextElement` objects.
4. Gemma 4 produces an initial Trust & Safety analysis.
5. The access tracker records which context elements Gemma referenced.
6. ACPA computes LFU/LRU/importance/dependency retention scores and prunes cold
   context while preserving citation-bearing evidence.
7. Gemma 4 adjudicates the final result using only the retained, grounded
   context and the initial analysis.
8. The pipeline writes JSONL records for analysis notebooks, hackathon demos,
   and publication appendices.

The offline benchmark path reuses the same ingestion and context builder, then
compares ACPA with no pruning, random eviction, LRU, LFU, importance ranking,
and sliding-window truncation. It writes per-record CSV details plus an
aggregate Markdown report without calling Gemma or requiring API keys.

## ACPA scoring

```text
score = alpha * frequency + beta * importance + gamma * recency + delta * dependency
```

- **frequency**: access count multiplied by cache priority.
- **importance**: lightweight safety-keyword and information-density score.
- **recency**: exponential LRU decay, `0.9 ** age`.
- **dependency**: pinned citation boost for evidence that must not be evicted.

The default `prune_ratio` is `0.45`, so ACPA removes approximately the coldest
45% of non-pinned context at each adjudication step.
