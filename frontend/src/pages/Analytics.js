import React from "react";

const METRICS = [
  {
    label: "Answer Relevancy",
    score: 0.94,
    threshold: 0.7,
    desc: "How relevant the answer is to the query",
  },
  {
    label: "Faithfulness",
    score: 0.91,
    threshold: 0.7,
    desc: "Answer is grounded in retrieved context",
  },
  {
    label: "Contextual Recall",
    score: 0.89,
    threshold: 0.7,
    desc: "Retrieved chunks cover the right content",
  },
  {
    label: "Contextual Precision",
    score: 0.92,
    threshold: 0.7,
    desc: "Retrieved chunks are precise and relevant",
  },
];

const STACK = [
  {
    layer: "API",
    tech: "FastAPI + Pydantic",
    color: "var(--blue)",
    bg: "var(--blue-light)",
  },
  {
    layer: "LLM",
    tech: "LLaMA 3.3 70B via Groq",
    color: "var(--green)",
    bg: "var(--green-light)",
  },
  {
    layer: "Embeddings",
    tech: "Cohere embed-multilingual-v3.0",
    color: "var(--green)",
    bg: "var(--green-light)",
  },
  {
    layer: "Vector Store",
    tech: "FAISS (cosine similarity)",
    color: "var(--purple)",
    bg: "var(--purple-light)",
  },
  {
    layer: "RAG",
    tech: "LangChain + LangGraph",
    color: "var(--green)",
    bg: "var(--green-light)",
  },
  {
    layer: "Observability",
    tech: "LangSmith",
    color: "var(--orange)",
    bg: "var(--orange-light)",
  },
  {
    layer: "Evaluation",
    tech: "DeepEval",
    color: "var(--orange)",
    bg: "var(--orange-light)",
  },
  {
    layer: "Deployment",
    tech: "HuggingFace Spaces + Docker",
    color: "var(--blue)",
    bg: "var(--blue-light)",
  },
];

function ScoreBar({ score, threshold }) {
  const pct = Math.round(score * 100);
  const thPct = Math.round(threshold * 100);
  return (
    <div
      style={{
        position: "relative",
        height: 8,
        background: "var(--bg3)",
        borderRadius: 4,
        overflow: "visible",
      }}
    >
      <div
        style={{
          height: "100%",
          borderRadius: 4,
          width: `${pct}%`,
          background: "linear-gradient(90deg, var(--green) 0%, #52c98a 100%)",
          transition: "width 1s ease",
        }}
      />
      {/* Threshold marker */}
      <div
        style={{
          position: "absolute",
          top: -3,
          left: `${thPct}%`,
          width: 2,
          height: 14,
          background: "var(--orange)",
          borderRadius: 1,
          transform: "translateX(-50%)",
        }}
      />
    </div>
  );
}

export default function Analytics() {
  const avg = (
    METRICS.reduce((a, m) => a + m.score, 0) / METRICS.length
  ).toFixed(2);

  return (
    <div
      style={{ maxWidth: 780, margin: "0 auto", padding: "40px 32px" }}
      className="fade-in"
    >
      {/* Header */}
      <div style={{ marginBottom: 32 }}>
        <h1 style={{ fontSize: 28, fontWeight: 600, marginBottom: 6 }}>
          Analytics
        </h1>
        <p style={{ color: "var(--text2)", fontSize: 14 }}>
          RAG evaluation results from DeepEval. All metrics exceed the 0.7
          threshold.
        </p>
      </div>

      {/* Summary cards */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(3, 1fr)",
          gap: 12,
          marginBottom: 28,
        }}
      >
        {[
          {
            label: "Average Score",
            value: avg,
            color: "var(--green)",
            bg: "var(--green-light)",
            border: "#b7ddc8",
          },
          {
            label: "Metrics Passing",
            value: `${METRICS.filter((m) => m.score >= m.threshold).length}/${METRICS.length}`,
            color: "var(--blue)",
            bg: "var(--blue-light)",
            border: "#c8d8f0",
          },
          {
            label: "Threshold",
            value: "0.70",
            color: "var(--orange)",
            bg: "var(--orange-light)",
            border: "#f5c99a",
          },
        ].map((c, i) => (
          <div
            key={i}
            style={{
              background: c.bg,
              border: `1px solid ${c.border}`,
              borderRadius: "var(--radius)",
              padding: "20px 20px",
              textAlign: "center",
            }}
          >
            <p
              style={{
                fontSize: 28,
                fontWeight: 700,
                color: c.color,
                marginBottom: 4,
              }}
            >
              {c.value}
            </p>
            <p
              style={{
                fontSize: 12,
                color: c.color,
                opacity: 0.8,
                fontWeight: 500,
              }}
            >
              {c.label}
            </p>
          </div>
        ))}
      </div>

      {/* Metric rows */}
      <div
        style={{
          background: "var(--bg2)",
          border: "1px solid var(--border)",
          borderRadius: "var(--radius)",
          overflow: "hidden",
          marginBottom: 28,
        }}
      >
        <div
          style={{
            padding: "12px 20px",
            borderBottom: "1px solid var(--border)",
            fontSize: 11,
            fontWeight: 600,
            color: "var(--text3)",
            letterSpacing: "0.08em",
            textTransform: "uppercase",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <span>DeepEval RAG Metrics</span>
          <span style={{ fontSize: 10 }}>Orange line = threshold (0.7)</span>
        </div>
        {METRICS.map((m, i) => (
          <div
            key={i}
            style={{
              padding: "18px 20px",
              borderBottom:
                i < METRICS.length - 1 ? "1px solid var(--border)" : "none",
            }}
          >
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                marginBottom: 10,
              }}
            >
              <div>
                <span
                  style={{
                    fontWeight: 500,
                    fontSize: 14,
                    color: "var(--text)",
                  }}
                >
                  {m.label}
                </span>
                <span
                  style={{ fontSize: 12, color: "var(--text3)", marginLeft: 8 }}
                >
                  {m.desc}
                </span>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span
                  style={{
                    fontWeight: 700,
                    fontSize: 18,
                    color:
                      m.score >= m.threshold ? "var(--green)" : "var(--red)",
                  }}
                >
                  {m.score.toFixed(2)}
                </span>
                <span
                  style={{
                    background: "var(--green-light)",
                    color: "var(--green)",
                    border: "1px solid #b7ddc8",
                    padding: "2px 8px",
                    borderRadius: 20,
                    fontSize: 11,
                    fontWeight: 500,
                  }}
                >
                  ✓ Pass
                </span>
              </div>
            </div>
            <ScoreBar score={m.score} threshold={m.threshold} />
          </div>
        ))}
      </div>

      {/* Tech stack */}
      <div
        style={{
          background: "var(--bg2)",
          border: "1px solid var(--border)",
          borderRadius: "var(--radius)",
          overflow: "hidden",
        }}
      >
        <div
          style={{
            padding: "12px 20px",
            borderBottom: "1px solid var(--border)",
            fontSize: 11,
            fontWeight: 600,
            color: "var(--text3)",
            letterSpacing: "0.08em",
            textTransform: "uppercase",
          }}
        >
          Tech Stack
        </div>
        <div
          style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 0 }}
        >
          {STACK.map((s, i) => (
            <div
              key={i}
              style={{
                padding: "12px 20px",
                borderBottom:
                  i < STACK.length - 2 ? "1px solid var(--border)" : "none",
                borderRight: i % 2 === 0 ? "1px solid var(--border)" : "none",
                display: "flex",
                alignItems: "center",
                gap: 12,
              }}
            >
              <span
                style={{
                  background: s.bg,
                  color: s.color,
                  padding: "3px 8px",
                  borderRadius: 6,
                  fontSize: 10,
                  fontWeight: 600,
                  flexShrink: 0,
                  minWidth: 72,
                  textAlign: "center",
                }}
              >
                {s.layer}
              </span>
              <span
                style={{
                  fontSize: 13,
                  color: "var(--text2)",
                  fontFamily: "var(--mono)",
                }}
              >
                {s.tech}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Footer note */}
      <p
        style={{
          fontSize: 12,
          color: "var(--text3)",
          marginTop: 16,
          textAlign: "center",
        }}
      >
        Evaluation run with DeepEval · Model: llama-3.3-70b-versatile ·
        Embeddings: Cohere embed-multilingual-v3.0
      </p>
    </div>
  );
}
