import React, { useState } from "react";
import axios from "axios";
import { API } from "../App";

const EXAMPLES = [
  "What are the key skills of this candidate?",
  "Summarize the candidate's work experience.",
  "Which candidate has the most Python experience?",
  "Who has experience with machine learning and NLP?",
  "List candidates with FastAPI or Django experience.",
  "Which candidate is best suited for a backend role?",
];

export default function Query() {
  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(5);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);

  const handleQuery = async () => {
    if (!question.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await axios.post(`${API}/api/v1/query`, {
        question,
        top_k: topK,
        document_id: null,
      });
      const data = res.data;
      setResult({
        answer: data.answer,
        sources: data.sources || [],
        model: data.model,
        latency: data.latency,
      });
      setHistory((prev) => [{ question, ts: Date.now() }, ...prev.slice(0, 4)]);
    } catch (e) {
      setError(
        e.response?.data?.detail ||
          "Query failed. Make sure CVs are uploaded first.",
      );
    } finally {
      setLoading(false);
    }
  };

  const handleKey = (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) handleQuery();
  };

  return (
    <div style={{ maxWidth: 780, margin: "0 auto", padding: "40px 32px" }}>
      {/* Header */}
      <div style={{ marginBottom: 32 }}>
        <h1 style={{ fontSize: 28, fontWeight: 600, marginBottom: 6 }}>
          Query Candidates
        </h1>
        <p style={{ color: "var(--text2)", fontSize: 14 }}>
          Ask natural language questions across all indexed CVs.
        </p>
      </div>

      {/* Query box */}
      <div
        style={{
          background: "var(--bg2)",
          border: "1px solid var(--border)",
          borderRadius: "var(--radius)",
          padding: 20,
          marginBottom: 16,
        }}
      >
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleKey}
          placeholder="Which candidate has the most Python experience?"
          style={{
            width: "100%",
            border: "none",
            background: "transparent",
            resize: "none",
            fontSize: 15,
            color: "var(--text)",
            lineHeight: 1.6,
            minHeight: 80,
          }}
        />
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            marginTop: 12,
            paddingTop: 12,
            borderTop: "1px solid var(--border)",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <label style={{ fontSize: 12, color: "var(--text3)" }}>
              Top K results:
            </label>
            <select
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              style={{
                border: "1px solid var(--border)",
                borderRadius: 6,
                padding: "3px 8px",
                fontSize: 12,
                background: "var(--bg)",
                color: "var(--text)",
              }}
            >
              {[3, 5, 8, 10].map((k) => (
                <option key={k} value={k}>
                  {k}
                </option>
              ))}
            </select>
            <span style={{ fontSize: 11, color: "var(--text3)" }}>
              ⌘+Enter to search
            </span>
          </div>
          <button
            onClick={handleQuery}
            disabled={!question.trim() || loading}
            style={{
              background:
                question.trim() && !loading ? "var(--blue)" : "var(--bg3)",
              color: question.trim() && !loading ? "white" : "var(--text3)",
              padding: "8px 20px",
              borderRadius: "var(--radius-sm)",
              fontWeight: 500,
              fontSize: 13,
              display: "flex",
              alignItems: "center",
              gap: 8,
              transition: "all 0.2s",
            }}
          >
            {loading && (
              <div
                style={{
                  width: 12,
                  height: 12,
                  borderRadius: "50%",
                  border: "2px solid rgba(255,255,255,0.3)",
                  borderTopColor: "white",
                  animation: "spin 0.7s linear infinite",
                }}
              />
            )}
            {loading ? "Searching..." : "Search"}
          </button>
        </div>
      </div>

      {/* Example queries */}
      {!result && !loading && (
        <div style={{ marginBottom: 28 }}>
          <p
            style={{
              fontSize: 11,
              color: "var(--text3)",
              marginBottom: 8,
              fontWeight: 600,
              letterSpacing: "0.06em",
              textTransform: "uppercase",
            }}
          >
            Example queries
          </p>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
            {EXAMPLES.map((ex, i) => (
              <button
                key={i}
                onClick={() => setQuestion(ex)}
                style={{
                  background: "var(--bg2)",
                  border: "1px solid var(--border)",
                  borderRadius: 20,
                  padding: "5px 12px",
                  fontSize: 12,
                  color: "var(--text2)",
                  transition: "all 0.15s",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = "var(--blue-mid)";
                  e.currentTarget.style.color = "var(--blue)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = "var(--border)";
                  e.currentTarget.style.color = "var(--text2)";
                }}
              >
                {ex}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div
          style={{
            background: "var(--red-light)",
            border: "1px solid #fecaca",
            borderRadius: "var(--radius-sm)",
            padding: "12px 16px",
            color: "var(--red)",
            fontSize: 13,
            marginBottom: 16,
          }}
        >
          {error}
        </div>
      )}

      {/* Loading skeleton */}
      {loading && (
        <div
          style={{
            background: "var(--bg2)",
            border: "1px solid var(--border)",
            borderRadius: "var(--radius)",
            padding: 24,
          }}
        >
          {[80, 100, 60].map((w, i) => (
            <div
              key={i}
              style={{
                height: 14,
                borderRadius: 4,
                background: "var(--bg3)",
                width: `${w}%`,
                marginBottom: 10,
                animation: "pulse 1.5s infinite",
                animationDelay: `${i * 0.1}s`,
              }}
            />
          ))}
        </div>
      )}

      {/* Result */}
      {result && (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          {/* Answer box */}
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
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
              }}
            >
              <span style={{ fontWeight: 600, fontSize: 13 }}>Answer</span>
              <div style={{ display: "flex", gap: 8 }}>
                {result.model && (
                  <span
                    style={{
                      background: "var(--blue-light)",
                      color: "var(--blue)",
                      border: "1px solid #c8d8f0",
                      padding: "2px 8px",
                      borderRadius: 20,
                      fontSize: 11,
                      fontWeight: 500,
                    }}
                  >
                    {result.model.includes("llama")
                      ? "llama 3.3"
                      : result.model}
                  </span>
                )}
                {result.latency && (
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
                    {Math.round(result.latency)}ms
                  </span>
                )}
              </div>
            </div>
            <div style={{ padding: 20 }}>
              <p
                style={{ fontSize: 14, lineHeight: 1.7, color: "var(--text)" }}
              >
                {result.answer}
              </p>
            </div>
          </div>

          {/* Sources — only show if sources exist */}
          {result.sources && result.sources.length > 0 && (
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
                Sources ({result.sources.length})
              </div>
              {result.sources.map((s, i) => (
                <div
                  key={i}
                  style={{
                    padding: "14px 20px",
                    borderBottom:
                      i < result.sources.length - 1
                        ? "1px solid var(--border)"
                        : "none",
                    display: "flex",
                    alignItems: "flex-start",
                    gap: 12,
                  }}
                >
                  <div
                    style={{
                      width: 32,
                      height: 32,
                      borderRadius: 8,
                      background: "var(--purple-light)",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontSize: 16,
                      flexShrink: 0,
                    }}
                  >
                    📎
                  </div>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    {/* Document ID + page + score badges */}
                    <div
                      style={{
                        display: "flex",
                        gap: 6,
                        marginBottom: 6,
                        alignItems: "center",
                        flexWrap: "wrap",
                      }}
                    >
                      <span
                        style={{
                          fontSize: 11,
                          fontWeight: 600,
                          color: "var(--text)",
                          fontFamily: "var(--mono)",
                        }}
                      >
                        {s.document_id
                          ? `${String(s.document_id).slice(0, 18)}...`
                          : `Source ${i + 1}`}
                      </span>
                      {s.page !== undefined && (
                        <span
                          style={{
                            background: "var(--blue-light)",
                            color: "var(--blue)",
                            border: "1px solid #c8d8f0",
                            padding: "1px 7px",
                            borderRadius: 10,
                            fontSize: 10,
                            fontWeight: 500,
                          }}
                        >
                          page {s.page}
                        </span>
                      )}
                      {s.score !== undefined && (
                        <span
                          style={{
                            background: "var(--green-light)",
                            color: "var(--green)",
                            border: "1px solid #b7ddc8",
                            padding: "1px 7px",
                            borderRadius: 10,
                            fontSize: 10,
                            fontWeight: 500,
                          }}
                        >
                          score {Number(s.score).toFixed(3)}
                        </span>
                      )}
                    </div>
                    {/* Content preview */}
                    {s.content && (
                      <p
                        style={{
                          fontSize: 11,
                          color: "var(--text3)",
                          lineHeight: 1.5,
                          fontFamily: "var(--mono)",
                          display: "-webkit-box",
                          WebkitLineClamp: 2,
                          WebkitBoxOrient: "vertical",
                          overflow: "hidden",
                        }}
                      >
                        {s.content}
                      </p>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Recent query history */}
      {history.length > 1 && (
        <div style={{ marginTop: 32 }}>
          <p
            style={{
              fontSize: 11,
              color: "var(--text3)",
              marginBottom: 12,
              fontWeight: 600,
              letterSpacing: "0.06em",
              textTransform: "uppercase",
            }}
          >
            Recent queries
          </p>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {history.slice(1).map((h, i) => (
              <button
                key={i}
                onClick={() => setQuestion(h.question)}
                style={{
                  background: "var(--bg2)",
                  border: "1px solid var(--border)",
                  borderRadius: "var(--radius-sm)",
                  padding: "10px 14px",
                  textAlign: "left",
                  fontSize: 13,
                  color: "var(--text2)",
                  transition: "all 0.15s",
                }}
                onMouseEnter={(e) =>
                  (e.currentTarget.style.borderColor = "var(--blue-mid)")
                }
                onMouseLeave={(e) =>
                  (e.currentTarget.style.borderColor = "var(--border)")
                }
              >
                🔍 {h.question}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
