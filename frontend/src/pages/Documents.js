import React, { useState, useEffect } from "react";
import axios from "axios";
import { API } from "../App";

export default function Documents() {
  const [docs, setDocs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchDocs = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await axios.get(`${API}/api/v1/documents`);
      setDocs(res.data?.documents || res.data || []);
    } catch (e) {
      setError("Could not fetch documents. Is the API running?");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDocs();
  }, []);

  return (
    <div
      style={{ maxWidth: 780, margin: "0 auto", padding: "40px 32px" }}
      className="fade-in"
    >
      {/* Header */}
      <div
        style={{
          display: "flex",
          alignItems: "flex-start",
          justifyContent: "space-between",
          marginBottom: 32,
        }}
      >
        <div>
          <h1 style={{ fontSize: 28, fontWeight: 600, marginBottom: 6 }}>
            Documents
          </h1>
          <p style={{ color: "var(--text2)", fontSize: 14 }}>
            All indexed CVs in the vector store.
          </p>
        </div>
        <button
          onClick={fetchDocs}
          style={{
            background: "var(--bg2)",
            border: "1px solid var(--border)",
            borderRadius: "var(--radius-sm)",
            padding: "8px 16px",
            fontSize: 13,
            color: "var(--text2)",
            fontWeight: 500,
            display: "flex",
            alignItems: "center",
            gap: 6,
            transition: "all 0.15s",
          }}
          onMouseEnter={(e) =>
            (e.currentTarget.style.borderColor = "var(--blue-mid)")
          }
          onMouseLeave={(e) =>
            (e.currentTarget.style.borderColor = "var(--border)")
          }
        >
          ↻ Refresh
        </button>
      </div>

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

      {/* Loading */}
      {loading && (
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {[1, 2, 3].map((i) => (
            <div
              key={i}
              style={{
                background: "var(--bg2)",
                border: "1px solid var(--border)",
                borderRadius: "var(--radius)",
                padding: 20,
                display: "flex",
                gap: 14,
                alignItems: "center",
              }}
            >
              <div
                style={{
                  width: 38,
                  height: 38,
                  borderRadius: 8,
                  background: "var(--bg3)",
                  animation: "pulse 1.5s infinite",
                }}
              />
              <div style={{ flex: 1 }}>
                <div
                  style={{
                    height: 12,
                    width: "40%",
                    background: "var(--bg3)",
                    borderRadius: 4,
                    marginBottom: 8,
                    animation: "pulse 1.5s infinite",
                  }}
                />
                <div
                  style={{
                    height: 10,
                    width: "25%",
                    background: "var(--bg3)",
                    borderRadius: 4,
                    animation: "pulse 1.5s infinite",
                  }}
                />
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Empty */}
      {!loading && !error && docs.length === 0 && (
        <div
          style={{
            background: "var(--bg2)",
            border: "1px solid var(--border)",
            borderRadius: "var(--radius)",
            padding: "48px 32px",
            textAlign: "center",
          }}
        >
          <div style={{ fontSize: 40, marginBottom: 16 }}>📭</div>
          <p style={{ fontWeight: 500, color: "var(--text)", marginBottom: 6 }}>
            No documents indexed yet
          </p>
          <p style={{ fontSize: 13, color: "var(--text3)" }}>
            Upload CVs from the Upload page to get started.
          </p>
        </div>
      )}

      {/* Document list */}
      {!loading && docs.length > 0 && (
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
            <span
              style={{
                fontSize: 11,
                fontWeight: 600,
                color: "var(--text3)",
                letterSpacing: "0.08em",
                textTransform: "uppercase",
              }}
            >
              Indexed CVs
            </span>
            <span
              style={{
                background: "var(--blue-light)",
                color: "var(--blue)",
                border: "1px solid #c8d8f0",
                padding: "2px 10px",
                borderRadius: 20,
                fontSize: 11,
                fontWeight: 600,
              }}
            >
              {docs.length} total
            </span>
          </div>
          {docs.map((doc, i) => {
            const name =
              doc.filename ||
              doc.name ||
              doc.document_id ||
              `Document ${i + 1}`;
            const chunks = doc.chunks || doc.chunk_count || "—";
            const docId = doc.document_id || doc.id || "—";
            return (
              <div
                key={i}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 14,
                  padding: "14px 20px",
                  borderBottom:
                    i < docs.length - 1 ? "1px solid var(--border)" : "none",
                }}
              >
                <div
                  style={{
                    width: 38,
                    height: 38,
                    borderRadius: 8,
                    background: "var(--blue-light)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: 18,
                    flexShrink: 0,
                  }}
                >
                  📄
                </div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <p
                    style={{
                      fontWeight: 500,
                      fontSize: 14,
                      color: "var(--text)",
                      marginBottom: 3,
                    }}
                  >
                    {name}
                  </p>
                  <p
                    style={{
                      fontSize: 11,
                      color: "var(--text3)",
                      fontFamily: "var(--mono)",
                    }}
                  >
                    {chunks !== "—" ? `${chunks} chunks · ` : ""}
                    {docId !== "—"
                      ? `id: ${String(docId).slice(0, 16)}...`
                      : ""}
                  </p>
                </div>
                <span
                  style={{
                    background: "var(--green-light)",
                    color: "var(--green)",
                    border: "1px solid #b7ddc8",
                    padding: "3px 10px",
                    borderRadius: 20,
                    fontSize: 11,
                    fontWeight: 500,
                    flexShrink: 0,
                  }}
                >
                  indexed
                </span>
              </div>
            );
          })}
        </div>
      )}

      {/* Stats row */}
      {!loading && docs.length > 0 && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr 1fr",
            gap: 12,
            marginTop: 16,
          }}
        >
          {[
            {
              label: "Total CVs",
              value: docs.length,
              color: "var(--blue)",
              bg: "var(--blue-light)",
            },
            {
              label: "Total Chunks",
              value: docs.reduce((a, d) => a + (d.chunks || 0), 0) || "—",
              color: "var(--purple)",
              bg: "var(--purple-light)",
            },
            {
              label: "Vector Store",
              value: "FAISS",
              color: "var(--green)",
              bg: "var(--green-light)",
            },
          ].map((s, i) => (
            <div
              key={i}
              style={{
                background: s.bg,
                border: `1px solid ${s.color}30`,
                borderRadius: "var(--radius-sm)",
                padding: "14px 16px",
                textAlign: "center",
              }}
            >
              <p
                style={{
                  fontSize: 20,
                  fontWeight: 600,
                  color: s.color,
                  marginBottom: 2,
                }}
              >
                {s.value}
              </p>
              <p style={{ fontSize: 11, color: s.color, opacity: 0.8 }}>
                {s.label}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
