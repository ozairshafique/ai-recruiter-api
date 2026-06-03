import React, { useState, useEffect } from "react";
import "./index.css";
import Upload from "./pages/Upload";
import Query from "./pages/Query";
import Documents from "./pages/Documents";
import Analytics from "./pages/Analytics";
import axios from "axios";

export const API = "https://ozair1112-ai-recruiter-api.hf.space";

const NAV = [
  { id: "upload", label: "Upload CV", icon: UploadIcon },
  { id: "query", label: "Query", icon: SearchIcon },
  { id: "documents", label: "Documents", icon: DocIcon },
  { id: "analytics", label: "Analytics", icon: ChartIcon },
];

export default function App() {
  const [page, setPage] = useState("upload");
  const [apiOk, setApiOk] = useState(null);

  useEffect(() => {
    axios
      .get(`${API}/api/v1/health`)
      .then(() => setApiOk(true))
      .catch(() => setApiOk(false));
  }, []);

  const pages = {
    upload: Upload,
    query: Query,
    documents: Documents,
    analytics: Analytics,
  };
  const Page = pages[page];

  return (
    <div style={{ display: "flex", height: "100vh", overflow: "hidden" }}>
      {/* Sidebar */}
      <aside
        style={{
          width: 240,
          background: "var(--bg2)",
          borderRight: "1px solid var(--border)",
          display: "flex",
          flexDirection: "column",
          padding: "24px 16px",
          flexShrink: 0,
        }}
      >
        {/* Logo */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 10,
            marginBottom: 32,
            paddingLeft: 8,
          }}
        >
          <div
            style={{
              width: 32,
              height: 32,
              borderRadius: 8,
              background: "var(--blue)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <span style={{ color: "white", fontSize: 16 }}>🤖</span>
          </div>
          <span style={{ fontWeight: 600, fontSize: 15, color: "var(--text)" }}>
            AI Recruiter
          </span>
        </div>

        {/* Nav */}
        <nav
          style={{ display: "flex", flexDirection: "column", gap: 4, flex: 1 }}
        >
          {NAV.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setPage(id)}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 10,
                padding: "10px 12px",
                borderRadius: "var(--radius-sm)",
                background: page === id ? "var(--blue-light)" : "transparent",
                color: page === id ? "var(--blue)" : "var(--text2)",
                fontWeight: page === id ? 500 : 400,
                fontSize: 14,
                textAlign: "left",
                transition: "all 0.15s ease",
                border:
                  page === id ? "1px solid #c8d8f0" : "1px solid transparent",
              }}
              onMouseEnter={(e) => {
                if (page !== id)
                  e.currentTarget.style.background = "var(--bg3)";
              }}
              onMouseLeave={(e) => {
                if (page !== id)
                  e.currentTarget.style.background = "transparent";
              }}
            >
              <Icon
                size={16}
                color={page === id ? "var(--blue)" : "var(--text3)"}
              />
              {label}
            </button>
          ))}
        </nav>

        {/* API status */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            padding: "10px 12px",
            borderRadius: "var(--radius-sm)",
            background:
              apiOk === null
                ? "var(--bg3)"
                : apiOk
                  ? "var(--green-light)"
                  : "var(--red-light)",
            border: `1px solid ${apiOk === null ? "var(--border)" : apiOk ? "#b7ddc8" : "#fecaca"}`,
          }}
        >
          <div
            style={{
              width: 8,
              height: 8,
              borderRadius: "50%",
              background:
                apiOk === null
                  ? "var(--text3)"
                  : apiOk
                    ? "var(--green)"
                    : "var(--red)",
              animation: apiOk === null ? "pulse 1.5s infinite" : "none",
            }}
          />
          <span
            style={{
              fontSize: 12,
              color:
                apiOk === null
                  ? "var(--text3)"
                  : apiOk
                    ? "var(--green)"
                    : "var(--red)",
              fontWeight: 500,
            }}
          >
            {apiOk === null
              ? "Connecting..."
              : apiOk
                ? "API connected"
                : "API offline"}
          </span>
        </div>
      </aside>

      {/* Main content */}
      <main style={{ flex: 1, overflow: "auto", background: "var(--bg)" }}>
        <Page />
      </main>
    </div>
  );
}

/* ── Icons ── */
function UploadIcon({ size = 16, color = "currentColor" }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke={color}
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" y1="3" x2="12" y2="15" />
    </svg>
  );
}
function SearchIcon({ size = 16, color = "currentColor" }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke={color}
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="11" cy="11" r="8" />
      <line x1="21" y1="21" x2="16.65" y2="16.65" />
    </svg>
  );
}
function DocIcon({ size = 16, color = "currentColor" }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke={color}
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
      <line x1="16" y1="13" x2="8" y2="13" />
      <line x1="16" y1="17" x2="8" y2="17" />
    </svg>
  );
}
function ChartIcon({ size = 16, color = "currentColor" }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke={color}
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <line x1="18" y1="20" x2="18" y2="10" />
      <line x1="12" y1="20" x2="12" y2="4" />
      <line x1="6" y1="20" x2="6" y2="14" />
    </svg>
  );
}
