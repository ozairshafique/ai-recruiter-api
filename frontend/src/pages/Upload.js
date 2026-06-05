import React, { useState, useRef, useCallback } from "react";
import axios from "axios";
import { API } from "../App";

export default function Upload() {
  const [file, setFile] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [uploads, setUploads] = useState([]);
  const inputRef = useRef();

  const handleFile = (f) => {
    if (!f || f.type !== "application/pdf") {
      setError("Please select a PDF file.");
      return;
    }
    if (f.size > 10 * 1024 * 1024) {
      setError("File must be under 10MB.");
      return;
    }
    setError(null);
    setFile(f);
  };

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    handleFile(e.dataTransfer.files[0]);
  }, []);

  const onDragOver = (e) => {
    e.preventDefault();
    setDragging(true);
  };
  const onDragLeave = () => setDragging(false);

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    const fd = new FormData();
    fd.append("file", file);
    try {
      const res = await axios.post(`${API}/api/v1/upload`, fd);
      setUploads((prev) => [
        { ...res.data, name: file.name, ts: Date.now() },
        ...prev,
      ]);
      setFile(null);
    } catch (e) {
      setError(
        e.response?.data?.detail || "Upload failed. Is the API running?",
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{ maxWidth: 780, margin: "0 auto", padding: "40px 32px" }}
      className="fade-in"
    >
      {/* Header */}
      <div style={{ marginBottom: 32 }}>
        <h1
          style={{
            fontSize: 28,
            fontWeight: 600,
            color: "var(--text)",
            marginBottom: 6,
          }}
        >
          Upload CV
        </h1>
        <p style={{ color: "var(--text2)", fontSize: 14 }}>
          Upload PDF resumes to index them for querying.
        </p>
      </div>

      {/* Drop zone */}
      <div
        onClick={() => inputRef.current.click()}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        style={{
          background: dragging
            ? "var(--blue-light)"
            : file
              ? "var(--green-light)"
              : "var(--bg2)",
          border: `2px dashed ${dragging ? "var(--blue-mid)" : file ? "var(--green)" : "var(--border)"}`,
          borderRadius: "var(--radius)",
          padding: "48px 32px",
          textAlign: "center",
          cursor: "pointer",
          transition: "all 0.2s ease",
          marginBottom: 16,
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".pdf"
          style={{ display: "none" }}
          onChange={(e) => handleFile(e.target.files[0])}
        />
        <div style={{ marginBottom: 12 }}>
          {file ? (
            <span style={{ fontSize: 36 }}>📄</span>
          ) : (
            <svg
              width="40"
              height="40"
              viewBox="0 0 24 24"
              fill="none"
              stroke="var(--blue)"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
          )}
        </div>
        {file ? (
          <>
            <p
              style={{
                fontWeight: 500,
                color: "var(--green)",
                marginBottom: 4,
              }}
            >
              {file.name}
            </p>
            <p style={{ fontSize: 12, color: "var(--text3)" }}>
              {(file.size / 1024).toFixed(0)} KB · Click to change
            </p>
          </>
        ) : (
          <>
            <p
              style={{ fontWeight: 500, color: "var(--text)", marginBottom: 4 }}
            >
              Click to upload or drag and drop
            </p>
            <p style={{ fontSize: 12, color: "var(--text3)" }}>
              PDF files only · Max 10MB
            </p>
          </>
        )}
      </div>

      {/* Error */}
      {error && (
        <div
          style={{
            background: "var(--red-light)",
            border: "1px solid #fecaca",
            borderRadius: "var(--radius-sm)",
            padding: "10px 14px",
            color: "var(--red)",
            fontSize: 13,
            marginBottom: 16,
          }}
        >
          {error}
        </div>
      )}

      {/* Upload button */}
      <button
        onClick={handleUpload}
        disabled={!file || loading}
        style={{
          background: file && !loading ? "var(--blue)" : "var(--bg3)",
          color: file && !loading ? "white" : "var(--text3)",
          padding: "11px 28px",
          borderRadius: "var(--radius-sm)",
          fontWeight: 500,
          fontSize: 14,
          transition: "all 0.2s ease",
          display: "flex",
          alignItems: "center",
          gap: 8,
          marginBottom: 36,
        }}
      >
        {loading && (
          <div
            style={{
              width: 14,
              height: 14,
              borderRadius: "50%",
              border: "2px solid rgba(255,255,255,0.3)",
              borderTopColor: "white",
              animation: "spin 0.7s linear infinite",
            }}
          />
        )}
        {loading ? "Uploading..." : "Upload & Index"}
      </button>

      {/* Recent uploads */}
      {uploads.length > 0 && (
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
              padding: "14px 20px",
              borderBottom: "1px solid var(--border)",
              fontSize: 11,
              fontWeight: 600,
              color: "var(--text3)",
              letterSpacing: "0.08em",
              textTransform: "uppercase",
            }}
          >
            Recent Uploads
          </div>
          {uploads.map((u, i) => (
            <div
              key={i}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 14,
                padding: "14px 20px",
                borderBottom:
                  i < uploads.length - 1 ? "1px solid var(--border)" : "none",
                animation: "fadeIn 0.3s ease forwards",
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
                    marginBottom: 2,
                  }}
                >
                  {u.filename || u.name}
                </p>
                <p style={{ fontSize: 12, color: "var(--text3)" }}>
                  {u.chunks} chunks · {u.latency_ms?.toFixed(0) || "—"}ms
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
                }}
              >
                indexed
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
