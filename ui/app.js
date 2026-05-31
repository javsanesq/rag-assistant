const apiBase = "/api";

const state = {
  documents: [],
  jobs: [],
  evals: [],
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function fetchJson(url, options) {
  const token = localStorage.getItem("ragAssistantApiToken");
  const headers = { ...(options && options.headers ? options.headers : {}) };
  if (token) headers["x-api-key"] = token;
  const response = await fetch(url, { ...(options || {}), headers });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed: ${response.status}`);
  }
  return response.json();
}

function renderStatus(message, ok = true) {
  const node = document.getElementById("api-status");
  const plain = String(message).replace(/<[^>]*>/g, "").trim();
  node.textContent = ok ? plain : (plain.length > 40 ? plain.slice(0, 40) + "…" : plain);
  node.style.color = ok ? "var(--accent)" : "var(--danger)";
  node.title = plain;
}

function renderDocuments() {
  const list = document.getElementById("document-list");
  const select = document.getElementById("category-filter");
  select.innerHTML = '<option value="">All categories</option>';

  if (!state.documents.length) {
    list.innerHTML = '<p class="empty-state">No documents indexed yet.</p>';
    return;
  }

  const categories = new Set();
  list.innerHTML = state.documents
    .map((doc) => {
      if (doc.category) categories.add(doc.category);
      return `
        <article class="doc-row">
          <div class="doc-header">
            <div>
              <strong>${escapeHtml(doc.title)}</strong>
              <div class="doc-meta">${escapeHtml(doc.document_id)} · ${escapeHtml(doc.source_type)} · ${doc.chunk_count} chunks</div>
            </div>
            <button data-delete="${escapeHtml(doc.document_id)}">Delete</button>
          </div>
          <div class="doc-meta">${escapeHtml(doc.category || "uncategorized")} ${doc.document_date ? `· ${escapeHtml(doc.document_date)}` : ""}</div>
        </article>
      `;
    })
    .join("");

  [...categories].sort().forEach((category) => {
    const option = document.createElement("option");
    option.value = category;
    option.textContent = category;
    select.appendChild(option);
  });

  list.querySelectorAll("[data-delete]").forEach((button) => {
    button.addEventListener("click", async () => {
      await fetchJson(`${apiBase}/v1/documents/${button.dataset.delete}`, { method: "DELETE" });
      await refreshAll();
    });
  });
}

function statusClass(status) {
  if (status === "completed") return "ok";
  if (status === "failed") return "fail";
  if (status === "running") return "run";
  return "wait";
}

function formatTime(iso) {
  if (!iso) return "";
  return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function resultPairs(result) {
  if (!result || typeof result !== "object") return "";
  const rows = Object.entries(result)
    .filter(([, value]) => value === null || ["string", "number", "boolean"].includes(typeof value))
    .map(
      ([key, value]) =>
        `<div class="kv"><span>${escapeHtml(key.replaceAll("_", " "))}</span><strong>${escapeHtml(value ?? "—")}</strong></div>`
    )
    .join("");
  return rows ? `<div class="kv-grid">${rows}</div>` : "";
}

function renderJobs() {
  const list = document.getElementById("job-list");
  const jobs = state.jobs.filter((item) => item.job_type === "ingestion");
  if (!jobs.length) {
    list.innerHTML = '<p class="empty-state-sm">No ingestion jobs yet.</p>';
    return;
  }
  list.innerHTML = jobs
    .map((job) => {
      const time = formatTime(job.completed_at || job.started_at || job.created_at);
      return `
        <article class="timeline-row">
          <div class="timeline-header">
            <span class="status-badge ${statusClass(job.status)}">${escapeHtml(job.status)}</span>
            ${time ? `<span class="timeline-meta">${time}</span>` : ""}
          </div>
          ${
            job.status === "running" || job.progress < 100
              ? `<div class="progress-track"><div class="progress-fill" style="width:${job.progress ?? 0}%"></div></div>`
              : ""
          }
          ${job.attempts > 1 ? `<div class="metrics">attempts ${job.attempts}/${job.max_attempts ?? 3}</div>` : ""}
          ${job.error_message ? `<div class="metrics error-text">${escapeHtml(job.error_code || "error")}: ${escapeHtml(job.error_message)}</div>` : ""}
          ${resultPairs(job.result)}
        </article>
      `;
    })
    .join("");
}

function renderEvals() {
  const list = document.getElementById("eval-list");
  if (!state.evals.length) {
    list.innerHTML = '<p class="empty-state">No evaluation runs yet.</p>';
    return;
  }
  list.innerHTML = state.evals
    .map((job) => {
      const summary = (job.result && job.result.summary) || {};
      const metric = (value) => (value === null || value === undefined ? "—" : value);
      const byFilter = summary.by_filter || {};
      const filterRows = Object.entries(byFilter)
        .map(
          ([name, stats]) =>
            `<div class="kv"><span>${escapeHtml(name)}</span><strong>hit ${metric(stats.hit_rate)} · n=${metric(stats.examples)}</strong></div>`
        )
        .join("");
      return `
        <article class="timeline-row">
          <div class="timeline-header">
            <strong>${escapeHtml(job.dataset_name || "dataset")}</strong>
            <span class="status-badge ${statusClass(job.status)}">${escapeHtml(job.status)}</span>
          </div>
          <div class="metric-grid">
            <div class="metric-cell"><span>precision@k</span><strong>${metric(summary.precision_at_k)}</strong></div>
            <div class="metric-cell"><span>hit rate</span><strong>${metric(summary.hit_rate)}</strong></div>
            <div class="metric-cell"><span>recall@k</span><strong>${metric(summary.recall_at_k)}</strong></div>
            <div class="metric-cell"><span>MRR</span><strong>${metric(summary.mrr)}</strong></div>
            <div class="metric-cell"><span>faithfulness</span><strong>${metric(summary.faithfulness_score)}</strong></div>
            <div class="metric-cell"><span>examples</span><strong>${metric(summary.examples)}</strong></div>
            <div class="metric-cell"><span>abstention</span><strong>${metric(summary.abstention_accuracy)}</strong></div>
            <div class="metric-cell"><span>unsupported</span><strong>${metric(summary.unsupported_answer_rate)}</strong></div>
            <div class="metric-cell"><span>citation rel.</span><strong>${metric(summary.citation_relevance_rate)}</strong></div>
            <div class="metric-cell"><span>no-answer</span><strong>${metric(summary.no_answer_examples)}</strong></div>
          </div>
          ${filterRows ? `<div class="kv-grid"><div class="kv-label">by filter</div>${filterRows}</div>` : ""}
        </article>
      `;
    })
    .join("");
}

function appendMessage(role, body, citations = [], metrics = null, grounding = null) {
  const log = document.getElementById("chat-log");

  const emptyHint = document.getElementById("empty-hint");
  if (emptyHint) emptyHint.remove();

  const wrapper = document.createElement("article");
  wrapper.className = `message ${role}`;
  wrapper.innerHTML = `
    <div class="message-role">${role === "user" ? "You" : "Assistant"}</div>
    <div class="message-body">${escapeHtml(body)}</div>
    ${
      metrics
        ? `<div class="message-metrics">latency ${metrics.latency_ms}ms · retrieved ${metrics.retrieved_count}</div>`
        : ""
    }
    ${
      grounding
        ? `<div class="message-metrics ${grounding.grounded ? "grounded-ok" : "grounded-warn"}">
            ${grounding.grounded ? "grounded" : "not fully grounded"}
            ${grounding.usedCitationIds.length ? ` · used chunks ${grounding.usedCitationIds.map(escapeHtml).join(", ")}` : ""}
          </div>`
        : ""
    }
    ${
      grounding && grounding.warnings.length
        ? `<div class="message-warning">${grounding.warnings.map(escapeHtml).join(" ")}</div>`
        : ""
    }
    ${
      citations.length
        ? `<div class="citation-strip">${citations
            .map(
              (citation) => `
              <div class="citation">
                <strong>${escapeHtml(citation.title)}</strong>
                <div class="doc-meta">${escapeHtml(citation.document_id)} · final ${citation.final_score.toFixed(3)} · dense ${citation.dense_score.toFixed(3)} · lexical ${citation.lexical_score.toFixed(3)}</div>
                <div>${escapeHtml(citation.excerpt)}</div>
              </div>
            `
            )
            .join("")}</div>`
        : ""
    }
  `;
  log.appendChild(wrapper);
  wrapper.scrollIntoView({ behavior: "smooth", block: "end" });
}

async function refreshAll() {
  const [health, docs, jobs, evals] = await Promise.all([
    fetchJson("/health/ready"),
    fetchJson(`${apiBase}/v1/documents`),
    fetchJson(`${apiBase}/v1/jobs`),
    fetchJson(`${apiBase}/v1/evals/runs`),
  ]);
  renderStatus(health.status === "ok" ? "API ready" : "API degraded", health.status === "ok");
  state.documents = docs.documents;
  state.jobs = jobs;
  state.evals = evals;
  renderDocuments();
  renderJobs();
  renderEvals();
}

document.getElementById("file-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const files = document.getElementById("file-input").files;
  if (!files.length) return;
  const formData = new FormData();
  [...files].forEach((file) => formData.append("files", file));
  const metadataRaw = document.getElementById("metadata-json").value.trim();
  if (metadataRaw) {
    try {
      JSON.parse(metadataRaw);
    } catch (error) {
      renderStatus(`Invalid metadata JSON: ${error.message}`, false);
      return;
    }
    formData.append("metadata_json", metadataRaw);
  }
  await fetchJson(`${apiBase}/v1/documents/files`, { method: "POST", body: formData });
  await refreshAll();
});

document.getElementById("url-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const value = document.getElementById("url-input").value.trim();
  const mode = document.getElementById("url-mode").value;
  if (!value) return;
  const payload = mode === "sitemap" ? { sitemap_url: value } : { url: value };
  await fetchJson(`${apiBase}/v1/documents/urls`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  await refreshAll();
});

document.getElementById("query-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = document.getElementById("question-input").value.trim();
  if (!question) return;
  appendMessage("user", question);
  const response = await fetchJson(`${apiBase}/v1/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      category: document.getElementById("category-filter").value || null,
      document_ids: document
        .getElementById("document-filter")
        .value.split(",")
        .map((item) => item.trim())
        .filter(Boolean),
      include_trace: true,
    }),
  });
  appendMessage("assistant", response.answer, response.citations, response.metrics, {
    grounded: response.grounded,
    usedCitationIds: response.used_citation_ids || [],
    warnings: response.warnings || [],
  });
  if (!response.citations.length) {
    renderStatus("Query returned no citations", false);
  }
});

document.getElementById("eval-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const datasetName = document.getElementById("dataset-name").value.trim();
  await fetchJson(`${apiBase}/v1/evals/runs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dataset_name: datasetName }),
  });
  await refreshAll();
});

refreshAll().catch((error) => renderStatus(error.message, false));
window.setInterval(() => {
  refreshAll().catch((error) => renderStatus(error.message, false));
}, 5000);
