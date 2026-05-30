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

function safeJson(value) {
  return escapeHtml(JSON.stringify(value ?? {}, null, 2));
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed: ${response.status}`);
  }
  return response.json();
}

function renderStatus(message, ok = true) {
  const node = document.getElementById("api-status");
  node.textContent = message;
  node.style.color = ok ? "var(--accent)" : "var(--danger)";
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

function renderJobs() {
  const list = document.getElementById("job-list");
  const jobs = state.jobs.filter((item) => item.job_type === "ingestion");
  if (!jobs.length) {
    list.innerHTML = '<p class="empty-state">No ingestion jobs yet.</p>';
    return;
  }
  list.innerHTML = jobs
    .map(
      (job) => `
        <article class="timeline-row">
          <div class="timeline-header">
            <strong>${escapeHtml(job.status)}</strong>
            <span class="pill">${escapeHtml(job.job_type)}</span>
          </div>
          <div class="metrics">progress ${job.progress ?? 0}% · attempts ${job.attempts ?? 0}/${job.max_attempts ?? 3}</div>
          <div class="timeline-meta">${escapeHtml(job.id)}</div>
          ${job.error_message ? `<div class="timeline-meta">${escapeHtml(job.error_code || "error")}: ${escapeHtml(job.error_message)}</div>` : ""}
          <pre class="timeline-meta">${safeJson(job.result)}</pre>
        </article>
      `
    )
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
      const summary = job.result.summary || {};
      return `
        <article class="timeline-row">
          <div class="timeline-header">
            <strong>${escapeHtml(job.dataset_name || "dataset")}</strong>
            <span class="pill">${escapeHtml(job.status)}</span>
          </div>
          <div class="metrics">
            precision@k ${summary.precision_at_k ?? "-"} · hit rate ${summary.hit_rate ?? "-"} · faithfulness ${summary.faithfulness_score ?? "-"}
          </div>
          <pre class="timeline-meta">${safeJson(summary.by_filter || {})}</pre>
        </article>
      `;
    })
    .join("");
}

function appendMessage(role, body, citations = [], metrics = null) {
  const log = document.getElementById("chat-log");
  const wrapper = document.createElement("article");
  wrapper.className = `message ${role}`;
  wrapper.innerHTML = `
    <div class="section-kicker">${role === "user" ? "Prompt" : "Answer"}</div>
    <div class="message-body">${escapeHtml(body)}</div>
    ${
      metrics
        ? `<div class="metrics">latency ${metrics.latency_ms}ms · retrieved ${metrics.retrieved_count}</div>`
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
  log.prepend(wrapper);
}

async function refreshAll() {
  const [health, docs, jobs, evals] = await Promise.all([
    fetchJson(`${apiBase.replace(/\/$/, "")}/health/ready`),
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
  appendMessage("assistant", response.answer, response.citations, response.metrics);
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
