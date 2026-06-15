const folderInput = document.getElementById("folderInput");
const zipInput = document.getElementById("zipInput");
const labelInput = document.getElementById("labelInput");
const bcSelect = document.getElementById("bcSelect");
const basisInput = document.getElementById("basisInput");
const validateButton = document.getElementById("validateButton");
const selectionSummary = document.getElementById("selectionSummary");
const actionMessage = document.getElementById("actionMessage");
const healthBadge = document.getElementById("healthBadge");
const historyList = document.getElementById("historyList");
const refreshHistoryButton = document.getElementById("refreshHistoryButton");
const downloadLinks = document.getElementById("downloadLinks");
const resultsSummary = document.getElementById("resultsSummary");
const casesTableWrap = document.getElementById("casesTableWrap");
const casesTableBody = document.getElementById("casesTableBody");
const caseCards = document.getElementById("caseCards");

function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "—";
  }
  return Number(value).toPrecision(6).replace(/\.?0+$/, "");
}

function describeSelection() {
  if (zipInput.files.length > 0) {
    const file = zipInput.files[0];
    selectionSummary.textContent = `Zip bundle selected: ${file.name} (${file.size.toLocaleString()} bytes).`;
    return;
  }

  if (folderInput.files.length > 0) {
    const files = Array.from(folderInput.files);
    const sample = files[0]?.webkitRelativePath || files[0]?.name || "";
    selectionSummary.textContent = `Folder selection contains ${files.length} files. Root sample: ${sample}`;
    return;
  }

  selectionSummary.textContent = "No files selected yet.";
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = String(reader.result);
      const commaIndex = result.indexOf(",");
      resolve(commaIndex >= 0 ? result.slice(commaIndex + 1) : result);
    };
    reader.onerror = () => reject(reader.error || new Error(`Failed to read ${file.name}`));
    reader.readAsDataURL(file);
  });
}

async function buildPayload() {
  const bc_y = bcSelect.value;
  const max_basis_states = Number.parseInt(basisInput.value, 10);
  const label = labelInput.value.trim() || null;

  if (zipInput.files.length > 0) {
    const file = zipInput.files[0];
    return {
      label,
      bc_y,
      max_basis_states,
      files: [
        {
          path: file.name,
          content_base64: await fileToBase64(file),
        },
      ],
    };
  }

  if (folderInput.files.length > 0) {
    const files = await Promise.all(
      Array.from(folderInput.files).map(async (file) => ({
        path: file.webkitRelativePath || file.name,
        content_base64: await fileToBase64(file),
      })),
    );
    return { label, bc_y, max_basis_states, files };
  }

  throw new Error("Select a folder tree or a zip file first.");
}

function renderDownloads(downloads) {
  downloadLinks.innerHTML = "";
  Object.entries(downloads || {}).forEach(([key, href]) => {
    const anchor = document.createElement("a");
    anchor.className = "download-link";
    anchor.href = href;
    anchor.textContent = key.replaceAll("_", " ");
    downloadLinks.appendChild(anchor);
  });
}

function renderSummary(run) {
  resultsSummary.classList.remove("empty-state");
  resultsSummary.innerHTML = "";

  const cards = [
    ["Run ID", run.run_id],
    ["Created", run.created_at],
    ["Boundary", run.bc_y],
    ["Cases", String(run.case_count)],
  ];

  cards.forEach(([label, value]) => {
    const card = document.createElement("div");
    card.className = "summary-card";
    card.innerHTML = `<span>${label}</span><strong>${value}</strong>`;
    resultsSummary.appendChild(card);
  });
}

function renderCases(run) {
  const cases = run.cases || [];
  casesTableBody.innerHTML = "";
  caseCards.innerHTML = "";

  if (cases.length === 0) {
    casesTableWrap.classList.add("hidden");
    return;
  }

  casesTableWrap.classList.remove("hidden");
  cases.forEach((item) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${item.case}</td>
      <td>${item.benchmark_mode}</td>
      <td>${formatNumber(item.qmc_energy)}</td>
      <td>${formatNumber(item.ed_energy)}</td>
      <td>${formatNumber(item.ed_minus_qmc)}</td>
      <td>${item.green_status}</td>
      <td>${formatNumber(item.green_relative_frobenius_up)}</td>
      <td>${formatNumber(item.green_relative_frobenius_dn)}</td>
      <td>${formatNumber(item.green_trace_qmc_up)} / ${formatNumber(item.green_trace_ref_up)}</td>
      <td>${formatNumber(item.green_trace_qmc_dn)} / ${formatNumber(item.green_trace_ref_dn)}</td>
    `;
    casesTableBody.appendChild(row);

    const card = document.createElement("article");
    card.className = "case-card";
    const notes = Array.isArray(item.benchmark_notes) ? item.benchmark_notes : [];
    card.innerHTML = `
      <div class="case-card-head">
        <div>
          <strong>${item.case}</strong>
          <div class="muted">${item.lx}x${item.ly}, nup=${item.nup}, ndn=${item.ndn}, U=${formatNumber(item.u)}</div>
        </div>
        <span class="status-pill ${item.green_status === "compared" ? "ok" : "neutral"}">${item.green_status}</span>
      </div>
      <div class="pill-row">
        <span class="mini-pill">Mode: ${item.benchmark_mode}</span>
        <span class="mini-pill">QMC: ${formatNumber(item.qmc_energy)}</span>
        <span class="mini-pill">ED: ${formatNumber(item.ed_energy)}</span>
        <span class="mini-pill">relF up: ${formatNumber(item.green_relative_frobenius_up)}</span>
        <span class="mini-pill">relF dn: ${formatNumber(item.green_relative_frobenius_dn)}</span>
      </div>
      <p class="notes">${notes.length > 0 ? notes.join(" | ") : "No benchmark notes were emitted for this case."}</p>
    `;
    caseCards.appendChild(card);
  });
}

function renderRun(run) {
  renderSummary(run);
  renderDownloads(run.downloads);
  renderCases(run);
}

async function loadHistory() {
  historyList.innerHTML = `<p class="muted">Loading stored runs…</p>`;
  const response = await fetch("/api/runs");
  if (!response.ok) {
    historyList.innerHTML = `<p class="muted">Failed to load run history.</p>`;
    return;
  }

  const payload = await response.json();
  const runs = payload.runs || [];
  if (runs.length === 0) {
    historyList.innerHTML = `<p class="muted">No stored runs yet.</p>`;
    return;
  }

  historyList.innerHTML = "";
  let autoLoaded = false;
  runs.forEach((run) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "history-item";
    button.innerHTML = `
      <strong>${run.label || run.run_id}</strong>
      <span class="history-meta">${run.created_at}</span>
      <span class="history-meta">${run.case_count} case(s): ${run.case_names.join(", ")}</span>
    `;
    button.addEventListener("click", async () => {
      const detail = await fetch(`/api/runs/${run.run_id}`);
      if (!detail.ok) {
        actionMessage.textContent = `Failed to load run ${run.run_id}.`;
        return;
      }
      renderRun(await detail.json());
      actionMessage.textContent = `Loaded stored run ${run.run_id}.`;
    });
    historyList.appendChild(button);

    if (!autoLoaded && caseCards.children.length === 0) {
      autoLoaded = true;
      button.click();
    }
  });
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) {
      throw new Error("health check failed");
    }
    healthBadge.className = "status-pill ok";
    healthBadge.textContent = "Backend ready";
  } catch (error) {
    healthBadge.className = "status-pill error";
    healthBadge.textContent = "Backend unavailable";
  }
}

folderInput.addEventListener("change", () => {
  if (folderInput.files.length > 0) {
    zipInput.value = "";
  }
  describeSelection();
});

zipInput.addEventListener("change", () => {
  if (zipInput.files.length > 0) {
    folderInput.value = "";
  }
  describeSelection();
});

validateButton.addEventListener("click", async () => {
  validateButton.disabled = true;
  actionMessage.textContent = "Encoding files and sending the validation request…";

  try {
    const payload = await buildPayload();
    const response = await fetch("/api/validate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.detail || "Validation failed.");
    }
    renderRun(result);
    actionMessage.textContent = `Validation complete for ${result.case_count} case(s).`;
    await loadHistory();
  } catch (error) {
    actionMessage.textContent = error.message || "Validation failed.";
  } finally {
    validateButton.disabled = false;
  }
});

refreshHistoryButton.addEventListener("click", () => {
  loadHistory().catch((error) => {
    actionMessage.textContent = error.message || "Failed to refresh history.";
  });
});

checkHealth();
loadHistory();
describeSelection();
