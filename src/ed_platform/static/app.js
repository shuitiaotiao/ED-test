const folderInput = document.getElementById("folderInput");
const zipInput = document.getElementById("zipInput");
const labelInput = document.getElementById("labelInput");
const bcSelect = document.getElementById("bcSelect");
const basisInput = document.getElementById("basisInput");
const validateButton = document.getElementById("validateButton");
const selectionSummary = document.getElementById("selectionSummary");
const actionMessage = document.getElementById("actionMessage");
const healthBadge = document.getElementById("healthBadge");
const energyRuleValue = document.getElementById("energyRuleValue");
const greenRuleValue = document.getElementById("greenRuleValue");
const resultEmpty = document.getElementById("resultEmpty");
const resultContent = document.getElementById("resultContent");
const overallStatusBadge = document.getElementById("overallStatusBadge");
const overallHeadline = document.getElementById("overallHeadline");
const overallText = document.getElementById("overallText");
const scoreCases = document.getElementById("scoreCases");
const scorePassed = document.getElementById("scorePassed");
const scoreFailed = document.getElementById("scoreFailed");
const scoreEnergySkipped = document.getElementById("scoreEnergySkipped");
const energySummary = document.getElementById("energySummary");
const greenSummary = document.getElementById("greenSummary");
const updatedSummary = document.getElementById("updatedSummary");
const failedCases = document.getElementById("failedCases");
const passedCases = document.getElementById("passedCases");

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  return Number(value).toPrecision(6).replace(/\.?0+$/, "");
}

function formatTimestamp(value) {
  if (!value) {
    return "-";
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return parsed.toLocaleString("zh-CN", { hour12: false });
}

function localizeStatus(status) {
  if (status === "pass") {
    return "\u901a\u8fc7";
  }
  if (status === "fail") {
    return "\u4e0d\u901a\u8fc7";
  }
  if (status === "skipped") {
    return "\u8df3\u8fc7";
  }
  return status || "-";
}

function localizeCheckName(name) {
  if (name === "Green function") {
    return "Green \u51fd\u6570";
  }
  return name || "-";
}

function setRuleChips(rules) {
  if (!rules) {
    energyRuleValue.textContent = "Energy \u9608\u503c\u4e0d\u53ef\u7528";
    greenRuleValue.textContent = "Green \u9608\u503c\u4e0d\u53ef\u7528";
    return;
  }
  energyRuleValue.textContent = `|ED - QMC| <= ${formatNumber(rules.energy_abs_tolerance)}`;
  greenRuleValue.textContent = `max(relF) <= ${formatNumber(rules.green_relative_tolerance)}`;
}

function describeSelection() {
  if (zipInput.files.length > 0) {
    const file = zipInput.files[0];
    selectionSummary.textContent = `\u5df2\u9009\u62e9 zip \u538b\u7f29\u5305: ${file.name}\uff08${file.size.toLocaleString()} bytes\uff09\u3002`;
    return;
  }

  if (folderInput.files.length > 0) {
    const files = Array.from(folderInput.files);
    const rootSample = files[0]?.webkitRelativePath || files[0]?.name || "";
    selectionSummary.textContent = `\u5df2\u9009\u62e9\u6587\u4ef6\u5939\uff0c\u5171 ${files.length} \u4e2a\u6587\u4ef6\u3002\u76ee\u5f55\u793a\u4f8b: ${rootSample}`;
    return;
  }

  selectionSummary.textContent = "\u8fd8\u6ca1\u6709\u9009\u62e9\u4efb\u4f55\u6587\u4ef6\u3002";
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
      files: [{ path: file.name, content_base64: await fileToBase64(file) }],
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

  throw new Error("\u8bf7\u5148\u9009\u62e9 case \u6587\u4ef6\u5939\u6216 zip \u538b\u7f29\u5305\u3002");
}

function badgeClass(status, prefix) {
  return `${prefix} ${status}`;
}

function renderCheck(check) {
  const metricText =
    check.metric !== null && check.metric !== undefined && check.tolerance !== null && check.tolerance !== undefined
      ? ` \u6307\u6807 ${formatNumber(check.metric)}\uff0c\u9608\u503c ${formatNumber(check.tolerance)}\u3002`
      : "";
  const referenceText = check.reference_source ? ` \u53c2\u8003\u6765\u6e90: ${escapeHtml(check.reference_source)}\u3002` : "";
  return `
    <div class="check-row ${escapeHtml(check.status)}">
      <div class="check-top">
        <strong>${escapeHtml(localizeCheckName(check.name))}</strong>
        <span class="${badgeClass(escapeHtml(check.status), "check-pill")}">${escapeHtml(localizeStatus(check.status))}</span>
      </div>
      <p>${escapeHtml(check.summary)}${metricText}${referenceText}</p>
    </div>
  `;
}

function renderFailureCase(caseItem) {
  const notes = Array.isArray(caseItem.notes) ? caseItem.notes : [];
  const notesHtml =
    notes.length > 0
      ? `<div class="note-list">${notes.map((note) => `<div class="note-item">${escapeHtml(note)}</div>`).join("")}</div>`
      : "";
  return `
    <article class="failure-card">
      <div class="case-head">
        <div>
          <strong>${escapeHtml(caseItem.case)}</strong>
          <div class="case-meta">${escapeHtml(caseItem.lattice)}</div>
        </div>
        <span class="${badgeClass(escapeHtml(caseItem.status), "overall-badge")}">${escapeHtml(localizeStatus(caseItem.status))}</span>
      </div>
      <p class="case-meta">${escapeHtml(caseItem.headline)}</p>
      <div class="check-list">${caseItem.checks.map(renderCheck).join("")}</div>
      ${notesHtml}
    </article>
  `;
}

function renderPassCase(caseItem) {
  return `
    <article class="pass-card">
      <div class="case-head">
        <div>
          <strong>${escapeHtml(caseItem.case)}</strong>
          <div class="case-meta">${escapeHtml(caseItem.lattice)}</div>
        </div>
        <span class="${badgeClass(escapeHtml(caseItem.status), "overall-badge")}">${escapeHtml(localizeStatus(caseItem.status))}</span>
      </div>
      <p class="case-meta">${escapeHtml(caseItem.headline)}</p>
      <div class="check-list">${caseItem.checks.map(renderCheck).join("")}</div>
    </article>
  `;
}

function renderResults(payload) {
  resultEmpty.classList.add("hidden");
  resultContent.classList.remove("hidden");

  const overallFail = payload.overall_status === "fail";
  overallStatusBadge.className = badgeClass(payload.overall_status, "overall-badge");
  overallStatusBadge.textContent = overallFail ? "\u4e0d\u901a\u8fc7" : "\u901a\u8fc7";
  overallHeadline.textContent = overallFail
    ? "\u5b58\u5728 case \u672a\u901a\u8fc7\u5f53\u524d\u6821\u9a8c\u89c4\u5219\u3002"
    : "\u672c\u6b21\u68c0\u6d4b\u4e2d\u7684\u6240\u6709 case \u5747\u901a\u8fc7\u5f53\u524d\u6821\u9a8c\u89c4\u5219\u3002";
  overallText.textContent = overallFail
    ? "\u8bf7\u5728\u4e0b\u65b9\u5931\u8d25\u8fd4\u56de\u533a\u67e5\u770b\u6bcf\u4e2a\u5931\u8d25 case \u7684\u963b\u585e\u9879\u548c\u5177\u4f53\u539f\u56e0\u3002"
    : "\u5f53\u524d Energy \u4e0e Green \u6821\u9a8c\u4e2d\u6ca1\u6709\u53d1\u73b0\u963b\u585e\u6027\u4e0d\u4e00\u81f4\u3002";

  scoreCases.textContent = String(payload.case_count);
  scorePassed.textContent = String(payload.summary.passed_cases);
  scoreFailed.textContent = String(payload.summary.failed_cases);
  scoreEnergySkipped.textContent = String(payload.summary.energy.skipped);
  energySummary.textContent = `\u901a\u8fc7 ${payload.summary.energy.pass} / \u5931\u8d25 ${payload.summary.energy.fail} / \u8df3\u8fc7 ${payload.summary.energy.skipped}`;
  greenSummary.textContent = `\u901a\u8fc7 ${payload.summary.green.pass} / \u5931\u8d25 ${payload.summary.green.fail} / \u8df3\u8fc7 ${payload.summary.green.skipped}`;
  updatedSummary.textContent = formatTimestamp(payload.created_at);
  setRuleChips(payload.rules);

  const failures = payload.failed_cases || [];
  const passes = (payload.cases || []).filter((caseItem) => caseItem.status === "pass");

  failedCases.innerHTML =
    failures.length > 0
      ? failures.map(renderFailureCase).join("")
      : '<div class="pass-card">\u672c\u6b21\u8fd0\u884c\u6ca1\u6709\u8fd4\u56de\u5931\u8d25 case\u3002</div>';
  passedCases.innerHTML =
    passes.length > 0
      ? passes.map(renderPassCase).join("")
      : '<div class="failure-card">\u672c\u6b21\u8fd0\u884c\u6ca1\u6709\u8fd4\u56de\u901a\u8fc7 case\u3002</div>';
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) {
      throw new Error("health check failed");
    }
    healthBadge.className = "status-pill ok";
    healthBadge.textContent = "\u540e\u7aef\u5df2\u5c31\u7eea";
  } catch (error) {
    healthBadge.className = "status-pill error";
    healthBadge.textContent = "\u540e\u7aef\u4e0d\u53ef\u7528";
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
  actionMessage.textContent = "\u6b63\u5728\u8bfb\u53d6\u4e0a\u4f20\u5185\u5bb9\u5e76\u6267\u884c\u6821\u9a8c...";

  try {
    const payload = await buildPayload();
    const response = await fetch("/api/validate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.detail || "\u6821\u9a8c\u5931\u8d25\u3002");
    }
    renderResults(result);
    actionMessage.textContent =
      result.overall_status === "fail"
        ? `\u6821\u9a8c\u5b8c\u6210\uff0c\u8fd4\u56de ${result.summary.failed_cases} \u4e2a\u672a\u901a\u8fc7 case\u3002`
        : `\u6821\u9a8c\u5b8c\u6210\uff0c${result.case_count} \u4e2a case \u5168\u90e8\u901a\u8fc7\u3002`;
    resultContent.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    actionMessage.textContent = error.message || "\u6821\u9a8c\u5931\u8d25\u3002";
  } finally {
    validateButton.disabled = false;
  }
});

checkHealth();
setRuleChips({
  energy_abs_tolerance: 1.0e-3,
  green_relative_tolerance: 1.0e-1,
});
describeSelection();
