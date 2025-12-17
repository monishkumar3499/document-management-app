const apiBase = "http://127.0.0.1:8000/api/v1";
const departments = [
  "HR",
  "Finance",
  "Operations",
  "Engineering",
  "Safety & Compliance",
];

let cachedDocs = null;
let sectionHistory = [];
let currentSlides = [];
let currentIndex = 0;
let currentTab = "content";
let currentTablePage = 1;
const tablePageSize = 10;

// ---------------- HELPERS ----------------

// Safely escape text to avoid HTML issues
function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

// Normalize different table shapes into a 2D array of cells
// Supports:
//  - { rows: [...] }
//  - [ ["col1","col2"], ... ]
//  - [ "col1 | col2", "val1 | val2", ... ]
function normalizeTable(rawTable) {
  if (!rawTable) return [[]];

  // Case 1: object with .rows
  if (Array.isArray(rawTable.rows)) {
    return rawTable.rows;
  }

  // Case 2: array of rows
  if (Array.isArray(rawTable)) {
    // If rows are strings like "Field | Value"
    if (rawTable.length && typeof rawTable[0] === "string") {
      return rawTable.map((line) => line.split("|").map((col) => col.trim()));
    }

    // Already array-of-arrays
    if (rawTable.length && Array.isArray(rawTable[0])) {
      return rawTable;
    }
  }

  // Fallback: wrap into single row, single cell
  return [[String(rawTable)]];
}

// ---------------- SECTIONS ----------------
function showSection(sectionId) {
  // Update active nav item
  document
    .querySelectorAll(".nav-item")
    .forEach((item) => item.classList.remove("active"));

  const current = document.querySelector("section:not(.hidden)");
  if (current && current.id !== sectionId && sectionId !== "home") {
    sectionHistory.push(current.id);
  }
  document
    .querySelectorAll("section")
    .forEach((s) => s.classList.add("hidden"));
  document.getElementById(sectionId).classList.remove("hidden");
  document.getElementById("backBtn").style.display =
    sectionHistory.length > 0 && sectionId !== "home" ? "block" : "none";

  // Add active class to current nav item
  if (sectionId === "home") {
    document.getElementById("homeBtn").classList.add("active");
  } else if (sectionId === "upload") {
    document.getElementById("uploadBtn").classList.add("active");
  }

  if (window.innerWidth < 768) {
    document.getElementById("sidebar").classList.add("sidebar-hidden");
  }
}

function goBack() {
  const prev = sectionHistory.pop();
  if (prev) {
    showSection(prev);
  }
}

// ---------------- DEPARTMENT DROPDOWN ----------------
function loadDepartmentsDropdown() {
  const deptList = document.getElementById("departmentsList");
  deptList.innerHTML = "";
  departments.forEach((d) => {
    const li = document.createElement("li");
    li.innerHTML = `<button type="button" onclick="loadDepartmentDocs('${d}')" class="w-full text-left px-4 py-2 hover:bg-white hover:bg-opacity-10 rounded text-white hover:text-white transition text-sm" aria-label="Select ${d} Department">${d}</button>`;
    deptList.appendChild(li);
  });
}

function toggleDeptDropdown(e) {
  e.stopPropagation();
  const deptList = document.getElementById("departmentsList");
  deptList.classList.toggle("hidden");
}

window.addEventListener("click", function (e) {
  const deptList = document.getElementById("departmentsList");
  const button = deptList.previousElementSibling;
  if (!deptList.contains(e.target) && !button.contains(e.target)) {
    deptList.classList.add("hidden");
  }
});

// ---------------- MOBILE MENU ----------------
document.getElementById("menuToggle").addEventListener("click", () => {
  document.getElementById("sidebar").classList.toggle("sidebar-hidden");
});

window.addEventListener("resize", () => {
  if (window.innerWidth >= 768) {
    document.getElementById("sidebar").classList.remove("sidebar-hidden");
  }
});

// ---------------- HOME DOCS ----------------
async function loadHomeDocs() {
  showSection("home");
  const container = document.getElementById("docList");
  container.innerHTML = `<div class="col-span-full text-center py-8 text-gray-500 flex items-center justify-center gap-2"><div class="spinner"></div>Loading...</div>`;

  try {
    const res = await fetch(`${apiBase}/documents-list`);
    cachedDocs = await res.json();

    cachedDocs.forEach((doc) => {
      if (!doc.departments) doc.departments = {};
      if (doc.departments_approval) {
        Object.keys(doc.departments_approval).forEach(
          (d) => (doc.departments[d] = doc.departments_approval[d])
        );
      }
      const assignedDepartments = Object.keys(doc.departments);
      doc.approved =
        assignedDepartments.length > 0 &&
        assignedDepartments.every((d) => doc.departments[d] === true);
    });

    const high = cachedDocs.filter((d) => d.priority === "High").length;
    const low = cachedDocs.filter((d) => d.priority === "Low").length;
    const total = cachedDocs.length;
    const pending = cachedDocs.filter((d) => !d.approved).length;
    const approved = cachedDocs.filter((d) => d.approved).length;

    renderDashboard("homeDashboard", total, high, low, pending, approved);
    renderDocs(cachedDocs, container);
  } catch (err) {
    console.error(err);
    container.innerHTML =
      "<div class='text-red-600 font-medium text-center py-8'>Failed to load documents</div>";
  }
}

function renderDocs(docs, container) {
  container.innerHTML = "";
  if (docs.length === 0) {
    container.innerHTML = `<div class="col-span-full text-center py-8 text-gray-500">No documents available.</div>`;
    return;
  }

  docs.forEach((doc) => {
    const div = document.createElement("div");
    div.className =
      "doc-card bg-white p-6 rounded-xl shadow-sm transition flex flex-col";
    const assignedDepartments = Object.keys(doc.departments);
    const overallApproved =
      assignedDepartments.length > 0 &&
      assignedDepartments.every((d) => doc.departments[d] === true)
        ? "Approved"
        : "Pending";

    div.innerHTML = `
      <div class="flex flex-col">
        <h3 class="font-semibold text-lg text-gray-800 truncate" title="${
          doc.filename
        }">${doc.filename}</h3>
        <div class="flex items-center gap-2 mt-2">
          <span class="text-sm text-gray-500">${new Date(
            doc.created_at
          ).toLocaleDateString()}</span>
          <span class="${
            doc.priority === "High" ? "priority-high" : "priority-low"
          }">
            ${doc.priority}
          </span>
        </div>
        <span class="inline-block mt-2 px-3 py-1 text-sm rounded-full ${
          overallApproved === "Approved"
            ? "bg-green-100 text-green-700"
            : "bg-yellow-100 text-yellow-700"
        }">
          ${overallApproved}
        </span>
      </div>
      <div class="mt-4 flex flex-wrap gap-2">
        ${Object.keys(doc.departments)
          .map(
            (d) => `
              <button
                type="button"
                onclick="viewDepartment('${doc.id}', '${d}')"
                class="px-3 py-1 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-full text-sm transition"
                aria-label="View ${doc.filename} in ${d}"
              >
                ${d}
              </button>
            `
          )
          .join("")}
      </div>
    `;
    container.appendChild(div);
  });
}

// ---------------- TABLE FUNCTIONS ----------------
function renderDocumentsTable() {
  if (!cachedDocs) return;

  const tableBody = document.getElementById("documentsTableBody");
  const startIndex = (currentTablePage - 1) * tablePageSize;
  const endIndex = startIndex + tablePageSize;
  const paginatedDocs = cachedDocs.slice(startIndex, endIndex);

  tableBody.innerHTML = "";

  paginatedDocs.forEach((doc) => {
    const row = document.createElement("tr");
    const assignedDepts = Object.keys(doc.departments || {});
    const deptText =
      assignedDepts.length > 0 ? assignedDepts.join(", ") : "Unassigned";

    row.innerHTML = `
      <td class="font-medium text-gray-900">${doc.filename}</td>
      <td>${deptText}</td>
      <td>System User</td>
      <td>${new Date(doc.created_at).toLocaleDateString()}</td>
      <td>
        <span class="${
          doc.priority === "High" ? "priority-high" : "priority-low"
        }">
          ${doc.priority}
        </span>
      </td>
      <td>
        <div class="flex gap-2">
          <button onclick="viewFullDocument('${
            doc.id
          }')" class="btn-secondary text-xs">
            <i class="fa fa-eye"></i>
          </button>
          <a href="${apiBase}/documents/${
      doc.id
    }/download-original" class="btn-secondary text-xs" target="_blank">
            <i class="fa fa-download"></i>
          </a>
        </div>
      </td>
    `;
    tableBody.appendChild(row);
  });

  // Update pagination controls
  const totalPages = Math.ceil(cachedDocs.length / tablePageSize);
  document.getElementById(
    "tablePageInfo"
  ).textContent = `Page ${currentTablePage} of ${totalPages}`;
  document.getElementById("prevTableBtn").disabled = currentTablePage === 1;
  document.getElementById("nextTableBtn").disabled =
    currentTablePage === totalPages;
}

function nextTablePage() {
  const totalPages = Math.ceil(cachedDocs.length / tablePageSize);
  if (currentTablePage < totalPages) {
    currentTablePage++;
    renderDocumentsTable();
  }
}

function prevTablePage() {
  if (currentTablePage > 1) {
    currentTablePage--;
    renderDocumentsTable();
  }
}

// ---------------- LOAD DEPARTMENT DOCUMENTS ----------------
async function loadDepartmentDocs(dept) {
  showSection("department");
  document.getElementById("deptTitle").innerHTML = dept;

  const container = document.getElementById("deptDocs");
  container.innerHTML = `
    <div class="text-gray-500 py-8 text-center flex items-center justify-center gap-2">
      <div class="spinner"></div>Loading documents for ${dept}...
    </div>
  `;

  try {
    if (!cachedDocs) {
      const res = await fetch(`${apiBase}/documents-list`);
      cachedDocs = await res.json();
    }

    // Only documents assigned to this department
    const filteredDocs = cachedDocs.filter(
      (doc) => doc.departments && doc.departments.hasOwnProperty(dept)
    );

    container.innerHTML = "";

    for (const doc of filteredDocs) {
      try {
        const resDept = await fetch(`${apiBase}/departments/${doc.id}/${dept}`);
        const deptData = await resDept.json();
        if (!deptData || deptData.length === 0) continue;

        doc.departments[dept] = deptData.every((p) => p.approved);
        doc.approved = Object.values(doc.departments).every((v) => v === true);

        const combinedContent = deptData.map((p) => p.content).join("\n\n");
        const deptApproved = doc.departments[dept];

        const details = document.createElement("details");
        details.className =
          "doc-card bg-white p-6 rounded-xl shadow-sm transition";
        details.innerHTML = `
          <summary class="font-semibold text-lg text-gray-800 cursor-pointer flex items-center justify-between">
            <span class="truncate" title="${doc.filename}">${
          doc.filename
        }</span>
            <i class="fa fa-chevron-down text-gray-500"></i>
          </summary>
          <div class="mt-4">
            <div class="flex items-center gap-2 mb-2">
              <span class="text-sm text-gray-500">${new Date(
                doc.created_at
              ).toLocaleDateString()}</span>
              <span class="${
                doc.priority === "High" ? "priority-high" : "priority-low"
              }">
                ${doc.priority}
              </span>
            </div>
            <p class="text-gray-600 text-sm">${
              combinedContent.substring(0, 200) +
              (combinedContent.length > 200 ? "..." : "")
            }</p>
            <div class="mt-4 flex gap-2">
              <button type="button" onclick="viewFullDocumentDept('${
                doc.id
              }', '${dept}')" class="btn-primary">
                <i class="fa fa-eye"></i> View Full
              </button>
              <button type="button" class="btn-secondary approve-btn ${
                deptApproved
                  ? "bg-yellow-100 text-yellow-800 border-yellow-300"
                  : ""
              }" data-doc="${doc.id}" data-dept="${dept}">
                <i class="fa fa-${deptApproved ? "edit" : "check"}"></i>
                ${deptApproved ? "Review" : "Approve"}
              </button>
            </div>
          </div>
        `;
        container.appendChild(details);

        details.querySelector(".approve-btn").onclick = (e) => {
          approveDepartment(doc.id, dept, e.currentTarget);
        };
      } catch (err) {
        console.error(
          `Failed to load department content for doc ${doc.filename}:`,
          err
        );
      }
    }

    // Update department dashboard
    const total = filteredDocs.length;
    const high = filteredDocs.filter((d) => d.priority === "High").length;
    const low = filteredDocs.filter((d) => d.priority === "Low").length;
    const approved = filteredDocs.filter(
      (d) => d.departments[dept] === true
    ).length;
    const pending = total - approved;

    renderDashboard("deptDashboard", total, high, low, pending, approved);

    if (filteredDocs.length === 0) {
      container.innerHTML = `<div class="text-gray-500 text-center py-8">No documents for this department</div>`;
      renderDashboard("deptDashboard", 0, 0, 0, 0, 0);
    }
  } catch (err) {
    console.error(err);
    container.innerHTML =
      "<div class='text-red-600 font-medium text-center py-8'>Failed to load department documents</div>";
    renderDashboard("deptDashboard", 0, 0, 0, 0, 0);
  }
}

// ---------------- VIEW DOCUMENT MODAL ----------------
async function viewFullDocument(docId) {
  const modal = document.getElementById("docModal");
  const titleEl = document.getElementById("modalTitle");
  modal.classList.remove("hidden");
  titleEl.innerText = "Loading...";

  currentSlides = [];
  currentIndex = 0;

  try {
    const doc = cachedDocs.find((d) => d.id === docId);
    if (!doc) throw new Error("Document not found");

    // Load content from all departments for this document
    const allContent = [];
    const allTables = [];

    for (const dept of Object.keys(doc.departments || {})) {
      try {
        const resDept = await fetch(`${apiBase}/departments/${docId}/${dept}`);
        const deptData = await resDept.json();
        allContent.push(...deptData);
      } catch (err) {
        console.error(`Failed to load ${dept} content:`, err);
      }
    }

    // Load tables
    try {
      const resTables = await fetch(`${apiBase}/documents/${docId}/ocr-tables`);
      const tableData = await resTables.json();
      allTables.push(...(tableData.pages || []));
    } catch (err) {
      console.error("Failed to load tables:", err);
    }

    // Organize by pages
    const pagesMap = {};
    allContent.forEach((page) => {
      const pageNum = page.page_start;
      if (!pagesMap[pageNum]) {
        pagesMap[pageNum] = { content: page.content || "", tables: [] };
      } else {
        pagesMap[pageNum].content += "\n\n" + page.content;
      }
    });

    allTables.forEach((page) => {
      if (page.tables && page.tables.length > 0) {
        if (!pagesMap[page.page_number]) {
          pagesMap[page.page_number] = { content: "", tables: [] };
        }
        page.tables.forEach((table) => {
          pagesMap[page.page_number].tables.push(table);
        });
      }
    });

    currentSlides = Object.keys(pagesMap)
      .map((pn) => ({
        page_start: parseInt(pn),
        content: pagesMap[pn].content,
        tables: pagesMap[pn].tables,
      }))
      .sort((a, b) => a.page_start - b.page_start);

    titleEl.innerText = doc.filename;
    const downloadLink = document.getElementById("downloadLink");
    const downloadOriginalLink = document.getElementById(
      "downloadOriginalLink"
    );

    downloadOriginalLink.href = `${apiBase}/documents/${docId}/download-original`;
    downloadOriginalLink.style.display = "block";
    downloadLink.style.display = "none"; // Hide processed PDF link for full document view

    renderSlide(0);
  } catch (err) {
    console.error(err);
    titleEl.innerText = "Error loading content";
    document.getElementById("tabContent").innerHTML =
      "<p class='text-gray-600'>Failed to load content.</p>";
    document.getElementById("tabTables").innerHTML =
      "<p class='text-gray-600'>Failed to load tables.</p>";
  }
}

async function viewFullDocumentDept(docId, dept) {
  const modal = document.getElementById("docModal");
  const titleEl = document.getElementById("modalTitle");
  modal.classList.remove("hidden");
  titleEl.innerText = "Loading...";

  currentSlides = [];
  currentIndex = 0;

  try {
    const resDept = await fetch(`${apiBase}/departments/${docId}/${dept}`);
    const deptData = await resDept.json();

    const resTables = await fetch(
      `${apiBase}/departments/${docId}/${dept}/tables`
    );
    const tableData = await resTables.json();

    const pagesMap = {};

    // ---- 1) Build pagesMap only from THIS department's paragraphs ----
    deptData.forEach((page) => {
      const pageNum = page.page_start;
      if (!pagesMap[pageNum]) {
        pagesMap[pageNum] = { content: page.content || "", tables: [] };
      } else {
        pagesMap[pageNum].content += "\n\n" + (page.content || "");
      }
    });

    // ---- 2) Attach tables ONLY to pages that already have this dept's content ----
    if (tableData && Array.isArray(tableData.pages)) {
      tableData.pages.forEach((page) => {
        const pageNum = page.page_number;
        let pageEntry = pagesMap[pageNum];

        // If this dept has NO content on that page, still create a slide for tables.
        if (!pageEntry) {
          pageEntry = pagesMap[pageNum] = { content: "", tables: [] };
        }

        if (page.tables && page.tables.length > 0) {
          page.tables.forEach((table) => {
            // Ensure a normalized shape: { rows: [...] }
            if (Array.isArray(table.rows)) {
              pageEntry.tables.push({ rows: table.rows });
            } else if (Array.isArray(table)) {
              // In case backend sends plain 2D array
              pageEntry.tables.push({ rows: table });
            }
          });
        }
      });
    }

    currentSlides = Object.keys(pagesMap)
      .map((pn) => ({
        page_start: parseInt(pn),
        content: pagesMap[pn].content,
        tables: pagesMap[pn].tables || [],
      }))
      .sort((a, b) => a.page_start - b.page_start);

    titleEl.innerText = `${dept} Content`;

    const downloadLink = document.getElementById("downloadLink");
    const downloadOriginalLink = document.getElementById(
      "downloadOriginalLink"
    );

    downloadLink.href = `${apiBase}/departments/${docId}/${dept}/download`;
    downloadLink.style.display = "block";

    downloadOriginalLink.href = `${apiBase}/documents/${docId}/download-original`;
    downloadOriginalLink.style.display = "block";

    renderSlide(0);
  } catch (err) {
    console.error(err);
    titleEl.innerText = "Error loading content";
    document.getElementById("tabContent").innerHTML =
      "<p class='text-gray-600'>Failed to load content.</p>";
    document.getElementById("tabTables").innerHTML =
      "<p class='text-gray-600'>Failed to load tables.</p>";
  }
}

// ---------------- SLIDE RENDERING (CONTENT + TABLES) ----------------
function renderSlide(index) {
  currentIndex = index;

  const slide = currentSlides[index];
  const contentTab = document.getElementById("tabContent");
  const tablesTab = document.getElementById("tabTables");

  contentTab.innerHTML = "";
  tablesTab.innerHTML = "";

  if (!slide) {
    contentTab.innerHTML =
      "<p class='text-gray-600 text-sm'>No content available</p>";
    tablesTab.innerHTML =
      "<p class='text-gray-600 text-sm'>No tables on this page</p>";
    document.getElementById("slideIndicator").innerText = "";
    document.getElementById("prevBtn").disabled = true;
    document.getElementById("nextBtn").disabled = true;
    return;
  }

  // -------- Content tab: page text ----------
  if (slide.content && slide.content.trim().length > 0) {
    const div = document.createElement("div");
    div.className = "text-gray-700 whitespace-pre-wrap text-sm leading-relaxed";
    div.innerText = slide.content;
    contentTab.appendChild(div);

    const meta = document.createElement("div");
    meta.className = "mt-4 text-xs text-gray-500 flex justify-between";
    meta.innerHTML = `
      <span>Page ${slide.page_start ?? "?"}</span>
      <span>Total pages: ${currentSlides.length}</span>
    `;
    contentTab.appendChild(meta);
  } else {
    contentTab.innerHTML =
      "<p class='text-gray-600 text-sm'>No content available</p>";
  }

  // -------- Tables tab: structured tables ----------
  if (slide.tables && slide.tables.length > 0) {
    slide.tables.forEach((rawTable, idx) => {
      const rows = normalizeTable(rawTable);

      const wrapper = document.createElement("div");
      wrapper.className =
        "mb-4 border border-gray-200 rounded-lg overflow-x-auto";

      const header = document.createElement("div");
      header.className =
        "px-4 py-2 text-xs text-gray-600 bg-gray-50 border-b flex justify-between";
      header.innerHTML = `
        <span>Table ${idx + 1}</span>
        <span>Page ${slide.page_start ?? "?"}</span>
      `;
      wrapper.appendChild(header);

      const table = document.createElement("table");
      table.className = "min-w-full text-xs text-left border-collapse";

      rows.forEach((row, ri) => {
        const tr = document.createElement("tr");
        tr.className =
          ri === 0 ? "bg-gray-50" : ri % 2 === 0 ? "bg-white" : "bg-gray-50";
        row.forEach((cell) => {
          const cellEl = document.createElement(ri === 0 ? "th" : "td");
          cellEl.className = "border border-gray-200 px-3 py-2 align-top";
          cellEl.innerText = cell;
          tr.appendChild(cellEl);
        });
        table.appendChild(tr);
      });

      wrapper.appendChild(table);
      tablesTab.appendChild(wrapper);
    });
  } else {
    tablesTab.innerHTML =
      "<p class='text-gray-600 text-sm'>No tables on this page</p>";
  }

  // -------- Controls ----------
  document.getElementById("slideIndicator").innerText = `Page ${
    slide.page_start ?? index + 1
  } (${index + 1} / ${currentSlides.length})`;
  document.getElementById("prevBtn").disabled = index === 0;
  document.getElementById("nextBtn").disabled =
    index === currentSlides.length - 1;

  // Respect currentTab (content/tables)
  switchTab(currentTab);
}

function showNextSlide() {
  if (currentIndex < currentSlides.length - 1) {
    currentIndex++;
    renderSlide(currentIndex);
  }
}

function showPrevSlide() {
  if (currentIndex > 0) {
    currentIndex--;
    renderSlide(currentIndex);
  }
}

// ---------------- MODAL CLOSE ----------------
function closeModal() {
  const modal = document.getElementById("docModal");
  const titleEl = document.getElementById("modalTitle");
  const contentTab = document.getElementById("tabContent");
  const tablesTab = document.getElementById("tabTables");
  const downloadLink = document.getElementById("downloadLink");
  const downloadOriginalLink = document.getElementById("downloadOriginalLink");

  modal.classList.add("hidden");
  titleEl.innerText = "";
  contentTab.innerHTML = "";
  tablesTab.innerHTML = "";
  downloadLink.style.display = "none";
  downloadLink.href = "#";
  downloadOriginalLink.style.display = "none";
  downloadOriginalLink.href = "#";
  currentSlides = [];
  currentIndex = 0;
}

// ---------------- APPROVE / REVIEW FUNCTION ----------------
async function approveDepartment(docId, dept, btn) {
  if (!cachedDocs) return;
  const doc = cachedDocs.find((d) => d.id === docId);
  if (!doc) return;

  const currentState = doc.departments[dept] || false;
  const newState = !currentState;

  // Optimistic UI update
  doc.departments[dept] = newState;
  doc.approved = Object.values(doc.departments).every((v) => v === true);
  updateButtonState(btn, newState);
  updateDashboards(dept);

  btn.disabled = true;

  try {
    const res = await fetch(`${apiBase}/departments/${docId}/${dept}/approve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Failed to update approval");
    }

    const data = await res.json();
    doc.departments[dept] = data.department_approved;
    doc.approved = data.document_approved;

    // Refresh all other departments from API
    for (const d of departments) {
      if (d === dept) continue;
      try {
        const resDept = await fetch(`${apiBase}/departments/${docId}/${d}`);
        const deptData = await resDept.json();
        doc.departments[d] = deptData?.every((p) => p.approved) || false;
      } catch (err) {
        console.error(`Failed to refresh ${d}:`, err);
      }
    }

    doc.approved = Object.values(doc.departments).every((v) => v === true);

    updateButtonState(btn, data.department_approved);
    btn.disabled = false;

    updateDashboards(dept);

    showNotification(
      `Department "${dept}" approval set to ${
        data.department_approved ? "approved" : "pending"
      }`,
      data.department_approved ? "success" : "pending"
    );
  } catch (err) {
    // Revert UI if API fails
    doc.departments[dept] = currentState;
    doc.approved = Object.values(doc.departments).every((v) => v === true);
    updateButtonState(btn, currentState);
    btn.disabled = false;

    updateDashboards(dept);
    console.error(err);
    showNotification(
      err.message || "Failed to update department approval",
      "error"
    );
  }
}

// ---------------- VIEW DEPARTMENT ----------------
async function viewDepartment(docId, dept) {
  showSection("department");
  document.getElementById(
    "deptTitle"
  ).innerHTML = `<i class="fa fa-building text-blue-600"></i> <span class="font-semibold">${dept}</span>`;
  const container = document.getElementById("deptDocs");
  container.innerHTML = `<div class="text-gray-500 py-8 text-center flex items-center justify-center gap-2"><div class="spinner"></div>Loading document for ${dept}...</div>`;

  try {
    if (!cachedDocs) {
      const res = await fetch(`${apiBase}/documents-list`);
      cachedDocs = await res.json();
    }

    const doc = cachedDocs.find((d) => d.id === docId);
    if (!doc) {
      container.innerHTML =
        "<div class='text-red-600 text-center py-8'>Document not found</div>";
      renderDashboard("deptDashboard", 0, 0, 0, 0, 0);
      return;
    }

    let deptData = [];
    try {
      const resDept = await fetch(`${apiBase}/departments/${doc.id}/${dept}`);
      deptData = await resDept.json();
    } catch (err) {
      console.error(err);
      deptData = [];
    }

    if (!deptData || deptData.length === 0) {
      container.innerHTML =
        "<div class='text-red-600 text-center py-8'>Document not assigned to this department</div>";
      renderDashboard("deptDashboard", 0, 0, 0, 0, 0);
      return;
    }

    container.innerHTML = "";
    const content = deptData.map((p) => p.content).join("\n\n");

    if (!doc.departments) doc.departments = {};
    doc.departments[dept] = deptData.every((p) => p.approved);
    doc.approved = Object.values(doc.departments).every((v) => v === true);

    const details = document.createElement("details");
    details.className = "doc-card bg-white p-6 rounded-xl shadow-sm transition";
    details.innerHTML = `
      <summary class="font-semibold text-lg text-gray-800 cursor-pointer flex items-center justify-between">
        <span class="truncate" title="${doc.filename}">${doc.filename}</span>
        <i class="fa fa-chevron-down text-gray-500"></i>
      </summary>
      <div class="mt-4">
        <div class="flex items-center gap-2 mb-2">
          <span class="text-sm text-gray-500">${new Date(
            doc.created_at
          ).toLocaleDateString()}</span>
          <span class="${
            doc.priority === "High" ? "priority-high" : "priority-low"
          }">
            ${doc.priority}
          </span>
        </div>
        <p class="text-gray-600 text-sm">${
          content.substring(0, 200) + (content.length > 200 ? "..." : "")
        }</p>
        <div class="mt-4 flex gap-2">
          <button type="button" onclick="viewFullDocumentDept('${
            doc.id
          }', '${dept}')" class="btn-primary" aria-label="View Full ${
      doc.filename
    } in ${dept}">
            <i class="fa fa-eye"></i> View Full
          </button>
          <button type="button" class="btn-secondary approve-btn ${
            doc.departments[dept]
              ? "bg-yellow-100 text-yellow-800 border-yellow-300"
              : ""
          }" data-doc="${doc.id}" data-dept="${dept}">
            <i class="fa fa-${doc.departments[dept] ? "edit" : "check"}"></i>
            ${doc.departments[dept] ? "Review" : "Approve"}
          </button>
        </div>
      </div>
    `;
    container.appendChild(details);

    details.querySelector(".approve-btn").onclick = (e) => {
      approveDepartment(doc.id, dept, e.currentTarget);
    };

    const total = 1;
    const high = doc.priority === "High" ? 1 : 0;
    const low = doc.priority === "Low" ? 1 : 0;
    const pending = doc.departments[dept] ? 0 : 1;
    const approved = doc.departments[dept] ? 1 : 0;

    renderDashboard("deptDashboard", total, high, low, pending, approved);
  } catch (err) {
    console.error(err);
    container.innerHTML =
      "<div class='text-red-600 font-medium text-center py-8'>Failed to load document</div>";
    renderDashboard("deptDashboard", 0, 0, 0, 0, 0);
  }
}

// ---------------- UPDATE UI FUNCTIONS ----------------
function updateHomeDashboard() {
  if (!cachedDocs) return;

  const total = cachedDocs.length;
  const high = cachedDocs.filter((d) => d.priority === "High").length;
  const low = cachedDocs.filter((d) => d.priority === "Low").length;
  const approved = cachedDocs.filter((d) => d.approved).length;
  const pending = total - approved;

  renderDashboard("homeDashboard", total, high, low, pending, approved);
}

function updateDepartmentDashboard(dept) {
  if (!cachedDocs || !dept) return;

  const deptDocs = cachedDocs.filter(
    (d) => d.departments && d.departments.hasOwnProperty(dept)
  );
  const total = deptDocs.length;
  const high = deptDocs.filter((d) => d.priority === "High").length;
  const low = deptDocs.filter((d) => d.priority === "Low").length;
  const approved = deptDocs.filter((d) => d.departments[dept] === true).length;
  const pending = total - approved;

  renderDashboard("deptDashboard", total, high, low, pending, approved);
}

function updateDashboards(dept) {
  updateHomeDashboard();
  updateDepartmentDashboard(dept);
  renderDocumentsTable(); // Update table if on upload section
}

function updateButtonState(btn, approved) {
  btn.innerHTML = `<i class="fa fa-${approved ? "edit" : "check"}"></i> ${
    approved ? "Review" : "Approve"
  }`;

  btn.classList.remove("bg-yellow-100", "text-yellow-800", "border-yellow-300");
  if (approved) {
    btn.classList.add("bg-yellow-100", "text-yellow-800", "border-yellow-300");
  }
}

// ---------------- UPLOAD HANDLER ----------------
async function handleUpload() {
  const fileInput = document.getElementById("fileInput");
  const priorityRadio = document.querySelector(
    'input[name="priority"]:checked'
  );
  const priority = priorityRadio ? priorityRadio.value : "Low";

  if (fileInput.files.length === 0) {
    showNotification("Please select a file", "error");
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);
  formData.append("priority", priority);

  try {
    const res = await fetch(`${apiBase}/upload`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const errData = await res.json();
      showNotification(errData.detail || "Failed to upload document", "error");
      return;
    }

    const data = await res.json();
    showNotification(
      `Document "${data.filename}" uploaded successfully!`,
      "success"
    );

    fileInput.value = "";
    document.querySelector(
      'input[name="priority"][value="Low"]'
    ).checked = true;
    cachedDocs = null;
    loadHomeDocs();
    setTimeout(() => {
      if (cachedDocs) renderDocumentsTable();
    }, 1000);
  } catch (err) {
    console.error(err);
    showNotification("Failed to upload document", "error");
  }
}

// ---------------- DASHBOARD SECTION ----------------
function showDashboard() {
  showSection("dashboard");
  if (cachedDocs) {
    const total = cachedDocs.length;
    const high = cachedDocs.filter((d) => d.priority === "High").length;
    const low = cachedDocs.filter((d) => d.priority === "Low").length;
    const approved = cachedDocs.filter((d) => d.approved).length;
    const pending = total - approved;

    renderDashboard("mainDashboard", total, high, low, pending, approved);
    renderRecentDocuments();
    renderDepartmentStatus();
  }
}

function renderRecentDocuments() {
  const container = document.getElementById("recentDocuments");
  if (!cachedDocs) return;

  const recent = cachedDocs
    .sort((a, b) => new Date(b.created_at) - new Date(a.created_at))
    .slice(0, 5);

  container.innerHTML = recent
    .map(
      (doc) => `
    <div class="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
      <div>
        <p class="font-medium text-sm text-gray-900">${doc.filename}</p>
        <p class="text-xs text-gray-500">${new Date(
          doc.created_at
        ).toLocaleDateString()}</p>
      </div>
      <span class="${
        doc.priority === "High" ? "priority-high" : "priority-low"
      }">${doc.priority}</span>
    </div>
  `
    )
    .join("");
}

function renderDepartmentStatus() {
  const container = document.getElementById("departmentStatus");
  if (!cachedDocs) return;

  container.innerHTML = departments
    .map((dept) => {
      const deptDocs = cachedDocs.filter(
        (d) => d.departments && d.departments.hasOwnProperty(dept)
      );
      const approved = deptDocs.filter(
        (d) => d.departments[dept] === true
      ).length;
      const total = deptDocs.length;
      const percentage = total > 0 ? Math.round((approved / total) * 100) : 0;

      return `
      <div class="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
        <div>
          <p class="font-medium text-sm text-gray-900">${dept}</p>
          <p class="text-xs text-gray-500">${approved}/${total} approved</p>
        </div>
        <div class="text-right">
          <span class="text-sm font-medium text-gray-900">${percentage}%</span>
          <div class="w-16 h-2 bg-gray-200 rounded-full mt-1">
            <div class="h-2 bg-blue-600 rounded-full" style="width: ${percentage}%"></div>
          </div>
        </div>
      </div>
    `;
    })
    .join("");
}

// ---------------- TAB SWITCHING ----------------
function switchTab(tab) {
  currentTab = tab;
  const contentTab = document.getElementById("tabContent");
  const tablesTab = document.getElementById("tabTables");
  const contentBtn = document.getElementById("tabContentBtn");
  const tablesBtn = document.getElementById("tabTablesBtn");

  if (tab === "content") {
    contentTab.classList.remove("hidden");
    tablesTab.classList.add("hidden");
    contentBtn.classList.add("border-blue-600", "text-blue-600");
    contentBtn.classList.remove("text-gray-600");
    tablesBtn.classList.remove("border-blue-600", "text-blue-600");
    tablesBtn.classList.add("text-gray-600");
  } else {
    contentTab.classList.add("hidden");
    tablesTab.classList.remove("hidden");
    contentBtn.classList.remove("border-blue-600", "text-blue-600");
    contentBtn.classList.add("text-gray-600");
    tablesBtn.classList.add("border-blue-600", "text-blue-600");
    tablesBtn.classList.remove("text-gray-600");
  }
}

// ---------------- NOTIFICATION ----------------
function showNotification(message, type = "info", duration = 3000) {
  const container = document.getElementById("notificationContainer");
  const notif = document.createElement("div");
  notif.className = `notification ${type} flex items-center`;

  notif.innerHTML = `
    <div class="flex-1 pr-4">${message}</div>
    <button
      class="ml-2 text-white hover:text-gray-200"
      onclick="this.parentElement.classList.remove('show'); setTimeout(() => this.parentElement.remove(), 400)"
      aria-label="Dismiss Notification"
    >
      <i class="fa fa-times"></i>
    </button>
  `;

  container.appendChild(notif);

  setTimeout(() => notif.classList.add("show"), 10);
  setTimeout(() => {
    notif.classList.remove("show");
    setTimeout(() => container.removeChild(notif), 400);
  }, duration);
}

// ---------------- DASHBOARD RENDER ----------------
function renderDashboard(targetId, total, high, low, pending, approved) {
  const container = document.getElementById(targetId);
  if (!container) return;

  container.innerHTML = "";

  const cards = [
    { title: "Total Documents", count: total, color: "blue", icon: "fa-file" },
    {
      title: "High Priority",
      count: high,
      color: "red",
      icon: "fa-exclamation-triangle",
    },
    {
      title: "Low Priority",
      count: low,
      color: "green",
      icon: "fa-check-circle",
    },
    {
      title: "Pending Documents",
      count: pending,
      color: "yellow",
      icon: "fa-hourglass-half",
    },
    {
      title: "Approved Documents",
      count: approved,
      color: "teal",
      icon: "fa-check",
    },
  ];

  cards.forEach((card) => {
    const div = document.createElement("div");
    div.className = "dashboard-card";
    div.innerHTML = `
      <div class="flex items-center justify-between">
        <div>
          <p class="text-sm font-medium text-gray-600 mb-1">${card.title}</p>
          <p class="text-2xl font-bold text-gray-900">${card.count}</p>
        </div>
        <div class="w-12 h-12 bg-${card.color}-50 rounded-lg flex items-center justify-center">
          <i class="fa ${card.icon} text-${card.color}-600 text-xl"></i>
        </div>
      </div>
    `;
    container.appendChild(div);
  });
}

// ---------------- FILE UPLOAD DRAG & DROP ----------------
function setupFileUpload() {
  const uploadArea = document.querySelector(".upload-area");
  const fileInput = document.getElementById("fileInput");

  if (!uploadArea || !fileInput) return;

  uploadArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadArea.classList.add("dragover");
  });

  uploadArea.addEventListener("dragleave", (e) => {
    e.preventDefault();
    uploadArea.classList.remove("dragover");
  });

  uploadArea.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadArea.classList.remove("dragover");
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      fileInput.files = files;
      updateUploadAreaText(files[0].name);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      updateUploadAreaText(e.target.files[0].name);
    }
  });
}

function updateUploadAreaText(filename) {
  const content = document.getElementById("uploadAreaContent");
  if (content) {
    content.innerHTML = `
      <div class="w-12 h-12 bg-green-50 rounded-full flex items-center justify-center mb-4">
        <i class="fa fa-check text-green-500 text-xl"></i>
      </div>
      <p class="text-green-600 font-medium mb-1">${filename}</p>
      <p class="text-sm text-gray-500">Click to change file</p>
    `;
  }
}

// ---------------- INIT ----------------
function init() {
  loadDepartmentsDropdown();
  setupFileUpload();

  // Show upload section by default
  showSection("upload");

  // Load documents for table
  loadHomeDocs().then(() => {
    if (cachedDocs) {
      renderDocumentsTable();
    }
  });
}

// Start the application
document.addEventListener("DOMContentLoaded", init);
