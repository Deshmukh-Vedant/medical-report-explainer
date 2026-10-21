/* ==========================================================
   upload.js — Drag & drop upload, progress bar, trigger analysis
   Used on dashboard.html
   ========================================================== */

document.addEventListener("DOMContentLoaded", () => {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const progressWrap = document.getElementById("uploadProgressWrap");
  const progressBar = document.getElementById("uploadProgressBar");
  const progressLabel = document.getElementById("uploadProgressLabel");
  const hospitalInput = document.getElementById("hospitalNameInput");

  if (!dropzone || !fileInput) return; // not on dashboard page

  dropzone.addEventListener("click", () => fileInput.click());

  ["dragenter", "dragover"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    const files = e.dataTransfer ? e.dataTransfer.files : [];
    if (files.length) handleFileUpload(files[0]);
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length) {
      handleFileUpload(e.target.files[0]);
      e.target.value = ""; // Reset so same file can be selected again
    }
  });

  function handleFileUpload(file) {
    if (!file) return;

    const isPdfFile =
      file.type === "application/pdf" ||
      (file.name && file.name.toLowerCase().endsWith(".pdf")) ||
      (file.type === "application/octet-stream" && file.name && file.name.toLowerCase().endsWith(".pdf"));

    if (!isPdfFile) {
      showToast("Only PDF files are supported.", "error");
      return;
    }
    if (file.size > 20 * 1024 * 1024) {
      showToast("File exceeds the 20MB size limit.", "error");
      return;
    }

    uploadFile(file);
  }

  function uploadFile(file) {
    const token = getToken();
    if (!token) {
      showToast("Please log in to upload a medical report.", "warning");
      setTimeout(() => { window.location.href = "login.html"; }, 1500);
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    const hospitalName = hospitalInput ? hospitalInput.value.trim() : "";
    const base = API_BASE_URL || window.location.origin;
    const url = new URL(`${base}/upload`, window.location.origin);
    if (hospitalName) url.searchParams.set("hospital_name", hospitalName);

    progressWrap.style.display = "block";
    progressBar.style.width = "0%";
    progressLabel.innerText = "Uploading...";

    const xhr = new XMLHttpRequest();
    xhr.open("POST", url.toString());
    xhr.setRequestHeader("Authorization", `Bearer ${token}`);

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) {
        const percent = Math.round((e.loaded / e.total) * 100);
        progressBar.style.width = `${percent}%`;
        progressBar.setAttribute("aria-valuenow", percent);
        progressLabel.innerText = `Uploading... ${percent}%`;
      }
    };

    xhr.onload = async () => {
      if (xhr.status === 201) {
        try {
          const data = JSON.parse(xhr.responseText);
          progressLabel.innerText = "Extracting report text complete. Running AI analysis...";
          showToast("File uploaded successfully!", "success");
          await runAnalysis(data.report_id);
        } catch (parseErr) {
          showToast("Upload response could not be parsed.", "error");
          progressWrap.style.display = "none";
        }
      } else if (xhr.status === 401) {
        showToast("Your session has expired. Please log in again.", "error");
        clearToken();
        setTimeout(() => { window.location.href = "login.html"; }, 1500);
      } else {
        let detail = "Upload failed. Please try again.";
        try {
          detail = JSON.parse(xhr.responseText).detail || detail;
        } catch (_) {}
        showToast(detail, "error");
        progressWrap.style.display = "none";
      }
    };

    xhr.onerror = () => {
      showToast("Network error during upload. Is the backend running?", "error");
      progressWrap.style.display = "none";
    };

    xhr.send(formData);
  }

  async function runAnalysis(reportId) {
    showSkeletonLoading(true);
    try {
      const response = await apiRequest(`/analyze/${reportId}`, { method: "POST" });
      if (!response) return;

      if (!response.ok) {
        const err = await response.json();
        showToast(err.detail || "AI analysis failed.", "error");
        return;
      }

      const analysis = await response.json();
      showToast("AI analysis complete!", "success");
      progressWrap.style.display = "none";

      if (typeof renderAnalysis === "function") {
        renderAnalysis(analysis, reportId);
      }
      if (typeof loadRecentReports === "function") {
        loadRecentReports();
      }
    } catch (err) {
      console.error(err);
      showToast("Something went wrong while analyzing the report.", "error");
    } finally {
      showSkeletonLoading(false);
    }
  }

  function showSkeletonLoading(show) {
    const skeleton = document.getElementById("analysisSkeleton");
    const resultsArea = document.getElementById("analysisResults");
    if (skeleton) skeleton.style.display = show ? "block" : "none";
    if (resultsArea) resultsArea.style.display = show ? "none" : "block";
  }
});
