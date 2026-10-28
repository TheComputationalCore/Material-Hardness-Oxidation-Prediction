(() => {
  const $ = (selector) => document.querySelector(selector);
  const form = $("#predict-form");
  const button = $("#predict-btn");
  const spinner = $("#btn-spinner");
  const buttonText = $("#btn-text");
  const resultsArea = $("#results-area");

  const showToast = (message, timeout = 3000) => {
    const toast = $("#toast");
    if (!toast) return;
    toast.textContent = message;
    toast.style.display = "block";
    window.setTimeout(() => { toast.style.display = "none"; }, timeout);
  };

  const applyTheme = (dark) => {
    document.body.classList.toggle("dark", dark);
    try {
      window.localStorage.setItem("mhoc_dark", dark ? "1" : "0");
    } catch (_) {
      // Theme remains usable when browser storage is unavailable.
    }
  };

  try {
    applyTheme(window.localStorage.getItem("mhoc_dark") === "1");
  } catch (_) {
    applyTheme(false);
  }

  const themeButton = $("#theme-toggle");
  if (themeButton) {
    themeButton.addEventListener("click", () => {
      applyTheme(!document.body.classList.contains("dark"));
    });
  }

  const schedule = {
    "120": { heatInput: "0.800", travelSpeed: "2.25" },
    "140": { heatInput: "0.864", travelSpeed: "2.43" },
    "160": { heatInput: "0.912", travelSpeed: "2.63" }
  };
  const currentSelect = $("#Current");
  const heatInputNote = $("#heat-input-note");
  const updateProcessNote = () => {
    const condition = schedule[currentSelect.value];
    heatInputNote.textContent =
      "Paper schedule at " + currentSelect.value + " A: heat input " +
      condition.heatInput + " kJ/mm and travel speed " +
      condition.travelSpeed + " mm/s. These conditions vary together.";
  };
  currentSelect.addEventListener("change", updateProcessNote);
  updateProcessNote();

  const renderResults = (hardness, massGain, hardnessError, oxidationError) => {
    resultsArea.replaceChildren();
    const box = document.createElement("div");
    box.className = "result-item";

    if (hardness !== null && hardness !== undefined) {
      const row = document.createElement("div");
      row.className = "result-row";
      row.innerHTML = "<strong>Hardness:</strong> <span id=\"hardness-value\">" +
        Number(hardness).toFixed(2) + "</span>";
      box.appendChild(row);
    }
    if (massGain !== null && massGain !== undefined) {
      const row = document.createElement("div");
      row.className = "result-row";
      row.innerHTML = "<strong>Areal mass gain:</strong> <span id=\"oxidation-value\">" +
        Number(massGain).toFixed(6) + " g/mm²</span>";
      box.appendChild(row);
    }
    if (hardnessError) {
      const error = document.createElement("div");
      error.className = "error";
      error.textContent = "Hardness error: " + hardnessError;
      box.appendChild(error);
    }
    if (oxidationError) {
      const error = document.createElement("div");
      error.className = "error";
      error.textContent = "Oxidation error: " + oxidationError;
      box.appendChild(error);
    }
    resultsArea.appendChild(box);
  };

  const initial = window.__INITIAL__ || {};
  if (initial.hardness !== null || initial.oxidation !== null ||
      initial.hardness_error || initial.oxidation_error) {
    renderResults(
      initial.hardness,
      initial.oxidation,
      initial.hardness_error,
      initial.oxidation_error
    );
  }

  const setBusy = (busy) => {
    button.disabled = busy;
    spinner.style.display = busy ? "inline-block" : "none";
    buttonText.textContent = busy ? "Predicting…" : "Predict";
  };

  if ($("#copy-btn")) {
    $("#copy-btn").addEventListener("click", async () => {
      const hardness = $("#hardness-value")?.textContent || "";
      const oxidation = $("#oxidation-value")?.textContent || "";
      try {
        await navigator.clipboard.writeText(
          "Hardness: " + hardness + "\nAreal mass gain: " + oxidation
        );
        showToast("Results copied");
      } catch (_) {
        showToast("Copy failed");
      }
    });
  }

  if ($("#reset-btn")) {
    $("#reset-btn").addEventListener("click", () => {
      form.reset();
      resultsArea.innerHTML =
        '<div class="muted">Choose paper-tested conditions and click Predict.</div>';
    });
  }

  if (form) {
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      setBusy(true);
      const payload = {
        Material: $("#Material").value,
        Current: Number($("#Current").value),
        Exposure_Time_h: Number($("#Exposure_Time_h").value)
      };
      try {
        const response = await fetch("/api/v1/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const data = await response.json();
        renderResults(
          data.hardness,
          data.oxidation_areal_mass_gain_g_mm2,
          data.hardness_error,
          data.oxidation_error || data.error
        );
        showToast(response.ok ? "Prediction complete" : "Input could not be predicted");
      } catch (_) {
        showToast("Prediction request failed");
      } finally {
        setBusy(false);
      }
    });
  }
})();
