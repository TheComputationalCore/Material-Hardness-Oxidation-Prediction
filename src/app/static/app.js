(() => {
  const $ = (selector) => document.querySelector(selector);
  const form = $("#predict-form");
  const button = $("#predict-btn");
  const spinner = $("#btn-spinner");
  const buttonText = $("#btn-text");
  const resultsArea = $("#results-area");

  let lastResults = "";

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

  const formatNumber = (value, decimals) => Number(value).toFixed(decimals);

  const appendMetaLine = (container, text) => {
    const line = document.createElement("div");
    line.className = "result-meta";
    line.textContent = text;
    container.appendChild(line);
  };

  const appendResult = (box, label, valueId, value, decimals, unit, meta) => {
    const row = document.createElement("div");
    row.className = "result-row";

    const name = document.createElement("strong");
    name.textContent = label + ":";
    const number = document.createElement("span");
    number.id = valueId;
    number.textContent = formatNumber(value, decimals);
    row.appendChild(name);
    row.appendChild(document.createTextNode(" "));
    row.appendChild(number);
    if (unit) row.appendChild(document.createTextNode(" " + unit));

    if (meta && meta.applicability) {
      const badge = document.createElement("span");
      badge.className = "badge " + meta.applicability.status;
      badge.textContent = meta.applicability.status === "reported_condition"
        ? "Reported condition"
        : meta.applicability.status.replaceAll("_", " ");
      row.appendChild(badge);
    }
    box.appendChild(row);

    if (meta && meta.interval) {
      appendMetaLine(
        box,
        "90% empirical interval: " +
          formatNumber(meta.interval.lower, decimals) + "–" +
          formatNumber(meta.interval.upper, decimals) +
          (unit ? " " + unit : "") +
          " (approximate, from " + meta.interval.n_oof_predictions +
          " out-of-fold residuals)"
      );
    }
    if (meta && meta.applicability && meta.applicability.measured_value !== null &&
        meta.applicability.measured_value !== undefined) {
      appendMetaLine(
        box,
        "Paper-reported value at this condition: " +
          formatNumber(meta.applicability.measured_value, decimals) +
          (unit ? " " + unit : "") +
          " (" + meta.applicability.measured_source_table + ")."
      );
    }
    if (meta && meta.applicability && meta.applicability.status !== "reported_condition") {
      appendMetaLine(box, meta.applicability.explanation);
    }
  };

  const appendWarnings = (container, warnings) => {
    if (!warnings || !warnings.length) return;
    const list = document.createElement("ul");
    list.className = "warning-list";
    warnings.forEach((warning) => {
      const item = document.createElement("li");
      item.textContent = warning;
      list.appendChild(item);
    });
    container.appendChild(list);
  };

  const renderResults = (
    hardness,
    massGain,
    hardnessError,
    oxidationError,
    hardnessMeta,
    oxidationMeta,
    warnings
  ) => {
    resultsArea.replaceChildren();
    const box = document.createElement("div");
    box.className = "result-item";
    const lines = [];

    if (hardness !== null && hardness !== undefined) {
      appendResult(box, "Hardness", "hardness-value", hardness, 2, "", hardnessMeta);
      lines.push("Hardness: " + formatNumber(hardness, 2) + " HV");
      if (hardnessMeta && hardnessMeta.interval) {
        lines.push(
          "90% interval: " + formatNumber(hardnessMeta.interval.lower, 2) +
          "–" + formatNumber(hardnessMeta.interval.upper, 2)
        );
      }
    }
    if (massGain !== null && massGain !== undefined) {
      appendResult(
        box, "Areal mass gain", "oxidation-value", massGain, 6, "g/mm²", oxidationMeta
      );
      lines.push("Areal mass gain: " + formatNumber(massGain, 6) + " g/mm²");
      if (oxidationMeta && oxidationMeta.interval) {
        lines.push(
          "90% interval: " + formatNumber(oxidationMeta.interval.lower, 6) +
          "–" + formatNumber(oxidationMeta.interval.upper, 6) + " g/mm²"
        );
      }
    }
    if (hardnessError) {
      const error = document.createElement("div");
      error.className = "error";
      error.textContent = "Hardness error: " + hardnessError;
      box.appendChild(error);
      lines.push("Hardness error: " + hardnessError);
    }
    if (oxidationError) {
      const error = document.createElement("div");
      error.className = "error";
      error.textContent = "Oxidation error: " + oxidationError;
      box.appendChild(error);
      lines.push("Oxidation error: " + oxidationError);
    }

    resultsArea.appendChild(box);
    appendWarnings(resultsArea, warnings);
    lastResults = lines.join("\n");
  };

  const initial = window.__INITIAL__ || {};
  if (initial.hardness !== null || initial.oxidation !== null ||
      initial.hardness_error || initial.oxidation_error) {
    renderResults(
      initial.hardness,
      initial.oxidation,
      initial.hardness_error,
      initial.oxidation_error,
      initial.hardness_meta,
      initial.oxidation_meta,
      initial.warnings
    );
  }

  const setBusy = (busy) => {
    button.disabled = busy;
    spinner.style.display = busy ? "inline-block" : "none";
    buttonText.textContent = busy ? "Predicting…" : "Predict";
  };

  if ($("#copy-btn")) {
    $("#copy-btn").addEventListener("click", async () => {
      const text = lastResults ||
        (resultsArea.textContent || "").trim();
      if (!text) {
        showToast("Nothing to copy yet");
        return;
      }
      try {
        await navigator.clipboard.writeText(text);
        showToast("Results copied");
      } catch (_) {
        showToast("Copy failed");
      }
    });
  }

  if ($("#reset-btn")) {
    $("#reset-btn").addEventListener("click", () => {
      form.reset();
      updateProcessNote();
      lastResults = "";
      resultsArea.innerHTML =
        '<div class="muted">Choose one of the published material, current, and exposure conditions to view model estimates.</div>';
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
          data.oxidation_error,
          { interval: data.hardness_interval_90, applicability: data.hardness_applicability },
          { interval: data.oxidation_interval_90, applicability: data.oxidation_applicability },
          data.warnings
        );
        showToast(response.ok ? "Prediction complete" : "Input could not be predicted");
      } catch (_) {
        showToast("Prediction request failed");
      } finally {
        setBusy(false);
      }
    });
  }

  const addInfoRow = (parent, label, value) => {
    const row = document.createElement("div");
    row.className = "info-row";
    const key = document.createElement("span");
    key.className = "info-label";
    key.textContent = label;
    const text = document.createElement("span");
    text.textContent = value;
    row.appendChild(key);
    row.appendChild(text);
    parent.appendChild(row);
  };

  const formatMetric = (name, value) => {
    if (name === "R2") return Number(value).toFixed(4);
    if (Math.abs(value) < 1e-3 && value !== 0) return Number(value).toExponential(3);
    return Number(value).toFixed(4);
  };

  const renderModelInfo = (info) => {
    const body = $("#model-info-body");
    if (!body) return;
    body.replaceChildren();
    const intro = document.createElement("p");
    intro.className = "muted";
    intro.textContent =
      "Versioned artifacts trained from the audited paper tables. Intervals are approximate.";
    body.appendChild(intro);

    Object.entries(info.models || {}).forEach(([kind, model]) => {
      const heading = document.createElement("h4");
      heading.textContent = model.model_name || kind;
      body.appendChild(heading);
      addInfoRow(body, "Version", model.model_version || "unknown");
      addInfoRow(body, "Target", model.target ? model.target.column : "unknown");
      addInfoRow(body, "Features", (model.features || []).join(", "));
      if (model.metrics) {
        Object.entries(model.metrics).forEach(([name, value]) => {
          addInfoRow(body, name, formatMetric(name, value));
        });
      }
      if (model.validation) {
        addInfoRow(
          body,
          "Validation",
          model.validation.method + " by " + model.validation.group +
          " (" + model.validation.n_samples + " rows)"
        );
        if (model.validation.effective_independent_observations) {
          addInfoRow(
            body,
            "Effective n",
            String(model.validation.effective_independent_observations)
          );
        }
      }
      if (model.dataset && model.dataset.sha256) {
        addInfoRow(body, "Dataset SHA-256", model.dataset.sha256.slice(0, 16) + "…");
      }
      if (model.artifact_sha256) {
        addInfoRow(body, "Artifact SHA-256", model.artifact_sha256.slice(0, 16) + "…");
      }
      if (model.known_limitations) {
        const list = document.createElement("ul");
        list.className = "warning-list";
        model.known_limitations.forEach((item) => {
          const li = document.createElement("li");
          li.textContent = item;
          list.appendChild(li);
        });
        body.appendChild(list);
      }
    });

    if (info.calibration && info.calibration.targets) {
      const heading = document.createElement("h4");
      heading.textContent = "Uncertainty";
      body.appendChild(heading);
      Object.entries(info.calibration.targets).forEach(([kind, target]) => {
        const offsets = target.interval_90_offsets;
        const summary = offsets
          ? "90% interval offsets " +
            formatMetric("x", offsets.lower) + " / +" +
            formatMetric("x", offsets.upper)
          : "90% interval half-width " +
            formatMetric("x", target.interval_90_half_width);
        addInfoRow(
          body,
          kind,
          summary +
          " from " + target.n_oof_predictions + " OOF residuals" +
          (target.effective_independent_observations
            ? " (" + target.effective_independent_observations + " effective conditions)"
            : "")
        );
      });
      const note = document.createElement("p");
      note.className = "muted";
      note.textContent = info.calibration.method;
      body.appendChild(note);
    }
  };

  if ($("#model-info")) {
    fetch("/api/v1/model-info")
      .then((response) => response.ok ? response.json() : Promise.reject(response))
      .then(renderModelInfo)
      .catch(() => {
        $("#model-info-body").textContent = "Model details are unavailable.";
      });
  }
})();
