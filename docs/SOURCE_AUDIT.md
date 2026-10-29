# Source audit: what the paper reports and what remains unresolved

This document records the audit of the data transcribed from:

> C. Dinesh Chandra, B. Rushikesh, Mohammed Numan, and B. Venkatesh,
> “Experimental Studies of Stellite-6 Hardfaced Layer on Ferrous Materials by TIG
> Surfacing Process,” *IOP Conference Series: Materials Science and Engineering*
> 998 (2020), 012061. DOI: [10.1088/1757-899X/998/1/012061](https://doi.org/10.1088/1757-899X/998/1/012061).

The paper is CC BY 3.0. Two PDF copies were compared: an author-supplied copy and
the open-access IOP copy. All tabulated values and both oxidation equations agree;
the files differ only in rendering metadata. Their SHA-256 digests are recorded in
[`data/source_tables.json`](../data/source_tables.json) and in the generated audit
report.

Run the audit with:

```bash
python -m src.data.audit
```

It writes [`reports/audit/source_audit.json`](../reports/audit/source_audit.json) and
[`reports/audit/source_audit.md`](../reports/audit/source_audit.md). The test suite
asserts the findings below.

## 1. Table transcription

| Check | Result |
|---|---|
| Hardness rows compared with Tables 5 and 6 | 6 / 6 exact |
| Oxidation rows compared with Tables 8–13 | 18 / 18 exact |
| Source-table labels | 24 / 24 correct |
| Dataset values changed during provenance work | none (maximum numeric difference 0) |

Every training row carries `Data_Status = reported_paper_measurement`,
`Source`, `Source_Table`, an assigned `Row_ID` and `Specimen_ID`, and explicit
derivation, replicate, and chemistry-status labels. The paper does not publish
specimen identifiers; the project assigns them.

## 2. Process schedule arithmetic

Table 1 prints current, travel speed, and heat input. The printed heat input is
reproduced by

```text
Q = V I / v,   V = 15 V
```

without an arc-efficiency factor:

| Current (A) | Travel speed (mm/s) | Printed Q (kJ/mm) | Computed Q (kJ/mm) |
|---:|---:|---:|---:|
| 120 | 2.25 | 0.800 | 0.800000 |
| 140 | 2.43 | 0.864 | 0.864198 |
| 160 | 2.63 | 0.912 | 0.912548 |

The maximum difference is 5.5 × 10⁻⁴ kJ/mm, consistent with rounding. **Current,
travel speed, and heat input therefore form a single one-dimensional schedule.**
The dataset cannot separate their effects, and no model in this project claims to.

## 3. Oxidation exposure-time multiples

Each Material × Current cell has values at 5, 10, and 15 h. Eleven of the twelve
non-base rows equal an exact multiple of the 5 h value at printed precision:

- 10 h = 2 × 5 h (all six cells);
- 15 h = 3 × 5 h (five of six cells);
- Mild Steel at 140 A, 15 h prints 0.007740 against 3 × 0.002580 = 0.007740
  (exact);
- Mild Steel at 140 A, 10 h prints 0.005156 against 2 × 0.002580 = 0.005160, a
  relative deviation of 7.8 × 10⁻⁴ (0.08%).

The pattern is consistent with the authors computing 10 h and 15 h values from a
linear rate. Regardless of how the table was produced, **the eighteen oxidation
rows do not contain eighteen independent kinetic observations.** The effective
number of independent Material × Current observations is six, and the exposure
time adds almost no independent information beyond a near-linear trend. The
oxidation model is therefore a rate model with six effective conditions.

## 4. Specimen dimensions and oxidation unit — unresolved

The paper gives two incompatible specimen sizes:

| Location | Statement |
|---|---|
| Methods | `Samples sizes of 30X30mm2 were cut from hardfaced plate` |
| Table 7 | `Sample size 15x15x6mm; surface area = 15x15mm2` |

The exposed areas differ by a factor of four (900 mm² vs 225 mm²). Table 7 defines
an areal quantity and its header reads `gm/mm2`. The project uses the Table 7 area
of 225 mm² as the nominal normalization because it accompanies the oxidation
tables, but the conflict means the normalization itself is not uniquely determined.

If the numbers are interpreted literally as g/mm² at 225 mm²:

- the implied mass change is 0.487–1.817 g;
- for a 15 × 15 × 6 mm steel specimen of about 10.6 g, that is 4.6–17.1 % of the
  specimen mass;
- this is not plausible for a 5–15 h exposure at 700 °C.

If the numbers are interpreted as raw weight change in grams:

- at 225 mm² the implied areal gain is 0.96–3.59 mg/cm²;
- at 900 mm² it is 0.24–0.90 mg/cm²;
- both are physically plausible for the stated exposure, with the 900 mm² reading
  closer to typical 700 °C oxidation gains for these alloys.

The magnitude evidence therefore favours the values being raw grams, but the fourfold
area conflict prevents a unique conversion. **Resolution requires the raw weights,
the measured exposed area, and the authors' clarification, none of which is
published.** The CSV preserves the printed numbers and labels the unit
`unresolved_inconsistent_with_stated_area`. The API and UI report values “as
printed” and never silently rescale them. Comparisons within the dataset are
unaffected by a common scale factor, but absolute physical interpretation is not
safe. The audit records the provisional reading in
`reports/audit/source_audit.json` under `unit_analysis.provisional_reading`.

## 5. Oxidation sign convention — unresolved

Table 7 prints:

```text
weight gain = w1 (before oxidation) - w2 (after oxidation)
```

For an oxidation test the specimen normally gains mass, so `w2 > w1` and this
subtraction would be negative. Every tabulated value is positive. Either the
equation is transposed, or the specimens lost mass (for example, oxide spallation
or substrate volatilization). The raw weights are not published.

**No sign correction is applied.** The dataset preserves the reported values and
the audit records the inconsistency. Any physical interpretation of the target as
literal mass gain remains provisional.

## 6. Independence and replication

| Target | Rows | Conditions | Replicates | Effective independent observations |
|---|---:|---:|---|---:|
| Hardness | 6 | 6 Material × Current cells | none reported | 6 |
| Oxidation | 18 | 6 Material × Current cells × 3 times | none reported | 6 |

The paper does not report replicate specimens, repeated indentations, repeated
weighings, or measurement uncertainty. Specimen reuse across exposure times is not
stated. It does say that three samples were kept in the furnace per run, which is
consistent with one specimen at each current rather than triplicate specimens at a
single condition. Treating the oxidation table as an 18-sample dataset overstates
the information by roughly a factor of three.

## 7. Substrate chemistry transcription warning

The paper's substrate composition table lists, for Mild Steel, sulfur 0.55 % and
phosphorus 0.40 %. Those values are implausible for a structural mild steel and
may be transposed or misprinted. The CSV records carbon and manganese as metadata
only, with `Chemistry_Status = midpoint_of_reported_range`; chemistry is not a
model feature. The sulfur and phosphorus values are retained in
`source_tables.json` with a transcription warning rather than silently corrected.

## 8. Missing experimental metadata

The paper does not report:

- the Vickers test load (only “a range of light loads”);
- the number and location of hardness indentations per specimen;
- the number of replicate specimens or measurements per condition;
- raw specimen weights w1 and w2;
- whether the exposed area was 225 or 900 mm² (the methods and Table 7 disagree);
- balance resolution or weighing procedure;
- whether the exposed area was measured or taken from nominal dimensions;
- furnace atmosphere, heating rate, or cooling rate;
- travel-speed measurement uncertainty (measured with a stopwatch);
- deposit dilution, overlay thickness, or interface characterization beyond
  optical microscopy;
- specimen identifiers.

These omissions bound what any model trained on this table can claim.

## 8a. Questions for the authors

Resolving the remaining audit items needs data the paper does not publish. A short
enquiry to the corresponding author (address printed on the paper) should ask:

1. Are the tabulated oxidation values raw mass change in grams or an areal
   quantity, and what area normalizes them?
2. What are the raw weights w1 and w2 for each coupon?
3. Is the printed `w1 − w2` the intended definition, or are the values |w2 − w1|?
4. Were the 10 h and 15 h rows measured on separate coupons or derived from the
   5 h rate?
5. Were the coupons 15 × 15 mm or 30 × 30 mm, and was the exposed area measured?
6. What Vickers load, indentation count, and replicate count were used?

Any reply that changes a value or unit must be committed with an updated audit,
dataset hash, reproducibility baseline, and model card, rather than edited in place.

## 9. What the project does about the findings

| Finding | Action |
|---|---|
| Values match the paper | Datasets are canonical and hashed; audit is a CI gate. |
| Process factors are collinear | Models use current as the process index; documentation states that effects are not separable. |
| Oxidation times are multiples | The model predicts a rate and scales by time; effective n = 6 is reported in every manifest and metric table. |
| Unit is unresolved | Values are labelled “as printed”; a warning appears in the API, UI, model metadata, and model card. |
| Specimen area is inconsistent | The nominal 225 mm² is used and the 4× methods conflict is documented; the audit records a provisional “raw grams” reading without applying it. |
| Sign is unresolved | No correction; the issue is documented and warned about. |
| No replicates | Intervals are empirical OOF intervals, explicitly approximate; no claim of experimental uncertainty. |
| Missing metadata | Listed in this document and in the model card limitations. |

## 10. Reproducing the audit

```bash
python -m src.data.audit
python -m src.models.reproducibility
pytest tests/test_audit.py tests/test_schema.py
```

Any change to a source value will fail the table comparison, change the dataset
hash, and fail the reproducibility test until the audit, baseline, and docs are
updated deliberately.
