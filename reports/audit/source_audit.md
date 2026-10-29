# Source audit

Source: Experimental Studies of Stellite-6 Hardfaced Layer on Ferrous Materials by TIG Surfacing Process (IOP Conference Series: Materials Science and Engineering 998 (2020) 012061), DOI 10.1088/1757-899X/998/1/012061, licensed CC BY 3.0.

## Result

- Table comparison: **pass** (24 of 24 rows checked, 0 mismatches).
- Exposure-time multiples: **warn** (11 exact, 1 near; maximum relative deviation 7.75e-04).
- Oxidation unit: **warn** (unresolved).
- Specimen dimensions: **warn** (methods and Table 7 disagree by a factor of 4).
- Oxidation sign: **warn** (unresolved).
- Observation independence: **warn** (effective n = 6 for oxidation).
- Heat-input arithmetic: **pass** (max difference 5.48e-04 kJ/mm).

## Table comparison

Every CSV value and every `Source_Table` label was compared to the corresponding
entry in Tables 5, 6, and 8-13. See `reports/audit/source_audit.json` for the
row-level result.

## Exposure-time multiples

| Material | Current (A) | Exposure (h) | Value | Ratio to 5 h | Classification |
|---|---:|---:|---:|---:|---|
| EN-8 | 120 | 10 | 0.005206 | 2.000000 | exact_to_printed_precision |
| EN-8 | 120 | 15 | 0.007809 | 3.000000 | exact_to_printed_precision |
| EN-8 | 140 | 10 | 0.004592 | 2.000000 | exact_to_printed_precision |
| EN-8 | 140 | 15 | 0.006888 | 3.000000 | exact_to_printed_precision |
| EN-8 | 160 | 10 | 0.004326 | 2.000000 | exact_to_printed_precision |
| EN-8 | 160 | 15 | 0.006489 | 3.000000 | exact_to_printed_precision |
| Mild Steel | 120 | 10 | 0.005384 | 2.000000 | exact_to_printed_precision |
| Mild Steel | 120 | 15 | 0.008076 | 3.000000 | exact_to_printed_precision |
| Mild Steel | 140 | 10 | 0.005156 | 1.998450 | near_multiple |
| Mild Steel | 140 | 15 | 0.007740 | 3.000000 | exact_to_printed_precision |
| Mild Steel | 160 | 10 | 0.004474 | 2.000000 | exact_to_printed_precision |
| Mild Steel | 160 | 15 | 0.006711 | 3.000000 | exact_to_printed_precision |

Eleven of twelve non-base rows reproduce k x the 5 h value to the last printed digit (k = 2 for 10 h, 3 for 15 h); one row deviates by 0.08 %. This is consistent with the 10 h/15 h rows being derived from a single rate per condition, or with implausibly exact linear kinetics.

## Oxidation unit

The paper prints `weight gain (g) / surface area (mm2)` with a stated area of 225 mm2.

| Reading | Implied quantity at the smallest value | Implied quantity at the largest value |
|---|---:|---:|
| Values are g/mm2 | 0.487 g mass change (4.6 % of specimen mass) | 1.817 g (17.1 %) |
| Values are grams | 0.96 mg/cm2 | 3.59 mg/cm2 |

If the tabulated numbers are g/mm2 as printed, the implied mass change is 0.487-1.817 g, or 4.6-17.1 % of a 15 x 15 x 6 mm steel specimen, which is not physically plausible for a 5-15 h exposure at 700 C. If the numbers are raw weight change in grams, the areal gain is 0.96-3.59 mg/cm2, which is plausible. The unit is therefore unresolved; the column name follows the paper's label and the absolute scale should not be treated as g/mm2 without the raw weights.

The printed magnitudes are more consistent with raw mass in grams than with
g/mm2. Under 225 mm2 they imply 0.96-3.59 mg/cm2; under the 900 mm2 area stated in the methods they imply 0.24-0.90 mg/cm2. No correction is applied.

## Specimen dimensions

Methods statement: `Samples sizes of 30X30mm2 were cut from hardfaced plate`.
Table 7 statement: `Sample size 15x15x6mm; surface area = 15x15mm2`.

The methods statement implies 900 mm2 while Table 7 uses 225 mm2, a factor-of-four disagreement. The project uses the Table 7 area for normalization and records the conflict rather than choosing silently.

## Sign convention

Printed equation: `weight gain = w1 (before oxidation) - w2 (after oxidation)`. Table 7 defines weight gain as w1 (before) minus w2 (after). For oxidation the specimen usually gains mass (w2 > w1), which would make this quantity negative, yet every tabulated value is positive. Either the equation is transposed or the specimens lost mass. The raw weights are not published, so the sign cannot be resolved. Values are preserved exactly as printed and no sign correction is applied.

## Independence

Hardness: 6 rows, 6 conditions, one value per condition.

Oxidation: 18 rows, 6 conditions, three exposure rows per condition.

Each material/current condition contributes one reported hardness value and three reported oxidation values. No replicate specimens, repeated readings, or uncertainty estimates are published, and specimen reuse across exposure times is not stated. The paper says three samples were kept in the furnace per run, which is consistent with one specimen at each current rather than triplicate specimens. Treating all 18 oxidation rows as independent would overstate the information by roughly a factor of three.

## Process arithmetic and chemistry

| Current (A) | Speed (mm/s) | Printed Q (kJ/mm) | Computed Q (kJ/mm) |
|---:|---:|---:|---:|
| 120 | 2.25 | 0.800 | 0.800 |
| 140 | 2.43 | 0.864 | 0.864 |
| 160 | 2.63 | 0.912 | 0.913 |

Q = 15 V * I / v reproduces the printed schedule without an arc-efficiency factor. Current, travel speed, and heat input therefore form a single one-dimensional schedule and cannot be separated as independent factors.

EN-8 carbon and manganese are midpoints of the reported ranges; mild-steel values are single reported values. Chemistry is metadata and is not a model feature because it is tied to substrate identity.

## Unresolved issues

1. Oxidation unit scale (g/mm2 vs grams) - needs the raw weights or author clarification.
2. Oxidation sign convention - the printed equation is reversed relative to typical mass gain.
3. Whether the 10 h and 15 h rows are independent measurements or derived values.
4. Specimen reuse across exposure times.
5. Hardness test load, indentation count, and replicate count.

These are documented in `docs/SOURCE_AUDIT.md` and `data/README.md`. They are not
resolved by assumption anywhere in the model pipeline.

## Unreported metadata

- Vickers test load (the paper only says 'a range of light loads')
- Number and location of hardness indentations per specimen
- Number of replicate specimens or replicate measurements per condition
- Raw specimen weights w1 and w2 (only the derived values are published)
- Balance resolution and weighing procedure
- Measured exposed area (225 mm2 is taken from the nominal 15 x 15 mm size)
- Furnace atmosphere and heating/cooling rates
- Travel-speed measurement uncertainty (measured with a stopwatch)
- Deposit dilution, overlay thickness measurement, and coating/substrate interface characterization beyond 100x optical microscopy
- Specimen identifiers (the project assigns them; they are not in the paper)

## Dataset hashes

| File | SHA-256 |
|---|---|
| data/hardness.csv | `1d83be25aa4b59453bc24b816c7f6267be0e0b7480bc8fc2a394fba9d6745a60` |
| data/oxidation.csv | `be5edafcebde4ada00ba3578df623b84953f79ed030f109f20cfdf9ade6f820b` |
| data/source_tables.json | `a444175f2ed578396b6a1d1a25de6cfd6553e4b45fcb799a9e0d1aeda8953bb0` |
| source_pdf_author_supplied | `07f3f8f94732cdf908ff8ce1cacf20edb107edd82a47be21c60073a3712ca06f` |
| source_pdf_retrieved | `7c199106eb974de26d0d806a6d2f5531dd1128bacd28cfbd618020ec195cb6be` |

