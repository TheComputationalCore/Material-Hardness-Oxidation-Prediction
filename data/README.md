# Dataset provenance and units

The checked-in training rows below are transcribed from the experimental tables in:

> C. Dinesh Chandra, B. Rushikesh, Mohammed Numan, and B. Venkatesh, “Experimental Studies of Stellite-6 Hardfaced Layer on Ferrous Materials by TIG Surfacing Process,” IOP Conference Series: Materials Science and Engineering 998 (2020), 012061. DOI: [10.1088/1757-899X/998/1/012061](https://doi.org/10.1088/1757-899X/998/1/012061).

## Files

- hardness.csv contains the six reported substrate/current measurements from Tables 5 and 6.
- oxidation.csv contains the eighteen reported substrate/current/exposure measurements from Tables 8–13.

Data_Status marks all training rows as reported_paper_measurement. Training scripts reject any row with another status. Do not add interpolated values as if they were independent experiments. If derived or newly measured records are added later, retain a status and source reference for each row and evaluate them separately.

## Inputs and experimental scope

- Substrates: EN-8 and Mild Steel.
- Welding currents: 120, 140, and 160 A.
- The paper's corresponding heat inputs are 0.800, 0.864, and 0.912 kJ/mm. Current, travel speed, and heat input vary together in the reported design; they are not independent experimental factors.
- Oxidation exposure durations: 5, 10, and 15 hours at 973 K (700 °C).
- Carbon and manganese columns record representative substrate chemistry stated in the paper. They are metadata and are not used as independent model features.

## Targets

- Hardness is reported as a Vickers hardness number. The paper does not state the test load alongside the tabulated values.
- Areal_Mass_Gain_g_mm2 is the paper's reported weight change divided by exposed area, in g/mm². It is cumulative mass gain per area, not divided by exposure time. The paper prints the subtraction order as pre-test weight minus post-test weight; verify that sign against the original lab records before making any reinterpretation.

The paper does not report replicate-level observations or uncertainty estimates for these table entries. These datasets therefore represent reported table values, not a replicate-rich experimental archive.
