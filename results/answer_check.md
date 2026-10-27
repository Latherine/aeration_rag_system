# Gate 4: end-to-end answers, for checking by hand

Run 2026-10-27: `qwen2.5:3b-instruct-q4_K_M`, dense retrieval, k=3. 12 questions: every third development question plus two the papers cannot answer. Fill in the *Correct?* column by reading each answer against its cited pages; no automatic judge is used.

- Time per answer: median **31.5 s**, worst 40.9 s (median prompt 1012 tokens; 6 cache-served runs excluded)
- Correct passage among the excerpts sent: 7 of 10
- Answers with at least one citation: 6 of 12
- NOT_IN_CORPUS: 2 of 2 unanswerable questions, 1 of 10 answerable ones
- Citations to excerpts that were never sent: 0

| Question | Gold sent | Cited | Seconds | Correct? |
|---|---|---|---|---|
| q01 (shared) | yes | [1] | 31 | |
| q04 (abbreviation) | yes | — | 25 | |
| q07 (shared) | **no** | — | 17 (cached) | |
| q10 (paraphrase) | yes | [3] | 20 (cached) | |
| q13 (shared) | **no** | [2] | 15 (cached) | |
| q16 (shared) | yes | [3] | 15 (cached) | |
| q19 (abbreviation) | yes | [1] | 41 | |
| q22 (abbreviation) | yes | [1] | 14 (cached) | |
| q25 (paraphrase) | **no** | — | 32 | |
| q28 (paraphrase) | yes | — | 6 (cached) | |
| x01 (out_of_corpus) | n/a | — | 25 | |
| x02 (out_of_corpus) | n/a | — | 36 | |

## q01: What does the ammonia valley in the pH profile indicate?

The ammonia valley in the pH profile indicates the completion of a nitrification process. [1]

- [1] `abuhasan2010:016` pp.5-6, 3. Results and Discussion
- [2] `abuhasan2010:017` p.6, 3. Results and Discussion
- [3] `abuhasan2010:015` p.5, 3. Results and Discussion

## q04: What solids retention time is recommended for stable nitrification?

15–20 d

- [1] `sakthivel2025:036` p.6, 3.2.1. Solids retention time
- [2] `sakthivel2025:035` p.6, 3.2.1. Solids retention time
- [3] `sakthivel2025:037` p.6, 3.2.1. Solids retention time

## q07: Why was no ammonia removed when the filter was run without aeration?

NOT_IN_CORPUS

- [1] `abuhasan2010:001` p.1, front matter
- [2] `abuhasan2010:003` pp.1-2, 1. Introduction
- [3] `abuhasan2010:002` p.1, 1. Introduction

## q10: Why is ammonia in drinking water harmful to people?

Ammonia in drinking water exceeding the Malaysian regulated limit (<1.5 mg/L) becomes a major problem to drinking water treatment plants (DWTPs). Moreover, high ammonia level in drinking water provided a complication on chlorination process due to the yield of chloramines, human nervous system damage and created a deteriorating taste and odour of water. Therefore, it is necessary to develop an effective additional water treatment system, typically for ammonia removal in order to prevent the plant shutdown due to the high level of ammonia in water. [3]

- [1] `abuhasan2010:000` p.1, front matter
- [2] `abuhasan2010:001` p.1, front matter
- [3] `abuhasan2010:002` p.1, 1. Introduction

## q13: Which simulation software was used to model the sequencing batch reactor?

GPS-X was used to model the sequencing batch reactor. [2]

- [1] `sakthivel2025:015` p.3, 1. Introduction
- [2] `sakthivel2025:011` p.2, 1. Introduction
- [3] `sakthivel2025:014` p.3, 1. Introduction

## q16: How well did the model predict the independent Plant B dataset?

[3] Table 5 compiles validation metrics against an independent Plant B dataset. All R2 ≥0.85 and NSE ≥0.70 confirm robust prediction of NH3–N, TN, TP, and COD under low/high load and elevated temperature.

- [1] `sakthivel2025:061` p.9, 4.4. Model robustness and practical applicability
- [2] `sakthivel2025:024` p.4, 2.9. Data management and statistical analysis
- [3] `sakthivel2025:044` p.7, 3.4. Model validation [tables/figures]

## q19: How does solids retention time affect ammonium left in the effluent?

[1] Solids retention time (SRT) affects the amount of ammonium (NH4+-N) left in the effluent by influencing the establishment and stability of nitrifier population. As SRT increases from 5 days to approximately 15 days, the removal of NH4+-N concentration falls steeply, reaching low residual values by the 12-15 days range. This indicates that longer SRT leads to a more stable and established nitrifier population, which helps in reducing NH4+-N in the effluent. Beyond 15 days, the removal of NH4+-N shows diminishing returns.

- [1] `sakthivel2025:036` p.6, 3.2.1. Solids retention time
- [2] `abuhasan2010:013` pp.4-5, 3. Results and Discussion
- [3] `sakthivel2025:035` p.6, 3.2.1. Solids retention time

## q22: What are the Iraqi effluent limits for chemical oxygen demand and biochemical oxygen demand?

[1] Iraqi effluent limits for COD = 100 mg/L, BOD = 40 mg/L, and TSS = 60 mg/L as specified in the Iraqi Environmental Standards, Contract No.: W3QR-50-M074, Rev. No.: 03 Oct 2011 (Ismael and Aziz, 2024).

- [1] `hasan2026:012` p.3, D. BioWin Modeling
- [2] `hasan2026:024` pp.6-7, A. Dynamic Simulation of Duhok SBRs with no EQ Basin Aeration (First Run)
- [3] `hasan2026:013` pp.3-4, D. BioWin Modeling

## q25: Did using less air help or hurt nitrogen and phosphorus removal?

Using less air helped with nitrogen removal, as intermittent aeration (C5) and combined pre + post anoxic (C6) configurations achieved >75 % TN removal with energy savings compared to continuous high-DO operation (C4). Continuous high-DO operation solved ammonia but trapped nitrogen as nitrate, indicating that less air helped in achieving higher TN removal.

- [1] `sakthivel2025:031` p.5, 3.1.3. Operational implications
- [2] `sakthivel2025:028` p.5, 3.1.1. Key patterns and contrasts
- [3] `sakthivel2025:059` p.9, 4.2. Trade-offs between energy use and removal efficiency

## q28: What could plant staff do with the hours freed up by a shorter cycle?

Plant staff could use the hours freed up by a shorter cycle to increase settling time or reduce cycle time, thus treating a greater quantity of wastewater per day, or have more plant shutdowns for applying preventive maintenance.

- [1] `hasan2026:010` pp.2-3, D. BioWin Modeling
- [2] `hasan2026:005` pp.1-2, I. Introduction
- [3] `hasan2026:034` p.10, B. Dynamic Simulation of Duhok SBRs with Optimized Aeration Time (Final Run)

## x01: How does chlorine dose affect trihalomethane formation in drinking water?

NOT_IN_CORPUS

- [1] `abuhasan2010:006` p.2, 2. Materials and Methods
- [2] `abuhasan2010:013` pp.4-5, 3. Results and Discussion
- [3] `abuhasan2010:010` pp.3-4, 2. Materials and Methods

## x02: What does a UV disinfection system cost to install per cubic metre of capacity?

NOT_IN_CORPUS

- [1] `abuhasan2010:007` pp.2-3, 2. Materials and Methods
- [2] `abuhasan2010:009` p.3, 2. Materials and Methods
- [3] `abuhasan2010:008` p.3, 2. Materials and Methods
