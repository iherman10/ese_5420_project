# Psychiatric Stays and Declining Per Diem Payments

*Researched 2026-10-15 from published rules, agency reports and literature (web research only, no data analysis). Optional side angle for the hospital-stays project.*

## Summary

- **The question.** A contact in NYC government raised it: psychiatric hospital stays are paid at a per-day rate that declines as the stay goes on. Do hospitals discharge psychiatric patients earlier because of it?
- **Is it real?** Yes. Medicare and NY Medicaid both pay inpatient psychiatric care with a **variable (declining) per diem adjustment**: a multiplier on the daily rate that is higher in the first days of the stay and steps down later. The drops are modest (roughly 17–33% over the first one to two weeks), not a cliff to 50% or 20%. Insurer **concurrent review** (stopping payment for further days) is a separate, sharper mechanism.
- **Does it change discharge timing?** Not much, probably.
  - Hospitals clearly time discharges around *large, sudden* payment jumps.
  - Studies of *gradual* psychiatric per diem tapers in Europe found little or no effect.
  - No one has tested this with NY SPARCS data.
- **Fit with our project.** It works as an optional exploratory figure or appendix: length-of-stay histograms for psychiatric stays, by payer, looking for spikes at the days where the multiplier changes. It's descriptive, not causal. The classifier itself can't answer it.

## The per diem adjustments

### NY Medicaid: psychiatric "exempt units" in general hospitals

Since October 2010, NY Medicaid fee-for-service has paid psychiatric units inside general (Article 28) hospitals a per diem. That per diem is built from APR-DRG psychiatric weights and adjusted for age, diagnosis, comorbidities and **length of stay**. The 2013 Medicaid Update gives this length-of-stay factor:

| Day of stay | Multiplier | Change |
| :--- | :--- | :--- |
| 1–4 | 1.20 | |
| 5–11 | 1.00 | −17% at day 5 |
| 12–22 | 0.96 | −4% at day 12 |
| 23+ | 0.92 | −4% at day 23 |

- The statewide base price was $950.43 per day (SPA NY-22-0047). **Unconfirmed:** that these factors are still current. Check the DOH rate worksheets.
- Applies to Medicaid fee-for-service. Medicaid managed-care plans negotiate their own rates.
- These are psychiatric units in general hospitals, which is what SPARCS covers.

### Medicare: Inpatient Psychiatric Facility Prospective Payment System (IPF PPS)

Medicare pays a per diem base rate × a **variable per diem adjustment** by day of stay. CMS's rationale is that costs are highest at admission and stabilization. The adjustment covers freestanding psychiatric hospitals and psychiatric units in general hospitals.

| Day | Old schedule (FY2005–FY2024) | Current schedule (FY2025 onward) |
| :--- | :--- | :--- |
| 1, hospital with an emergency department | 1.31 | 1.54 |
| 1, no emergency department | 1.19 | 1.28 |
| 2 | 1.12 | 1.20 |
| 3 | 1.08 | 1.15 |
| 4 | 1.05 | 1.12 |
| 5 | 1.04 | 1.08 |
| 6 | 1.02 | 1.06 |
| 7 | 1.01 | 1.03 |
| 8 | 1.01 | 1.02 |
| 9 | 1.00 | 1.01 |
| 10 | 1.00 | 1.00 (flat from here) |
| 11–21 | declines gradually to 0.95 | 1.00 |
| 22+ | 0.92 | 1.00 |

- **Unverified:** the old schedule's day 4–21 values are from secondary sources. Check them against a CMS FY2022 Addendum A before citing.
- **Relevant to our data:** Medicare's fiscal year starts October 1, so almost all 2023 and 2024 discharges fall under the **old** schedule. Only Oct–Dec 2024 discharges use the new one. Under the old schedule, the clearest change is the step to 0.92 at **day 22**.
- Base rate: $876.53/day in FY2025, $892.87 in FY2026.

### Related mechanisms (not per diem adjustments)

| Mechanism | Payer | What happens | Days |
| :--- | :--- | :--- | :--- |
| Concurrent / continued-stay review | Commercial, Medicaid managed care | The insurer approves a few days at a time and then denies further days, so payment stops | Varies |
| NY limits on that review | NY-regulated commercial plans (not self-insured employer plans) | No review allowed during an initial window | 14 (under 18, since 2020), 28 (substance use), 30 (adults, likely from May 2024) |
| IMD exclusion and the 15-day rule | Medicaid | Medicaid won't pay freestanding psychiatric hospitals ("institutions for mental disease") for adults aged 21–64. Managed care may pay for up to 15 days per month. Psychiatric units in general hospitals are exempt | 15 per month |
| 190-day lifetime limit | Medicare | Patients get 190 lifetime days in freestanding psychiatric hospitals | 190 lifetime |
| Part A coinsurance | Medicare | Patients start paying a daily copay | Day 61+ |
| Timothy's Law (NY, 2007) | NY group plans | Must cover at least 30 inpatient mental-health days per year | 30 per year |

## What the research says

**Large, sudden payment jumps do change discharge timing:**

| Study | Setting | Finding |
| :--- | :--- | :--- |
| Eliason, Grieco, McDevitt & Roberts 2018, *AER* | Medicare long-term care hospitals; a lump-sum payment kicks in at a threshold day | Daily discharge probability jumps from 1.4% to 10.2% right after the threshold (more than 7×). The effect is larger at for-profits, with no increase in deaths |
| Einav, Finkelstein & Mahoney 2018, *Econometrica* | Same setting | Large spike; the patients discharged at the threshold are healthier |
| Chatterjee, Werner et al. 2019, *JAMA Intern Med* | Medicare skilled nursing facilities; patient copays start on day 21 | About 1.7× as many discharges on day 20 as on neighbouring days |

**Gradual psychiatric per diem tapers show little effect:**

| Study | Setting | Finding |
| :--- | :--- | :--- |
| Pott et al. 2021, *Eur J Health Econ* | German psychiatric hospitals, declining per diem | **No bunching** at the points where pay drops |
| Pletscher 2016, *Eur J Health Econ* | Swiss psychiatric hospital, rate drops after day 5 | Length of stay responded only where the revenue drop was large |
| Heppa et al. 2020, *Swiss Med Wkly* | Switzerland's declining per diem (TARPSY) | Adult length of stay changed by −0.6 days, not significant |
| MedPAC / Urban Institute 2022 | US Medicare after the 2005 psychiatric payment system | Mean psychiatric stay was flat: 11.2 days (2004), 12.0 (2014) |

**Older US evidence on limits and review:**
- **State Medicaid day limits:** associated with about 30% shorter psychiatric stays (Lave & Frank 1988, which used length of stay in days only, like SPARCS).
- **Commercial review:** one insurer approved only about 6.9 of 19.0 requested days on average (Travis 1996).
- **Cut-short stays:** psychiatric patients whose stays were restricted were readmitted more often (Wickizer 1999).

**Gaps:**
- No published bunching analysis of US psychiatric discharges at payer-specific thresholds.
- Nothing using NY SPARCS.
- The closest NY analogue is NYC DOHMH Epi Data Brief 154 (April 2026), on psychiatric hospitalizations in SPARCS from 2016 to 2023. Read it first.

## Fit with the project

**Exploratory figure or appendix: yes.**

Procedure:
1. Select psychiatric stays with **APR MDC 19** (mental disorders). Analyse substance use (MDC 20) separately, since its rules differ.
2. Plot length-of-stay histograms by payer: Medicaid, Medicare, commercial.
3. Look for unusual drops or spikes at the days where payment changes:
   - **Medicaid:** days 5, 12, 23.
   - **Medicare (old schedule):** day 22.
   - **Commercial review windows:** day 14 for minors, day 30 for adults.
4. Optionally, fit a smooth curve to each histogram and measure the excess or missing mass around those days (a "bunching estimator").

The cleanest comparisons:
- **Medicaid vs. Medicare around day 5.** Medicaid steps down there, while Medicare declines smoothly.
- **Commercial vs. Medicaid minors around day 14.**

**Answering it with the trained classifier: not really.**
- The classifier predicts "over 7 days", not discharge timing on specific days.
- Payer importance in the model is an association, not hospital behaviour.
- The natural link is the fairness section: report payer calibration for psychiatric stays separately, and cite payment rules as one possible explanation for any gap.

**Caveats:**
- **Weekly discharge rhythm:** hospitals discharge more on weekdays, especially Fridays, so stay lengths show spikes every 7 days. With no dates in SPARCS, this can't be removed directly. Comparing payers against each other cancels much of it. Mind days 7, 14, 21 and 28 especially.
- **Expected result:** the tapers are gradual, so the literature predicts small or null effects. A null result is still reportable.
- **Missing facilities:** state psychiatric centers and OMH-only private psychiatric hospitals (Article 31), roughly 40–50% of NY psychiatric *beds* (a smaller share of discharges), aren't in SPARCS. Their payer mix differs.
- **Merged or unknown payment arrangements:** Medicaid fee-for-service and managed care are merged in SPARCS. Self-insured vs. NY-regulated commercial plans can't be distinguished. Private hospital–insurer contracts may have their own per diem tiers, which are confidential.
- **Split stays:** some stays are split into a medical record and a psychiatric-unit record, so psychiatric length of stay can start partway through a hospitalization.
- **Not causal:** payer groups differ in age, severity and diagnosis. Stratify by age group and diagnosis at a minimum.

## Terminology

| Term | Meaning |
| :--- | :--- |
| Per diem | Payment per day of stay |
| Variable / degressive / tapered per diem | A per-day rate that declines as the stay goes on |
| Variable per diem adjustment | Medicare's name for its day-of-stay multiplier in the psychiatric payment system |
| IPF PPS | Inpatient Psychiatric Facility Prospective Payment System, Medicare's psychiatric payment system |
| Exempt unit | A psychiatric unit inside a general hospital, paid outside the normal per-case DRG system |
| Concurrent / utilization / continued-stay review | Insurer review during the stay that can deny payment for further days |
| NQTL | Non-quantitative treatment limitation: rules like review requirements, which parity law (MHPAEA) regulates |
| IMD exclusion | Medicaid's ban on paying freestanding psychiatric hospitals ("institutions for mental disease") for adults aged 21–64 |
| Payment cliff / notch vs. kink | A sudden drop in total payment vs. a change in the per-day rate |
| Bunching / excess mass | An unusual pile-up of discharges at or just before a threshold day |
| Strategic discharge | Timing discharges to payment rules rather than clinical need |

## Key sources

- **NY Medicaid exempt-unit payment:**
  - [NYS Medicaid Update, July 2013](https://healthweb-back.health.ny.gov/health_care/medicaid/program/update/2013/2013-07.htm) (exempt-unit per diem and LOS factors)
  - [SPA NY-22-0047](https://www.medicaid.gov/medicaid/spa/downloads/NY-22-0047.pdf) (base price)
- **Medicare IPF PPS:**
  - [CMS FY2022 Addendum A](https://www.cms.gov/files/document/fy-2022-addendum-ipf-pps-final-payment-updates.pdf) (old variable per diem factors)
  - [CMS FY2025 Addendum A](https://cms.gov/files/document/fy-2025-addendum-ipf-pps-final-payment-updates.pdf) (current factors)
  - [MedPAC psychiatric payment basics](https://www.medpac.gov/wp-content/uploads/2024/10/MedPAC_Payment_Basics_25_psych_FINAL_SEC.pdf)
  - [MedPAC 2022 contractor report](https://www.medpac.gov/wp-content/uploads/2022/05/Apr22_IPF_CONTRACTOR_SEC.pdf) (LOS trends)
- **NY review rules:**
  - [NY DFS mental health and substance use rules](https://www.dfs.ny.gov/consumers/health_insurance/mhsud)
  - [OMH 14-day memo](https://omh.ny.gov/omhweb/bho/guidance-memo-on-14-day-no-um-for-under-18-inpatient.pdf)
- **Discharge-timing literature:**
  - [Eliason et al. 2018](https://www.nber.org/papers/w22598) (strategic discharge)
  - [Einav, Finkelstein & Mahoney 2018](https://www.nber.org/papers/23100)
  - [Chatterjee et al. 2019](https://jamanetwork.com/journals/jamainternalmedicine/article-abstract/2733559) (SNF day 20)
  - [Pott et al. 2021](https://pubmed.ncbi.nlm.nih.gov/33165668/) (German psychiatric per diem, no bunching)
  - [Lave & Frank 1988](https://pmc.ncbi.nlm.nih.gov/articles/PMC4192915) (Medicaid psychiatric day limits)
  - [Wickizer et al. 1999](https://pmc.ncbi.nlm.nih.gov/articles/PMC1508766) (concurrent review and readmission)
- **Method and NYC context:**
  - [Kleven 2016](https://eml.berkeley.edu/~saez/course/kleven_annualreview.pdf) (bunching estimators)
  - [NYC DOHMH Epi Data Brief 154](https://www.nyc.gov/assets/doh/downloads/pdf/epi/databrief154-mental-health-hospitalization-2026.pdf) (NYC psychiatric hospitalizations, SPARCS)
