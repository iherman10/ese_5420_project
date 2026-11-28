> **Source:** New York State Department of Health. Hospital Inpatient Discharges (SPARCS De-Identified): 2024 — Data Dictionary. https://health.data.ny.gov/api/views/sf4k-39ay/files/a768e7f9-db7a-42ce-bedc-e2c88186c687?download=true&filename=NYSDOH_SPARCS_De-Identified_Data_Dictionary_2024.pdf
>
> Published on Health Data NY under the [OPEN-NY Terms of Use](https://data.ny.gov/download/77gx-ii52/application/pdf). Original PDF: `../pdf/sparcs_dictionary2024.pdf`.
>
> **Changes:** converted from the published PDF to Markdown by Mathpix OCR; may contain conversion errors. Not otherwise modified.

---

# Hospital Inpatient Discharges <br> (SPARCS De-Identified): 2024 

## DATA DICTIONARY

Office of Health Services Quality and Analytics
Center For Health Data Innovation

October 2025

Health Data NY

| Field Name | Definition |
| :--- | :--- |
| Health Service Area | Type is Char. Length is 19. A description of the Health Service Area (HSA) in which the hospital is located. Blank for enhanced de-identification records. Capital/Adirondack, Central NY, Finger Lakes, Hudson Valley, Long Island, New York City, Southern Tier, Western NY. <br> https://regs.health.ny.gov/content/section-82-12-health-service-areas |
| Hospital County | Type is Char. Length is 15. A description of the county in which the hospital is located. Blank for records with enhanced de-identification. |
| Operating Certificate Number | Type is Char. Length is 12. The facility Operating Certificate Number assigned by NYS Department of Health. Blank for records with enhanced de-identification. |
| Permanent Facility Id | Type is Num. Length is 6. Permanent Facility Identifier. Blank for records with enhanced de-identification. |
| Facility Name | Type is Char. Length is 112. The name of the facility where services were performed based on the Permanent Facility Identifier (PFI), as maintained by the NYSDOH Division of Health Facility Planning. For records with enhanced de-identification, 'Redacted for Confidentiality' is noted. |
| Age Group | Type is Char. Length is 11. Age in years at time of admission. <br> Grouped into the following age groups: 0 to 17, 18 to 29, 30 to 49, 50 to 69, and 70 or Older. |
| Zip Code - 3 digits | Type is Char. Length is 3. The first three digits of the patient's zip code. Blank for: <br> population size less than 20,000 enhanced de-identification records, or cell size less than 10 on population classification strata. "OOS" are Out of State zip codes. |
| Gender | Type is Char. Length is 1. Patient gender: (M) Male, (F) Female, (U) Unknown. |
| Race | Type is Char. Length is 32. Patient race. <br> Black/African American, Multi-racial, Other Race, White. Other Race includes Native Americans and Asian/Pacific Islander. |
| Ethnicity | Type is Char. Length is 20. Patient ethnicity. <br> The ethnicity of the patient: Spanish/Hispanic Origin, Not of Spanish/Hispanic Origin, Multi-ethnic, Unknown. |
| Length of Stay | Type is Char. Length is 5. The total number of patient days at an acute level and/or other than acute care level (excluding leave of absence days) (Discharge Date - Admission Date) + 1. <br> Length of Stay greater than or equal to 120 days has been aggregated to 120+ days. |
| Type of Admission | Type is Char. Length is 15. A description of the manner in which the patient was admitted to the health care facility: <br> Elective, Emergency, Newborn, Not Available, Trauma, Urgent. |
| Patient Disposition | Type is Char. Length is 37. The patient's destination or status upon discharge. |
| Discharge Year | Type is Char. Length is 4. The year (CCYY) of discharge. |
| CCSR Diagnosis Code | Type is Char. Length is 10. AHRQ Clinical Classification Software Refined (CCSR) ICD-10-CM Diagnosis Category Code. <br> More information on the CCSR system may be found at the direct link: <br> https://www.hcup-us.ahrq.gov/toolssoftware/ccsr/ccs refined.jsp |


| Field Name | Definition |
| :--- | :--- |
| CCSR Diagnosis Description | Type is Char. Length is 250. AHRQ Clinical Classification Software Refined (CCSR) ICD-10-CM Diagnosis Category Description. <br> More information on the CCSR system may be found at the direct link: https://www.hcup-us.ahrq.gov/toolssoftware/ccsr/ccs refined.jsp |
| CCSR Procedure Code | Type is Char. Length is 10. AHRQ Clinical Classification Software Refined (CCSR) ICD-10-PCS Procedure Category Code. More information on the CCSR system may be found at the direct link: <br> https://www.hcup-us.ahrq.gov/toolssoftware/ccsr/ccs refined.jsp |
| CCSR Procedure Description | Type is Char. Length is 250. AHRQ Clinical Classification Software Refined (CCSR) ICD-10-PCS Procedure Category Description. More information on the CCSR system may be found at the direct link: <br> https://www.hcup-us.ahrq.gov/toolssoftware/ccsr/ccs refined.jsp |
| APR DRG Code | Type is Char. Length is 3. The All Patients Refined Diagnosis Related Groups (APR-DRG) Classification Code |
| APR DRG Description | Type is Char. Length is 500. The APR-DRG Classification Code Description. |
| APR MDC Code | Type is Char. Length is 2. All Patient Refined Major Diagnostic Category (APR MDC) Code. APR-DRG Codes 001-006 and 950-956 may group to more than one MDC Code. All other APR DRGs group to one MDC category. |
| APR MDC Description | Type is Char. Length is 500. All Patient Refined Major Diagnostic Category (APR MDC) Description. |
| APR Severity of Illness Code | Type is Char. Length is 1. The APR-DRG Severity of Illness Code: 0, 1, 2, 3, 4 |
| APR Severity of Illness Description | Type is Char. Length is 12. All Patient Refined Severity of Illness (APR SOI) Description: <br> Undetermined (0), Minor (1), Moderate (2), Major (3), Extreme (4). |
| APR Risk of Mortality | Type is Char. Length is 12. All Patient Refined Risk of Mortality (APR ROM) Description: <br> Undetermined (0), Minor (1), Moderate (2), Major (3), Extreme (4). |
| APR Medical Surgical Description | Type is Char. Length is 14. The APR-DRG specific classification of Medical, Surgical or Not Applicable. |
| Payment Typology 1 | Type is Char. Length is 25. A description of the source of payment provided for the discharge. |
| Payment Typology 2 | Type is Char. Length is 25. A description of the source of payment provided for the discharge. |
| Payment Typology 3 | Type is Char. Length is 25. A description of the source of payment provided for the discharge. |
| Birth Weight | Type is Char. Length is 5. The neonate birth weight in grams; rounded to nearest 100 g (gram). |
| Emergency Department Indicator | Type is Char. Length is 1. The Emergency Department Indicator is set based on the submitted revenue codes. If the record contained an Emergency Department revenue code of 045X series, the indicator is set to "Y" for Yes, otherwise it will be "N" for No. |
| Total Charges | Type is Char. Length is 11. Total charges for the discharge. |
| Total Costs | Type is Char. Length is 10. Total estimated cost for the discharge. |

