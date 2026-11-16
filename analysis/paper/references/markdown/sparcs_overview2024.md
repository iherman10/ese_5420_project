> **Source:** New York State Department of Health. Hospital Inpatient Discharges (SPARCS De-Identified): 2024 — Overview. https://health.data.ny.gov/api/views/sf4k-39ay/files/4569ea58-f227-4251-8d10-869f070c0444?download=true&filename=NYSDOH_SPARCS_De-Identified_Overview_2024.pdf
>
> Published on Health Data NY under the [OPEN-NY Terms of Use](https://data.ny.gov/download/77gx-ii52/application/pdf). Original PDF: `../pdf/sparcs_overview2024.pdf`.
>
> **Changes:** converted from the published PDF to Markdown by Mathpix OCR; may contain conversion errors. Not otherwise modified.

---

# Hospital Inpatient Discharges <br> (SPARCS De-Identified): 2024 

## OVERVIEW

## Office of Health Services Quality and Analytics <br> Center for Health Data Innovation

October 2025

Health Data NY

## General Description

New York State Planning and Research Cooperative System (SPARCS) is a comprehensive data reporting system established in 1979 as a result of cooperation between the health care industry and government. Initially created to collect information on discharges from hospitals, SPARCS currently collects patient level detail on patient characteristics, diagnoses and treatments, services, and charges for every hospital discharge, ambulatory surgery, outpatient services and emergency department visit in New York State.

The enabling legislation and regulations for SPARCS are located under Section 28.16 of the Public Health Law (PHL), Section 400.18 of Title 10 (Health) of the Official Compilation of Codes, Rules, and Regulations of the State of New York (NYCRR).

More information on SPARCS may be found on the New York State Department of Health's website at: http://www.health.ny.gov/statistics/sparcs/.

A goal of the Department is to provide the public with easy-to-use tools to answer health care questions while protecting the privacy of the individual patient. Protecting the privacy of the patient in the collected data is a primary function of the Department of Health SPARCS program staff. SPARCS has assigned personnel whose responsibility is the monitoring of the release and handling of the data.

SPARCS utilizes information technology protections for the collection, storage, and release of data.

SPARCS data files are available with and without identifying data elements. A multi-tiered approach in the release of data is utilized to provide patient privacy while still providing useful health care data for the public and researchers. While there are three levels of data release - de-identified, limited, and identifiable - this document describes the de-identified release.

## Software Versions

| Discharge Year | CCSR Version |  | APR-DRG Version |
| :--- | :--- | :--- | :--- |
|  | Diagnosis (ICD-10-CM) | Procedure (ICD-10-PCS) | For ICD-10-CM/PCS |
| 2018 | 2021.2 | 2021.1 | 35 |
| 2019 | 2021.2 | 2021.1 | 36 |
| 2020 | 2021.2 | 2021.1 | 37 |
| 2021 | 2023.1 | 2023.1 | 38 |
| 2022 | 2023.1 | 2023.1 | 39 |
| 2023 | 2025.1 | 2025.1 | 40 |
| 2024 | 2025.1 | 2025.1 | 41 |

## Software Details

- CCSR - Clinical Classifications Software Refined (https://hcup- us.ahrq.gov/toolssoftware/ccsr/ccs refined.jsp)
- ICD-10-CM - International Classification of Diseases, 10th Revision, Clinical Modification
- ICD-10-PCS - International Classification of Diseases, 10th Revision, Procedure Coding System
- APR DRG - All Patient Refined Diagnosis Related Groups (https://www.solventum.com/en-us/home/health-information-technology/solutions/apr-drg/ )

## Data Methodology

Any facility certified to provide Article 28 inpatient services, ambulatory surgery services, emergency department services or outpatient services is required to submit data to SPARCS.

Submitting facilities include New York State Hospitals and Diagnostic and Treatment Centers (D\&TCs, commonly known as clinics). This includes both hospital-owned and operated, as well as free-standing D\&TC facilities. Regardless of their ownership, each facility must report data for each specific facility.

More information on how SPARCS data is collected may be found at:
http://www.health.ny.gov/statistics/sparcs/

## Costing Methodology

Estimates of inpatient costs were calculated using hospital discharge data from SPARCS and Institutional Cost Report (ICR) data. ICRs include data on cost for each facility as well as ratios of Cost to Charges (RCCs). RCCs are certified, calculated and reported by facilities and are subject to external audit. For example, if hospital charge is \$20,000 and the RCC is 50\%, the estimated cost is \$10,000. As with charges, cost data are hospital-specific.

Estimated total cost for discharges per calendar year was calculated using the following facility-specific audited RCCs file.

| Discharge Year | RCC Version (Audited) |
| :--- | :--- |
| 2024 | 2021 |

## De-identification

The de-identified data file contains basic record level detail; it does not contain data that is protected health information (PHI) under Health Insurance Portability and Accountability Act (HIPAA). The health information contained in the file is not individually identifiable; all data
elements that are considered identifiable have been redacted. The de-identified data file is public under the Freedom of Information Law (FOIL). The de-identified dataset allows the public to answer such questions as the number of hospitalizations that occur in the State annually or the annual number of children born in the State's hospitals.

## Limitations of Use

The de-identified data file contains basic record-level detail. It does not contain data that is protected health information (PHI) under HIPAA. The health information is not individually identifiable; all data elements that are considered identifiable have been redacted.

In cases where the RCC is over the Medicare ceiling and no reasonable or logical explanation is provided, values are overwritten by the hospital average RCC value.

The de-identified data file contains facilities that while Article 28 certified have the majority of their discharges related to rehabilitation or life-support. These facilities are listed below.

| Facility ID | Facility Name |
| :--- | :--- |
| 000340 | United Memorial Medical Center Bank Street Campus |
| 000775 | Helen Hayes Hospital |
| 000831 | Sunnyview Hospital and Rehabilitation Center |
| 000968 | Garnet Health Medical Center - Catskills - G. Hermann Site |
| 001046 | Winifred Masterson Burke Rehabilitation Hospital |
| 001138 | Blythedale Children's Hospital |
| 001460 | New York Eye and Ear Infirmary of Mount Sinai |
| 001486 | Henry J. Carter Specialty Hospital |
| 009431 | St. Peter's Addition Recovery Center |
| 010223 | Calvary Hospital |

## Data Set is Subject to Change

The current data set is a provisional data set and could be updated in the future. For more information about compliance and applicable reporting requirements visit:
https://www.health.ny.gov/statistics/sparcs/

## Contact Information

New York State Department of Health
Office of Health Services Quality and Analytics
Empire State Plaza
Corning Tower, Room 1938
Albany, New York 12237

| Phone: | (518) 473-8144 |
| :--- | :--- |
| Fax: | (518) 486-3518 |
| E-mail: | SPARCS.submissions@health.ny.govSPARCS.requests@health.ny.gov |

