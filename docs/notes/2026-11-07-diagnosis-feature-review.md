# Diagnosis Feature Review
Created: 2026-11-07
Scope: How our features compare with the reference studies, and whether using a discharge-coded diagnosis is fair. This is a research summary, and nothing here changes the model yet.

## Summary

Keeping diagnosis is defensible, and our features are stricter than the published SPARCS models. Some AUC inflation is likely, but none of these studies measures it for a model like ours.

## How the studies compare

| Study | Target | Clinical information used | Result |
| :--- | :--- | :--- | :--- |
| **Ours** | Over 7 days, 23% long | Discharge-coded principal diagnosis, MDC, medical/surgical | AUC 0.773 |
| [Rajkomar 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6550175/) | 7+ days, 22–24% long | Full EHR up to prediction time, including labs, vitals, orders and diagnoses | AUC 0.80–0.81 at admission, 0.85–0.86 at 24 h |
| Rajkomar's logistic baseline | Same | Age, gender, condition categories, admission source, service, 24 labs | AUC 0.74–0.76 at 24 h |
| [Zeleke 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10426288/) | 6+ days, 61% long | Triage only: chief complaint, acuity, arrival mode, specialty | AUC 0.754 |
| [Jain 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11288104/) (SPARCS) | 5 stay-length bins | Diagnosis, plus procedure, DRG, severity and disposition | Multiclass AUC 0.78, not comparable |
| [Reddad 2026](https://thesai.org/Downloads/Volume17No3/Paper_86-Machine_Learning_Application_in_Healthcare.pdf) (SPARCS) | Days | Diagnosis, DRG, MDC, mortality risk, disposition, plus costs and charges | R² 0.87, not comparable |
| [GitHub project](https://github.com/annnova2007/hospital-length-of-stay-prediction) (SPARCS, student work) | Days | Diagnosis only, called a proxy for the working diagnosis | R² 0.215, not comparable |

[Szydziak & Sheather 2018](https://ww2.amstat.org/meetings/proceedings/2018/data/assets/pdf/866999.pdf) used stay length as a predictor rather than a target, so it doesn't apply.

Rajkomar has our target and a similar base rate, but it uses different hospitals, and the main text doesn't say which diagnosis codes fed its stay-length model.

## Evidence on discharge-coded diagnosis

**For keeping it**

- Every model above uses some clinical information. Even triage-only Zeleke has the chief complaint.
- In [Hautz 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6505221), the ER and discharge diagnoses differed substantially in 12.3% of 755 patients.
- [Ramadan 2025](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12743287/) looked only at mortality, and its top predictors were codes like cardiac arrest and brain death. It doesn't address principal diagnosis or length of stay, and it never compared models with and without the codes.

**For caution**

- Patients whose diagnosis changed stayed longer: 10.3 vs. 6.9 days in Hautz, and 4.2 vs. 3.4 days in [Johnson 2009](https://shmpublications.onlinelibrary.wiley.com/doi/10.1002/jhm.453). Johnson's 75.6% mismatch rate comes from exact-code matching, so it overstates clinical disagreement.
- Our model therefore sees the final answer most often where the early picture was wrong. That suggests some upward bias, but neither study measures AUC. This is our inference.
- Ramadan's timing argument still applies: codes finalized at discharge aren't available at admission.

## Open questions

- **Rerun without diagnosis:** fit the baseline without `ccsr_diagnosis_code`, `apr_mdc_code` and `apr_medical_surgical_description`. The two AUCs would bracket a real admission-day model. This doesn't isolate the inflation.
- **Medical/surgical flag:** we excluded procedures as leakage, but this flag may partly reveal whether an operation happened. MDC might also need a look, since the GitHub project dropped it as post-admission. This is our inference, not a study finding.
- **Notebook wording:** the baseline notebook says discharge coding makes scores "somewhat better", which states an inference as fact. It should be softened.

## How far to trust this

Sub-agents did the reading, mostly through a summarizing tool, so check any quotes against the originals. Rajkomar's supplement couldn't be opened, and Johnson's limitations section wasn't read.
