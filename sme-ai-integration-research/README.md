# AI Integration in SMEs — Mixed-Methods Research Study

**Viktória Kubišová · Breda University of Applied Sciences · 2024**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Research](https://img.shields.io/badge/Method-Mixed--Methods-blueviolet?style=flat)
![Statistics](https://img.shields.io/badge/Statistics-ANOVA%20%7C%20t--test%20%7C%20Correlation-blue?style=flat)
![GDPR](https://img.shields.io/badge/Compliance-GDPR%20%7C%20Ethics%20Review-green?style=flat)
![LaTeX](https://img.shields.io/badge/Writing-LaTeX-lightgrey?style=flat)
![CRISP-DM](https://img.shields.io/badge/Framework-Analytics%20Translator-orange?style=flat)

---

## Overview

This project was completed in collaboration with Digiwerkplaats — an organisation supporting the digital transformation of SMEs in the Netherlands. Working in a group of four, we conducted a full mixed-methods research study into how AI integration affects SME employees, combining a quantitative survey (N = 154) distributed via Prolific with semi-structured qualitative interviews.

The block's focus was on the **analytics translator** role — bridging technical data science and business stakeholders — alongside developing formal research methodology skills: study design, ethics and GDPR compliance, statistical analysis, thematic analysis, academic writing in LaTeX, and stakeholder-oriented policy communication.

**My individual research question:** *"How does having the skills to use AI tools relate to SME employees' perceptions of job security?"*

**Group research question:** *"How does the integration of third-party AI tools influence employee job performance and satisfaction in SMEs?"*

**Teammates:** Celine Wu, Sally Ibrahim, Deuza Varela

**My role:** In addition to my individual research contributions, I served as project manager for the group — coordinating team meetings, managing the Scrum/Trello board, tracking deliverables, and ensuring compliance documentation was completed on time.

---

## Research Questions & Hypotheses

**Individual:**
- H0: AI skills do not significantly influence SME employees' job security perceptions
- H1: Employees with AI skills perceive significantly higher job security than those without

**Group sub-questions covered four subdomains:** operational efficiency, job satisfaction across company sizes, AI impact in creative fields, and AI skills and job security perceptions (my focus).

---

## Dataset

| Property | Detail |
|----------|--------|
| Quantitative | Online survey via Prolific, N = 154 SME employees |
| Qualitative | Semi-structured interviews, 4 participants |
| Demographics | Majority aged 18–44; 101 female / 46 male; primarily Bachelor's degree holders |
| Geography | South Africa (84), United Kingdom (12), and other countries |
| Key variables | AI knowledge level, job security perceptions, AI tool usage frequency, confidence, company-provided training, demographics |

---

## Approach

| Step | Description |
|------|-------------|
| 01 Literature review | AI adoption in SMEs, job security, AI skill development; APA-formatted sources |
| 02 Study design | Mixed-methods: Qualtrics survey + semi-structured interviews; individual and group research questions defined |
| 03 Ethics & GDPR | BUas ethics review application, informed consent, research information letter, data management plan (NWO template), FAIR and GDPR checklists |
| 04 Data collection | Survey distributed via Prolific; interviews conducted via Microsoft Teams |
| 05 Quantitative analysis *(Viktória)* | Descriptive statistics, data visualisation (grouped bar charts, heatmaps, violin plots, box plots, swarm plots), hypothesis testing (ANOVA, Welch's t-test, Tukey HSD, Pearson correlation), ordinal encoding of key variables |
| 06 Qualitative analysis *(Viktória)* | Thematic analysis of interview transcripts: familiarisation, coding, theme generation, refinement |
| 07 Stakeholder analysis | Power/interest grid mapping 8 stakeholder groups for Digiwerkplaats |
| 08 Policy paper *(group)* | Synthesised findings from all four sub-studies into policy recommendations for SMEs; LaTeX-formatted |
| 09 Individual research paper *(Viktória)* | ~4,000-word academic paper covering full research cycle; LaTeX-formatted, published under CC BY 4.0 |
| 10 Conference presentation | Academic research poster presented at BUas Digital Conference (Digiwerkplaats client event) |

**Pipeline:**

```
Literature review → Study design + ethics/GDPR compliance
→ Qualtrics survey (Prolific, N=154) + semi-structured interviews
→ Quantitative analysis (ANOVA, t-test, correlation) + thematic analysis
→ Individual research paper + group policy paper
→ Conference poster presentation (Digiwerkplaats)
```

---

## Key Results

**Quantitative findings (my individual paper):**

| Statistical test | Result |
|-----------------|--------|
| One-way ANOVA (Levels 2–4, excluding small groups) | F = 3.91, p = 0.0222 — significant difference in job security across AI knowledge levels |
| One-way ANOVA (revised grouping: Lower / Moderate / Higher) | F = 6.74, p = 0.0016 — both Moderate and Higher knowledge groups significantly more secure than Lower |
| Welch's t-test (Higher vs Lower AI Knowledge) | t = −2.75, p = 0.0133, Cohen's d = 0.78 — moderate to large effect size |

**H1 supported:** employees with higher AI proficiency perceive significantly greater job security. Moderate proficiency yielded the most meaningful jump; further skill gains beyond that level did not significantly increase perceptions.

**Key nuances:**
- Employees with **advanced** AI skills were also more aware of AI's potential for job displacement — higher competence brings both confidence and clearer risk perception
- Company-provided AI training enhanced job security perceptions more than access alone; **active participation** in training mattered more than mere availability
- Younger, tech-oriented employees in software/consulting sectors reported stronger job security; senior and non-technical employees expressed more uncertainty
- **Amusement class** — qualitative findings showed AI is viewed as a productivity complement, not a replacement, particularly in roles requiring creativity and strategic thinking

**Group policy paper findings:**
- AI primarily viewed as a productivity enhancer for repetitive and administrative tasks
- Ethical concerns (data privacy, bias, pressure to adopt) were prominent across all interviews
- Recommended targeted, accessible AI training over degree-based filtering; support for non-technical roles; transparent communication about AI's role

---

## Responsible Research

Full ethics and GDPR compliance framework implemented in accordance with the Netherlands Code of Conduct for Research Integrity (2018) and BUas Research Ethics Review guidelines:

- BUas Ethics Review Application submitted and reviewed by lecturing staff
- Informed consent obtained from all participants
- Data anonymised; participants assigned unique non-traceable identifiers
- FAIR and GDPR checklists completed
- Data Management Plan based on NWO template
- Data stored securely on encrypted BUas GitHub repository; retained for 10 years post-project

---

## Technology Stack

| Tool | Purpose |
|------|---------|
| Python (pandas, matplotlib, seaborn) | Quantitative data analysis and visualisation |
| Qualtrics / Prolific | Survey design and distribution |
| phik | Correlation matrix for mixed variable types |
| LaTeX (Overleaf) | Academic paper and policy paper formatting |
| Microsoft Teams | Interview conduct and recording |
| Thematic analysis | Qualitative interview analysis |
| Scrum / Trello | Agile project management |

---

## Assumptions & Limitations

- Sample recruited via Prolific is dominated by South African respondents (84/154), limiting generalisability across SME contexts
- Over two-thirds of participants were female, which may introduce gender-related bias in AI perceptions
- Qualitative data limited to 4 interviews from employees at a single SME (VEED), introducing potential organisational bias
- Statistical analysis faced challenges from uneven distribution across AI knowledge levels; normality assumption not fully met — results interpreted with appropriate caution
- Self-reported AI skill levels may not reflect actual proficiency

---

## Deliverables

- `EDA/Exploratory_Data_Analysis.ipynb` — Quantitative data analysis: descriptive statistics, hypothesis testing (ANOVA, t-test, correlation), visualisations
- `research_paper/Individual_Research_Paper.pdf` — Individual research paper (~4,000 words, LaTeX-formatted, CC BY 4.0)
- `research_paper/Group_Research_Proposal.pdf` — Group research proposal
- `policy_paper/AI_INTEGRATION_IN_SMEs__ENHANCING_USAGE_FREQUENCY_AND_EMPLOYEE_OUTCOMES.pdf` — Group policy paper with SME recommendations
- `policy_paper/Stakeholder_Analysis.pdf` — Power/interest stakeholder mapping for Digiwerkplaats
- `qualitative/` — Interview questions, thematic analysis spreadsheet, thematic analysis summary, qualitative study report
- `DMP/` — Full ethics and GDPR compliance package: BUas ethics review application, data management plan, data storage protocol, research information letter, signed informed consent, FAIR checklist, privacy checklist, codebook
- `poster/SME_AI_integration_Scientific_Poster.pdf` — Academic research poster presented at BUas Digital Conference

---

*Developed for Digiwerkplaats as part of the BSc Applied Data Science & AI programme at Breda University of Applied Sciences.*
