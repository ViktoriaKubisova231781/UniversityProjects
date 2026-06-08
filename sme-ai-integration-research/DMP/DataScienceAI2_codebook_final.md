<br><br> <!-- line breaks-->
<div style="text-align: left;">
    <em>Friday, 27th September 2024</em>
</div>
<br><br> <!-- line breaks-->

<div style="text-align: center;">
    <h1>AI Integration in SMEs: Effects on Employee Performance and Satisfaction</h1>
</div>


<p style="text-align: center;">Ibrahim Sally, Kubišová Viktória, Varela Deuza, Wu Celine</p>


<br><br> <!-- line breaks-->

## Project Description
The research is motivated by the increasing tension between the advancement of AI technology and human-centred work environments. As AI technologies continue to evolve and expand into new sectors, it is crucial to understand the practical implications for employees. This study aims to address this gap by examining how the implementation of AI through third-party apps affects the job performance and satisfaction of employees in small and medium-sized enterprises (SMEs).

The primary research question is: "How does the integration of third-party AI tools influence employee job performance and satisfaction in SMEs?" Subsequent sub-questions will approach the subject from different angles to provide a comprehensive understanding of the topic. To investigate this broad question, the research will address several subdomains that are central to understanding the effects of AI in the workplace. These subdomains include AI’s influence on job performance, employee satisfaction, ethical concerns, and the role of AI third-party applications in employees’ perception of their job security. These sub-questions will guide the research and contribute to a broader understanding of AI implementation in human-centred work environments and its implications for job performance and satisfaction.


After outlining the main focus of this Research Proposal, the section on research questions presents the key inquiries that will steer the study, specifically examining the influence of AI integration on employee job performance and satisfaction in SMEs.
The methodology section of the Research Proposal document delineates the approach employed to collect and analyze data. It explains how surveys, online and in-person interviews were utilized to investigate the impact of AI on employees in SME's


## Data Collection and Cleaning
The raw data for the project was collected using both quantitative and qualitative methods. Quantitative data was gathered through an online survey distributed to SME employees via Prolific.<br>
 Qualitative data was collected through interviews conducted either on Zoom or in person, depending on the participants' preferences. <br>
 The data collected is secured and anonymized in compliance with ethical standards outlined in the Netherlands Code of Conduct for Research Integrity. 

- Surveys were administered to capture statistical insights from SME employees on their experiences with AI tools.
- Interviews aimed to gain a deeper understanding of employee perspectives, focusing on job performance, satisfaction related and security to AI adoption.
- All collected data was anonymized, and participants were assigned unique identifiers to maintain confidentiality. 


# Data Processing Steps

## 1. Download the Data

### Survey Data:
- Access the survey platform (e.g., **Prolific**) where the survey was distributed.
- Export the survey responses in **CSV** or **Excel** format, ensuring all participant responses and relevant metadata are included.

### Interview Data:
- If interviews were conducted via **Zoom**, **Google Meet**, or similar platforms, download the **audio/video recordings** and **transcripts** (if available).
- If transcription tools weren't used, manually transcribe the interviews, including spoken words and important non-verbal cues.
- Save the transcripts in **text files**, **Word documents**, or **Excel sheets** where each response is linked to a unique participant ID, question number, and response text.

---

## 2. Organize Survey Data

### Step 1: Data Cleaning
- Remove incomplete or irrelevant responses from the dataset.
- Standardize the values in categorical fields (e.g., ensure consistent spellings or formats in industry names).
- Convert qualitative responses (e.g., text inputs) into categories for easier analysis.

### Step 2: Structuring Data
- Ensure each **row** represents a single respondent.
- Ensure each **column** represents a **survey question** or **variable** (e.g., `Industry`, `Role`, `AI Tools Used`).
- Use numeric codes for multiple-choice questions (e.g., `1 = Strongly Disagree`, `5 = Strongly Agree`).

### Step 3: Anonymize Data
- Replace any personal identifiers (e.g., names, emails) with unique participant IDs to ensure confidentiality.
- Remove or mask any other sensitive information that could identify respondents.

---

## 3. Organize Interview Transcripts

### Step 1: Transcription Formatting
- Review automatically generated transcripts for accuracy.
- Structure each response so it is clearly associated with the **participant’s unique ID** and **interview question**.
- Example in Excel:
  - **Column 1**: Participant ID (e.g., `P1`, `P2`)
  - **Column 2**: Question Number (e.g., `Q1`, `Q2`)
  - **Column 3**: Full Response Text

### Step 2: Data Cleaning
- Remove irrelevant conversation portions.
- Standardize open-ended responses by grouping similar answers together.

### Step 3: Anonymization
- Replace any personal identifiers (e.g., participant names, company names) with generic identifiers (e.g., `Participant 1`, `Company A`).

---

## 4. Create Two Tidy Data Sets for Each Data Type

### 4.1 Tidy Dataset for Survey Data
- **Row Structure**: Each row represents a **single respondent**.
- **Column Structure**: Each column represents a **variable** from the survey (e.g., `Industry`, `AI Tools Used`, `AI Skills Level`).
  - Example columns:
    - `Respondent ID`: Unique identifier for each participant.
    - `Industry`: The industry the participant works in (e.g., `Retail`, `Manufacturing`).
    - `Company Size`: Size of the participant's company (e.g., `Small`, `Medium`, `Large`).
    - `AI Tools Used`: List of AI tools used by the participant.
    - `AI Skills Level`: Self-assessed AI skill level (e.g., `Beginner`, `Advanced`).
- **File Format**: Save as **CSV** or **Excel** file, ensuring all data is clean (no missing values or redundancies).

### 4.2 Tidy Dataset for Interview Data
- **Row Structure**: Each row represents a **unique response** to an interview question.
- **Column Structure**:
  - `Participant ID`: Unique identifier for the interviewee.
  - `Question Number`: The number of the interview question (e.g., `Q1`, `Q2`).
  - `Response`: The participant’s full response to the question.
  - (Optional) `Theme/Code`: If coding was applied, a column for labeling the theme associated with the response (e.g., `Job Security`).
- **File Format**: Save as **CSV** or **Excel** file with responses linked to the appropriate participant and question. 

 For further details please refer to the README.md file

 ---

## Variables
Variables in the `tidy_data.txt` File
### General Description
- **Dimensions**: The dataset contains **X rows** (each representing a unique participant) and **Y columns** (variables collected through the survey).
- **Summary**: The data was collected to study the influence of third-party AI tools on employee job performance and satisfaction in SMEs. The survey covers several themes, including industry, company size, AI tools used, and the participants' perceptions of AI's impact on job performance, security, and satisfaction.
- **Variables**: The dataset includes demographic variables, questions about the impact of AI tools, and participant attitudes towards AI in their job roles.


---

### Variable 1: `Industry`
- **Name of the variable**: `Industry`
- **What the variable represents**: The industry the participant works in.
- **How the variable was measured**: Nominal (categories)
- **Unit of measurement**: Nominal (categories)
- **Class of the variable**: String
- **Unique values/levels of the variable**: 
  - `Art & Design`
  - `Marketing & Advertising`
  - `IT & Software`
  - `Manufacturing`
  - `Retail`
  - `Other`

  
---

### Variable 2: `Role`
- **Name of the variable**: `Role`
- **What the variable represents**: The participant's role within their company.
- **How the variable was measured**: Nominal (categories)
- **Unit of measurement**: Nominal (categories)
- **Class of the variable**: String
- **Unique values/levels of the variable**: 
  - `Creative`
  - `Administrative`
  - `Technical`
  - `Managerial`
  - `Other`


---

### Variable 3: `Company Size`
- **Name of the variable**: `Company Size`
- **What the variable represents**: The size of the company the participant works for.
- **How the variable was measured**: Ordinal (ranked categories)
- **Unit of measurement**: Ordinal (categories)
- **Class of the variable**: String
- **Unique values/levels of the variable**: 
  - `Small (0-50 employees)`
  - `Medium (50-250 employees)`
  - `Large (250+ employees)`


---

### Variable 4: `AI_Error_Reduction`
- **Name of the variable**: `AI_Error_Reduction`
- **What the variable represents**: The participant’s perception of how AI has reduced human errors in their work.
- **How the variable was measured**: Ordinal scale from 1 to 5
- **Unit of measurement**: Ordinal (scale)
- **Class of the variable**: Numeric
- **Unique values/levels of the variable**: 
  - `1 = No reduction`
  - `2 = Minor reduction`
  - `3 = Moderate reduction`
  - `4 = Significant reduction`
  - `5 = Drastic reduction`


---

### Variable 5: `Time_Saved`
- **Name of the variable**: `Time_Saved`
- **What the variable represents**: The amount of time saved per day due to AI tools.
- **How the variable was measured**: Ordinal (ranked categories)
- **Unit of measurement**: Ordinal (ranked categories)
- **Class of the variable**: String
- **Unique values/levels of the variable**: 
  - `No time saved`
  - `Less than 1 hour`
  - `1-3 hours`
  - `3-5 hours`
  - `More than 5 hours`


---

### Variable 6: `AI_Tools_Used`
- **Name of the variable**: `AI_Tools_Used`
- **What the variable represents**: A list of up to 3 AI tools participants use most frequently, rated for their operational efficiency.
- **How the variable was measured**: Nominal (categories) 
- **Unit of measurement**: Nominal
- **Class of the variable**: String
- **Unique values/levels of the variable**: 
  - Various AI tools listed by participants


---

### Variable 7: `AI_Technologies_Used`
- **Name of the variable**: `AI_Technologies_Used`
- **What the variable represents**: The types of AI technologies used by the participant's company.
- **How the variable was measured**: Nominal (multiple-choice categories)
- **Unit of measurement**: Nominal (categories)
- **Class of the variable**: String
- **Unique values/levels of the variable**: 
  - `Chatbots or Virtual Assistants`
  - `Data Analytics or Predictive Algorithms`
  - `Automation (e.g., robotic process automation)`
  - `Machine Learning models`
  - `Generative AI (e.g., design, creative writing)`
  - `Other (please specify)`
  - `None`


---
### Variable 8: `AI_Skills_Level`
- **Name of the variable**: `AI_Skills_Level`
- **What the variable represents**: The participant’s self-assessed skill level in using the AI tools they selected or named.
- **How the variable was measured**: Ordinal (scale from 1 to 5)
- **Unit of measurement**: Ordinal (scale)
- **Class of the variable**: Numeric
- **Unique values/levels of the variable**:
  - `1 = Beginners`
  - `2`
  - `3 = Average`
  - `4`
  - `5 = Advanced`


---

### Variable 9: `AI_Content_Support`
- **Name of the variable**: `AI_Content_Support`
- **What the variable represents**: The participant’s opinion on AI-generated content in their field (e.g., art, design, writing).
- **How the variable was measured**: Ordinal (ranked categories)
- **Unit of measurement**: Ordinal (ranked categories)
- **Class of the variable**: String
- **Unique values/levels of the variable**: 
  - `I fully support it`
  - `I somewhat support it`
  - `I am neutral`
  - `I am somewhat against it`
  - `I am fully against it`
  

---

### Variable 10: `AI_Skills_Job_Security`
- **Name of the variable**: `AI_Skills_Job_Security`
- **What the variable represents**: The extent to which participants believe their ability to use AI tools contributes to job security.
- **How the variable was measured**:  Ordinal (scale from 1 to 5)
- **Unit of measurement**: Ordinal (scale)
- **Class of the variable**: Numeric
- **Unique values/levels of the variable**:
  - `1 = Not at all`
  - `2 = Slightly`
  - `3 = Moderately`
  - `4 = Considerably`
  - `5 = Significantly`


---
### Variable 11: `AI_Job_Risk`
- **Name of the variable**: `AI_Job_Risk`
- **What the variable represents**: The participant’s belief about whether their company's use of AI tools puts their current job at risk.
- **How the variable was measured**:  Ordinal (scale from 1 to 5)
- **Unit of measurement**: Ordinal (scale)
- **Class of the variable**: Numeric
- **Unique values/levels of the variable**:
  - `1 = Definitely not`
  - `2 = Probably not`
  - `3 = Might or might not`
  - `4 = Probably yes`
  - `5 = Definitely yes`
  

---

### Variable 12: `AI_Skills_Importance` 
- **Name of the variable**: `AI_Skills_Importance`
- **What the variable represents**: The importance of AI skills for maintaining the participant's current job position.
- **How the variable was measured**: Ordinal (scale from 1 to 5)
- **Unit of measurement**: Ordinal (scale)
- **Class of the variable**: Numeric
- **Unique values/levels of the variable**: 
  - `1 = Not important at all`
  - `2 = Slightly important`
  - `3 = Moderately important`
  - `4 = Very important`
  - `5 = Extremely important`


---
### Variable 13: `Job_Satisfaction`
- **Name of the variable**: `Job_Satisfaction`
- **What the variable represents**: The participant’s current level of job satisfaction.
- **How the variable was measured**:  Ordinal (ranked categories)
- **Unit of measurement**: Ordinal (ranked categories)
- **Class of the variable**: String
- **Unique values/levels of the variable**: 
  - `Very unsatisfied`
  - `Unsatisfied`
  - `Neutral`
  - `Satisfied`
  - `Very satisfied`

---

### Variable 14: `AI_Impact_on_Job_Satisfaction`
- **Name of the variable**: `AI_Impact_on_Job_Satisfaction`
- **What the variable represents**: The impact of AI implementation on the participant’s job satisfaction.
- **How the variable was measured**: Ordinal (ranked categories)
- **Unit of measurement**: Ordinal (ranked categories)
- **Class of the variable**: String
- **Unique values/levels of the variable**: 
  - `Yes, AI has significantly improved my job satisfaction`
  - `Yes, AI has slightly improved my job satisfaction`
  - `AI has no impact on my job satisfaction`
  - `No, AI has slightly decreased my job satisfaction`
  - `No, AI has significantly decreased my job satisfaction`

### Notes on Variables
- **Additional variables** include data on AI skills, job security, satisfaction, and perceived importance of AI tools. These are measured either on ordinal scales (1 to 5) or nominal categories.

---

### Sources
No external sources were used for this dataset; the survey was developed based on the research questions outlined.

---

### Annex
- The raw data has been anonymized, and participants were given unique IDs to protect their privacy. The survey responses were gathered using a combination of Likert scales, multiple-choice questions, and open-ended responses for qualitative analysis.

