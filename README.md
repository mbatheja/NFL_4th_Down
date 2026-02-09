# NFL Decision Optimization & Coaching Behavior Analysis
This project features an Inverse Reinforcement Learning (IRL) framework designed to decode the underlying reward functions optimized by NFL coaches during 4th-down situations. While the broader project provides analytical recommendations, the IRL component serves as the behavioral benchmark, revealing what coaches actually prioritize in real-world scenarios.
---

## Inverse Reinforcement Learning (IRL) & Behavioral Analysis
The IRL model enables a direct comparison between empirical coaching decisions and analytical optimization. By treating historical decisions as "expert" demonstrations, we decode the situational features that drive NFL decision-making.

<li> Behavioral Decoding: The model establishes a baseline behavior policy across three primary actions: Go for it, Field Goal, or Punt. </li>

<li> Feature Importance: Analysis reveals that coaches are most heavily influenced by field position and time remaining. Key decoded importance weights include: </li>

Yardline (100-0): 2.036

Seconds Remaining: 0.974

Yards to Go: 0.818

<li> Coaching "Desperation": The model evaluates how features like score_differential and def_time_on_field_cum quantify the game context that shifts a coach toward more aggressive "Go-for-it" behavior. </li>

## Model Evaluation
Specific evaluations were conducted to validate the IRL model's fidelity to actual NFL behavior:

<li> In-Sample Accuracy: The model achieved an 85.7% accuracy in predicting actual coach decisions based on the decoded reward function. </li>

<li> Action Probabilities: On average, the model correctly mirrors the league's conservative bias, with a baseline punt probability of 55.8% vs. a go probability of 17.8%. </li>

<li> Team-Specific Analysis: Aggressiveness is evaluated by comparing team "Go-for-it" rates against their "Desperation Coefficients," allowing for a nuanced look at which coaches deviate from the league-wide norm. </li>

## Integration with the 4th-Down Decision Calculator
The IRL analysis is a critical component of a larger 4th-Down Decision Calculator pipeline. This system allows users to compare the "Coach's Choice" (derived from IRL) against "Optimized Choices" (derived from MAB).

<li> Multi-Armed Bandit (MAB): Serves Greedy and LinUCB recommendations optimized for EPA (Expected Points Added) or WPA (Win Probability Added). </li>

<li> Data Pipeline: A reproducible workflow that pulls play-by-play data (2016–2024) and builds situational features like 4-week rolling EPA and FG accuracy. </li>

<li> Streamlit App: A real-time interface where users can input game states to see how an analytical model's recommendation compares to the decoded historical behavior of NFL coaches. </li>

## Repo Layout

```Plaintext
.
├── IRL.ipynb                  # Reward function model & coaching behavior analysis
├── app.py                     # Streamlit UI for the Decision Calculator
├── artifacts/                 # Trained models (Behavioral policy & MAB arm models)
├── data/                      # Cleaned PBP and decision CSVs (2016-2024)
├── behavior_*.ipynb           # MAB training and OPE evaluation notebooks
└── data_clean_2016_2024.ipynb # Feature engineering and data cleaning pipeline
```
## Installation
```Bash
# Recommended Python 3.9–3.12
pip install streamlit pandas numpy scikit-learn scipy joblib matplotlib nfl_data_py
Workflow
```
Data Build: Run data_clean_2016_2024.ipynb to generate the situational dataset.

MAB Training: Execute behavior_2016_2024_epa.ipynb to train the recommendation models.

IRL Analysis: Run IRL.ipynb to perform the coaching behavior decoding and evaluation.

Launch App: Use streamlit run app.py to compare analytical vs. empirical decisions.

## Contributors
Inverse Reinforcement Learning and related evaluation: Mahima Batheja

MAB model and data cleaning: Lucas Hyunh

MAB evaluation: Ardak Baizhaxynova

Documentation: Mburu Kagiri
