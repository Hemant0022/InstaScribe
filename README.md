# InfluenceIQ

InfluenceIQ is a Streamlit dashboard for influencer intelligence and lead scoring. It combines profile-level metrics, post analytics, lead prioritisation, and a searchable post inspector in a dark, card-based UI.

## Features

- Executive overview with KPI cards, smart insights, engagement trend, and lead-quality breakdowns
- Lead intelligence views for followers, engagement rate, authenticity, and ranked influencer profiles
- Post analytics with monthly engagement, likes vs comments charts, heatmaps, hashtag cloud, and post inspection
- AI lead scoring with pipeline funnel, score distribution, category comparison, and exportable outreach lists
- Sidebar filters for category, quality, engagement rate, followers, follower tier, date range, and year

## Data Files

The app reads CSV files from the `data/` folder:

- `influencer_master.csv`
- `post_metrics.csv`
- `category_dim.csv`
- `lead_scoring.csv` for reference or export, if needed

## Requirements

Install the Python dependencies listed in `requirements.txt`:

- streamlit
- pandas
- numpy
- matplotlib
- scikit-learn
- seaborn
- plotly
- wordcloud

## Run the App

```bash
streamlit run app1.py
```

If you are using the virtual environment in this project on Windows:

```powershell
& .\myenv\Scripts\Activate.ps1
streamlit run app1.py
```

## Project Structure

```text
Project/
	app1.py
	app.py
	file.py
	project.py
	requirements.txt
	data/
		category_dim.csv
		date_dim.csv
		influencer_master.csv
		lead_scoring.csv
		post_metrics.csv
```

## What the Dashboard Calculates

- `FF Ratio` = `Following_Count / Follower_Count`
- `Lead Score` is computed from engagement rate, follower count, sentiment, and SaaS relevance, then normalised to a 0-100 scale
- Lead quality tiers are assigned from the lead score: `low`, `medium`, and `high`

## Notes

- The app automatically looks for the CSV files in `data/` relative to the script location.
- If the dashboard does not start, make sure the required CSV files are present and the dependencies are installed.
