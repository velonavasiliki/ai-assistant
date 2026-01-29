"""
Mock data for testing the application without API calls.
Usage: Set MOCK_MODE=1 environment variable to enable.
"""

MOCK_VIDEOS = {
    "mock_vid_001": {
        "title": "Introduction to Python Programming",
        "channel": "CodeAcademy",
        "date": "2026-01-20T10:00:00Z",
        "id": "mock_vid_001",
        "transcript": None,
        "transcript_summary": None
    },
    "mock_vid_002": {
        "title": "Machine Learning Fundamentals",
        "channel": "AI School",
        "date": "2026-01-18T14:30:00Z",
        "id": "mock_vid_002",
        "transcript": None,
        "transcript_summary": None
    },
    "mock_vid_003": {
        "title": "Data Science with Pandas",
        "channel": "DataCamp",
        "date": "2026-01-15T09:00:00Z",
        "id": "mock_vid_003",
        "transcript": None,
        "transcript_summary": None
    }
}

MOCK_TRANSCRIPTS = {
    "mock_vid_001": """Welcome to Introduction to Python Programming.
Python is a versatile programming language used for web development, data science, and automation.
Today we'll cover variables, data types, and basic syntax.
Variables in Python are created when you assign a value using the equals sign.
Python supports integers, floats, strings, and boolean data types.
Functions are defined using the def keyword followed by the function name.
We recommend practicing with small projects to build your skills.
Thank you for learning Python with us today.""",

    "mock_vid_002": """Welcome to Machine Learning Fundamentals.
Machine learning is a subset of artificial intelligence that enables computers to learn from data.
There are three main types: supervised learning, unsupervised learning, and reinforcement learning.
Supervised learning uses labeled data to train models for prediction tasks.
Common algorithms include linear regression, decision trees, and neural networks.
Feature engineering is crucial for model performance.
Always split your data into training and testing sets to evaluate your model.
Thank you for joining this machine learning tutorial.""",

    "mock_vid_003": """Welcome to Data Science with Pandas.
Pandas is a powerful Python library for data manipulation and analysis.
DataFrames are the core data structure, similar to spreadsheets or SQL tables.
You can read CSV files using pd.read_csv and Excel files with pd.read_excel.
Data cleaning involves handling missing values, duplicates, and outliers.
Use groupby for aggregation and merge for combining datasets.
Visualization can be done directly with pandas or with matplotlib and seaborn.
Thank you for learning Pandas with us."""
}

MOCK_TRANSCRIPT_DEFAULT = """This is a mock transcript for testing purposes.
The video covers various interesting topics and concepts.
Thank you for watching this educational content."""
