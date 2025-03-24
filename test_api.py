import requests

# Define the API endpoint (update with your actual URL)
url = "http://127.0.0.1:5000/predict"

# Define sample job descriptions to test
test_jobs = [
    {"description": "Earn $5000 per week from home! No experience needed! Click the link to apply now!"},
    {"description": "Software Engineer needed. Must have 5 years of experience in Python and Machine Learning."},
    {"description": "Urgent hiring! Send us your bank details to receive your first salary immediately!"},
    {"description": "We are a reputed company looking for a front-end developer with experience in React and Tailwind CSS."}
]

# Send requests and print responses
for job in test_jobs:
    response = requests.post(url, json=job)
    print(f"Job Description: {job['description']}")
    print(f"Prediction: {response.json()}\n")
