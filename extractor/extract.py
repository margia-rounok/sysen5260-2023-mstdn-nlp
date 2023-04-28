import requests
import json
import time

while True:
    #Fetch the public timeline from Mastodon
    response = requests.get("https://mastodon.social/api/v1/timelines/public")
    if response.status_code == 200:
        # Parse the response as JSON
        timeline = json.loads(response.text)

        # Write the timeline to a new file in the data directory with a timestamp in the name
        timestamp = str(int(time.time()))
        with open(f"/opt/app/data/{timestamp}.json", "w") as f:
            json.dump(timeline, f)

    # Wait for 30 seconds before fetching the timeline again
    time.sleep(300000)