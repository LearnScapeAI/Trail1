def get_youtube_video_for_subtopic(subtopic_query, api_key, max_results=5):
    import requests, isodate
    search_url = "https://www.googleapis.com/youtube/v3/search"
    video_url = "https://www.googleapis.com/youtube/v3/videos"

    try:
        response = requests.get(search_url, params={
            "part": "snippet", "q": subtopic_query, "key": api_key,
            "maxResults": max_results, "type": "video"
        })
        response.raise_for_status()
        results = response.json().get("items", [])
    except Exception as e:
        print(f"YouTube search error: {e}")
        return None

    for item in results:
        video_id = item["id"]["videoId"]
        try:
            details = requests.get(video_url, params={
                "part": "contentDetails,snippet", "id": video_id, "key": api_key
            }).json()
            duration = isodate.parse_duration(
                details["items"][0]["contentDetails"]["duration"]
            ).total_seconds()
            if duration < 600:
                snippet = details["items"][0]["snippet"]
                return {
                    "video_id": video_id,
                    "title": snippet.get("title"),
                    "description": snippet.get("description"),
                    "duration": duration,
                    "url": f"https://www.youtube.com/watch?v={video_id}"
                }
        except Exception as e:
            print(f"Video details error: {e}")
            continue
    return None