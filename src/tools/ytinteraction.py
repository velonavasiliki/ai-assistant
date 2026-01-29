from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta
import html
import os
from config import Config

# Load mock data to test without API calls
if os.getenv("MOCK_MODE", "0") == "1":
    from tools.mock_data import MOCK_VIDEOS, MOCK_TRANSCRIPTS, MOCK_TRANSCRIPT_DEFAULT


class ytinteraction:
    """
    A class for interacting with the YouTube Data API to retrieve video information.
    Set MOCK_MODE=1 environment variable to use fake data for testing.
    """

    def __init__(self) -> None:
        self.info = {}
        self.mock_mode = os.getenv("MOCK_MODE", "0") == "1"
        if self.mock_mode:
            print("[MOCK MODE] YouTube API calls will return fake data")

    def ytretriever(self, query: str, order: str = 'viewCount', duration='medium', num_results: int = 1, before: str = None, after: str = None):
        """
        Searches YouTube for videos related to query, inside a timespan.
        Returns a dictionary containing the videos' ids as keys and title, channel, and date, as values.

        Parameters:
        -----------
            query : str
                Search term.
            order : str
                Search results sorted by one of the following: 'date', 'rating', 'relevance', 'viewCount'.
            duration : str
                Duration of search results: 'any', 'long' (minutes 20+), 'medium' (4-20), 'short' (<4).
            num_results : int
                Maximum number of videos to retrieve.
            before : str
                Date upper limit in form %m/%d/%Y.
            after : str
                Date lower limit in form %m/%d/%Y.

        Returns:
        --------
            dict[str, dict] :
                Dictionary self.info populated with ID keys and dictionary values. For a video ID key the
                associated value is of the form {title: , channel: , date:, transcript: , transcript_summary: }.
                The keys 'transcript', 'transcript_summary' are set to None.
        """
        # Mock mode - return fake data
        if self.mock_mode:
            print(f"[MOCK] Searching YouTube for: '{query}' (returning {num_results} results)")
            self.info = {}  # Clear previous results
            count = 0
            for vid, info in MOCK_VIDEOS.items():
                if count >= num_results:
                    break
                self.info[vid] = info.copy()
                count += 1
            return self.info

        from googleapiclient.discovery import build
        from googleapiclient.errors import HttpError

        self.info = {}  # Clear previous results
        youtube = build('youtube', 'v3', developerKey=Config.YOUTUBE_API_KEY)
        now_time = datetime.now(timezone.utc)

        try:
            if before:
                before_time = datetime.strptime(before, "%m/%d/%Y").replace(
                    hour=23, minute=59, second=59, microsecond=999999, tzinfo=timezone.utc
                )
            else:
                before_time = datetime.now(timezone.utc)

            if after:
                after_time = datetime.strptime(after, "%m/%d/%Y").replace(
                    hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc
                )
            else:
                after_time = (now_time - relativedelta(years=Config.YT_DEFAULT_YEARS_BACK)).replace(
                    hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc
                )

        except ValueError as e:
            print(f"Error parsing date strings: {e}")
            return {}

        try:
            search_response = youtube.search().list(
                part='id,snippet',
                q=query,
                type='video',
                order=order,
                relevanceLanguage='en',  # prefer videos relevant to english language
                safeSearch='strict',
                videoDuration=duration,
                videoCaption='closedCaption',
                maxResults=num_results,
                publishedAfter=after_time.isoformat(),
                publishedBefore=before_time.isoformat()
            ).execute()

            for item in search_response['items']:
                # eliminate videos set to premier later
                if item['snippet']['liveBroadcastContent'] == 'none' and item['id']['kind'] == 'youtube#video':
                    self.info[item['id']['videoId']] = {
                        'title': html.unescape(item['snippet']['title']),
                        'channel': item['snippet']['channelTitle'],
                        'date': item['snippet']['publishedAt'],
                        'id': item['id']['videoId'],
                        'transcript': None,
                        'transcript_summary': None
                    }

            return self.info

        except HttpError as e:
            print(f"A YouTube API error occurred: {e}")
            return {}
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return {}

    def yttranscript(self, ids: list[str]):
        """
        Retrieves transcripts of YouTube videos using youtube-transcript-api.
        Updates self.info with transcript data.

        Tries English variants first, then falls back to translating any available transcript.

        Parameters:
        -----------
            ids : list[str]
                List of video IDs to get transcript.

        Returns:
        --------
            dict[dict] :
                Dictionary self.info populated with ID keys and dictionary values. For a video ID key the
                associated value is {title, channel, date, transcript, transcript_summary}.
                This method updates the 'transcript' value.
        """
        # Mock mode - return fake transcripts
        if self.mock_mode:
            print(f"[MOCK] Fetching transcripts for: {ids}")
            for video_id in ids:
                # Use video-specific transcript or default
                transcript_text = MOCK_TRANSCRIPTS.get(video_id, MOCK_TRANSCRIPT_DEFAULT)
                transcript = 'TRANSCRIPT: ' + transcript_text
                if video_id in self.info:
                    self.info[video_id]['transcript'] = transcript
                else:
                    self.info[video_id] = {
                        'title': f'Mock Video {video_id}',
                        'channel': 'MockChannel',
                        'date': '2026-01-01T00:00:00Z',
                        'id': video_id,
                        'transcript': transcript,
                        'transcript_summary': None
                    }
                print(f"[MOCK] Got transcript for video {video_id}")
            return self.info

        import time
        import random
        from youtube_transcript_api import YouTubeTranscriptApi
        ytt_api = YouTubeTranscriptApi()
        english_variants = ['en', 'en-US', 'en-GB', 'en-AU', 'en-CA']

        for i, video_id in enumerate(ids):
            # Add delay between requests to avoid rate limiting (skip first request)
            if i > 0:
                delay = 1.0 + random.uniform(0.2, 0.8)
                time.sleep(delay)
            fetched_transcript = None

            # First, try English variants directly
            try:
                fetched_transcript = ytt_api.fetch(video_id, languages=english_variants)
                print(f"Found English transcript for video {video_id}")
            except Exception:
                # English not directly available, try to find a translatable transcript
                try:
                    transcript_list = ytt_api.list(video_id)

                    # Try to find any transcript that can be translated to English
                    for transcript in transcript_list:
                        if transcript.is_translatable:
                            print(f"Translating transcript from '{transcript.language}' to English for video {video_id}")
                            translated = transcript.translate('en')
                            fetched_transcript = translated.fetch()
                            break

                    if fetched_transcript is None:
                        print(f"No translatable transcripts found for video {video_id}")
                        continue

                except Exception as e:
                    print(f"Error retrieving transcript for video {video_id}: {e}")
                    continue

            # Combine all text segments
            transcript_text = ' '.join([entry.text for entry in fetched_transcript])
            transcript = 'TRANSCRIPT: ' + transcript_text

            # Update or create entry in self.info
            if video_id in self.info:
                self.info[video_id]['transcript'] = transcript
            else:
                self.info[video_id] = {
                    'title': None,
                    'channel': None,
                    'date': None,
                    'id': video_id,
                    'transcript': transcript,
                    'transcript_summary': None
                }
            print(f"Successfully retrieved transcript for video {video_id}")

        return self.info

