
from youtube_transcript_api import YouTubeTranscriptApi


def yttranscript(info: list[dict]):
    """
    Searches and assembles transcripts of YouTube videos.

    Parameters:
    -----------
        info : list[dict]
        List of dictionaries that contain title, channel, ID of youtube videos.

    Returns:
    --------
        all_transcripts : list[str]
        List of transcripts.
    """
    ytt_api = YouTubeTranscriptApi()
    all_transcripts = []

    for item in info:
        try:
            fetched_transcript = ytt_api.fetch(item['id'], languages=['en'])
            transcript_snippets = [
                snippet.text for snipper in fetched_transcript]
            transcript = 'TRANSCRIPT: ' + ' '.join(transcript_snippets)

            all_transcripts.append(transcript)

        except Exception as e:
            print(f"Error fetching transcript for video ID {item['id']}: {e}")
    return all_transcripts
