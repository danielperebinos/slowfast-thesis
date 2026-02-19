import os
import subprocess
import pandas as pd

# Path and URL configuration
BASE_DIR = "data/ava"
VIDEO_DIR = os.path.join(BASE_DIR, "videos")


# For simplicity, we assume you have already downloaded 'ava_train_v2.2.csv' into BASE_DIR

def download_and_trim(video_id, output_path):
    """
    Downloads the YouTube clip and trims the 15-minute segment (900s-1800s).
    """
    url = f"https://www.youtube.com/watch?v={video_id}"

    # yt-dlp command to get the stream URL without downloading the whole file.
    # We use ffmpeg to trim directly from the stream for efficiency.
    cmd = [
        "yt-dlp",
        "-g", "-f", "bestvideo[ext=mp4]",
        url
    ]

    try:
        print(f"Getting link for: {video_id}...")
        stream_url = subprocess.check_output(cmd).decode("utf-8").strip()

        # Trim the 900-1800 second segment (AVA standard) at 30 FPS.
        print(f"Downloading and trimming fragment for: {video_id}...")
        trim_cmd = [
            "ffmpeg", "-ss", "900", "-t", "900",
            "-i", stream_url,
            "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
            "-r", "30",  # Force constant 30 FPS for SlowFast compatibility.
            "-an",  # Remove audio as per training specifications.
            "-y", output_path
        ]
        subprocess.run(trim_cmd, check=True)
    except Exception as e:
        print(f"Error processing {video_id}: {e}")


def main():
    # Ensure the video directory exists
    os.makedirs(VIDEO_DIR, exist_ok=True)

    csv_path = os.path.join(BASE_DIR, "ava_train_v2.2.csv")

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please download annotations from the official AVA site.")
        return

    # Load video IDs from the official CSV
    df = pd.read_csv(csv_path, header=None, names=["video_id", "ts", "x1", "y1", "x2", "y2", "action", "person_id"])
    unique_videos = df["video_id"].unique()

    print(f"Found {len(unique_videos)} unique videos in the CSV.")

    # Iterate through all unique video IDs
    for vid in unique_videos:
        out_file = os.path.join(VIDEO_DIR, f"{vid}.mp4")
        if not os.path.exists(out_file):
            download_and_trim(vid, out_file)
        else:
            print(f"Clip {vid} already exists. Skipping.")


if __name__ == "__main__":
    main()