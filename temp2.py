import cv2

# List of codecs to test
codecs = {
    'MP4V': 'mp4v',  # MPEG-4
    'XVID': 'XVID',  # XviD codec
    'MJPG': 'MJPG',  # Motion JPEG
    'H264': 'avc1',  # H.264 codec
    'HEVC': 'hev1'   # H.265/HEVC codec (if supported)
}

def check_codec(codec_name, fourcc_code):
    try:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
        output = cv2.VideoWriter('src/sample_video.mp4', fourcc, 30.0, (640, 480))
        if output.isOpened():
            print(f"[✓] {codec_name} ({fourcc_code}) is supported.")
        else:
            print(f"[✗] {codec_name} ({fourcc_code}) is NOT supported.")
        output.release()
    except Exception as e:
        print(f"[!] Error checking {codec_name}: {e}")

if __name__ == "__main__":
    print("Checking OpenCV Codecs...\n")
    for name, code in codecs.items():
        check_codec(name, code)
