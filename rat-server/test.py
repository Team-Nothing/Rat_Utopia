import cv2


def get_video_info(video_path):
    try:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return {"error": "Cannot open video file"}

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cap.release()
        return {
            "frame_rate": fps,
            "total_frames": frame_count,
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    file_path = "data/records/20241121_220318.avi"
    info = get_video_info(file_path)
    print(info)
