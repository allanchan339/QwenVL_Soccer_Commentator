from openai import OpenAI
from dotenv import load_dotenv
import os
import cv2

load_dotenv()

client = OpenAI(
    api_key=os.getenv("MODELSCOPE_SDK_TOKEN"),
    base_url="https://api-inference.modelscope.cn/v1"
)

video_path = "cliped_video_final/Sun/raw/SunBall_Goal.mp4"


def get_video_length(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    cap.release()
    if video_fps == 0:
        return 0
    else:
        return total_frames / video_fps
    

video_messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
        {"type": "text", "text": "请用表格总结一下视频中的商品特点"},
        {
            "type": "video",
            "video": "https://duguang-labelling.oss-cn-shanghai.aliyuncs.com/qiansun/video_ocr/videos/50221078283.mp4",
            "total_pixels": 20480 * 28 * 28, "min_pixels": 16 * 28 * 2,
            # The default value is 2.0, but for demonstration purposes, we set it to 3.0.
            'fps': 3.0
        }]
     },
]


video_messages_local = [
    {"role": "system", "content": "You are a professional commentator for soccer. You are responsible for providing real-time commentary on the game."},
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": f"file://{video_path}",
                "max_pixels": 720 * 1280,
                "min_pixels": 256 * 256
            },
            {"type": "text",
                "text": f"Please descibe this game, FOCUS ON the action of players and THE BALL, explicitly for goals, assists, fouls, offsides, yellow/red cards, substitutions, and corner kicks. The video is {round(get_video_length(video_path),0)} seconds long and therefore the commentary should be around {round(get_video_length(video_path))*4 - 1} words long. You should also have a engaging tone. SKIP all non commentary content"},
            {"type": "text", "text": "用中文回答"},
        ]
    },
]

response = client.chat.completions.create(
    # model="Qwen/QVQ-72B-Preview",  # ModleScope Model-Id
    # model="Qwen/Qwen2.5-VL-32B-Instruct",
    model="Qwen/Qwen2.5-VL-72B-Instruct",
    # model = "Qwen/Qwen2.5-Omni-7B",
    messages=video_messages_local,
    stream=True
)


for chunk in response:
    print(chunk.choices[0].delta.content, end='', flush=True)
