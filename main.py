import base64
import os
import subprocess
import tempfile
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    groq_api_key: str = ""
    api_secret: str = ""  # 간단한 인증용

    class Config:
        env_file = ".env"


settings = Settings()

app = FastAPI(title="yt-dlp Worker", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TranscribeRequest(BaseModel):
    url: str
    cookies: str | None = None


class TranscriptSegment(BaseModel):
    id: int
    start: float
    end: float
    text: str


class TranscriptData(BaseModel):
    task: str
    language: str
    duration: float
    text: str
    segments: list[TranscriptSegment]


class VideoData(BaseModel):
    data: str  # Base64
    mimeType: str
    fileName: str


class TranscribeResponse(BaseModel):
    transcript: TranscriptData
    video: VideoData


def extract_video_id(url: str) -> str:
    """YouTube URL에서 video ID 추출"""
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    elif "/shorts/" in url:
        return url.split("/shorts/")[1].split("?")[0]
    elif "watch?v=" in url:
        from urllib.parse import parse_qs, urlparse

        parsed = urlparse(url)
        return parse_qs(parsed.query).get("v", [""])[0]
    return ""


def download_with_ytdlp(url: str, cookies: str | None, temp_dir: Path) -> tuple[Path, Path]:
    """yt-dlp로 영상과 오디오 다운로드"""
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError("Invalid YouTube URL")

    # 하이픈으로 시작하는 ID 처리
    safe_id = f"v{video_id}" if video_id.startswith("-") else video_id

    video_path = temp_dir / f"{safe_id}.mp4"
    audio_path = temp_dir / f"{safe_id}.m4a"

    # 쿠키 파일 생성
    cookie_file = None
    if cookies:
        cookie_file = temp_dir / "cookies.txt"
        cookie_file.write_text(cookies)

    base_cmd = ["yt-dlp"]
    if cookie_file:
        base_cmd.extend(["--cookies", str(cookie_file)])

    # 비디오 다운로드
    video_cmd = base_cmd + [
        "-f", "best[height<=720][ext=mp4]/best[height<=720]",
        "--merge-output-format", "mp4",
        "-o", str(video_path),
        url,
    ]
    subprocess.run(video_cmd, check=True, capture_output=True)

    # 오디오 다운로드
    audio_cmd = base_cmd + [
        "-x", "--audio-format", "m4a",
        "-o", str(audio_path),
        url,
    ]
    subprocess.run(audio_cmd, check=True, capture_output=True)

    # 쿠키 파일 삭제
    if cookie_file and cookie_file.exists():
        cookie_file.unlink()

    return video_path, audio_path


async def transcribe_with_groq(audio_path: Path) -> dict:
    """Groq Whisper API로 자막 추출"""
    async with httpx.AsyncClient(timeout=120.0) as client:
        with open(audio_path, "rb") as f:
            files = {"file": (audio_path.name, f, "audio/m4a")}
            data = {
                "model": "whisper-large-v3",
                "response_format": "verbose_json",
                "temperature": "0",
                "language": "ko",
            }
            response = await client.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {settings.groq_api_key}"},
                files=files,
                data=data,
            )

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Groq API error: {response.text}",
            )

        return response.json()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(request: TranscribeRequest):
    """영상 다운로드 + 자막 추출"""
    if not settings.groq_api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        try:
            # 1. yt-dlp로 다운로드
            video_path, audio_path = download_with_ytdlp(
                request.url, request.cookies, temp_path
            )

            # 2. Groq Whisper로 자막 추출
            whisper_result = await transcribe_with_groq(audio_path)

            # 3. 영상 Base64 인코딩
            with open(video_path, "rb") as f:
                video_base64 = base64.b64encode(f.read()).decode("utf-8")

            # 4. 응답 생성
            return TranscribeResponse(
                transcript=TranscriptData(
                    task=whisper_result.get("task", "transcribe"),
                    language=whisper_result.get("language", "ko"),
                    duration=whisper_result.get("duration", 0),
                    text=whisper_result.get("text", ""),
                    segments=[
                        TranscriptSegment(
                            id=seg.get("id", i),
                            start=seg.get("start", 0),
                            end=seg.get("end", 0),
                            text=seg.get("text", ""),
                        )
                        for i, seg in enumerate(whisper_result.get("segments", []))
                    ],
                ),
                video=VideoData(
                    data=video_base64,
                    mimeType="video/mp4",
                    fileName=video_path.name,
                ),
            )

        except subprocess.CalledProcessError as e:
            raise HTTPException(
                status_code=500,
                detail=f"yt-dlp error: {e.stderr.decode() if e.stderr else str(e)}",
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
