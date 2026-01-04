import base64
import logging
import os
import subprocess
import tempfile
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_settings import BaseSettings

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


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
    logger.debug(f"[extract_video_id] Input URL: {url}")

    if "youtu.be/" in url:
        video_id = url.split("youtu.be/")[1].split("?")[0]
    elif "/shorts/" in url:
        video_id = url.split("/shorts/")[1].split("?")[0]
    elif "watch?v=" in url:
        from urllib.parse import parse_qs, urlparse

        parsed = urlparse(url)
        video_id = parse_qs(parsed.query).get("v", [""])[0]
    else:
        video_id = ""

    logger.debug(f"[extract_video_id] Extracted video_id: {video_id}")
    return video_id


def download_with_ytdlp(url: str, cookies: str | None, temp_dir: Path) -> tuple[Path, Path]:
    """yt-dlp로 영상과 오디오 다운로드"""
    logger.info(f"[download_with_ytdlp] Starting download for URL: {url}")
    logger.debug(f"[download_with_ytdlp] temp_dir: {temp_dir}")
    logger.debug(f"[download_with_ytdlp] cookies provided: {bool(cookies)}")
    if cookies:
        logger.debug(f"[download_with_ytdlp] cookies length: {len(cookies)} chars")

    video_id = extract_video_id(url)
    if not video_id:
        logger.error("[download_with_ytdlp] Invalid YouTube URL - no video_id extracted")
        raise ValueError("Invalid YouTube URL")

    # 하이픈으로 시작하는 ID 처리
    safe_id = f"v{video_id}" if video_id.startswith("-") else video_id
    logger.debug(f"[download_with_ytdlp] video_id: {video_id}, safe_id: {safe_id}")

    video_path = temp_dir / f"{safe_id}.mp4"
    audio_path = temp_dir / f"{safe_id}.m4a"
    logger.debug(f"[download_with_ytdlp] Expected video_path: {video_path}")
    logger.debug(f"[download_with_ytdlp] Expected audio_path: {audio_path}")

    # 쿠키 파일 생성
    cookie_file = None
    if cookies:
        cookie_file = temp_dir / "cookies.txt"
        cookie_file.write_text(cookies)
        logger.debug(f"[download_with_ytdlp] Cookie file created: {cookie_file}")
        logger.debug(f"[download_with_ytdlp] Cookie file exists: {cookie_file.exists()}")

    base_cmd = ["yt-dlp", "-v"]  # -v for verbose
    if cookie_file:
        base_cmd.extend(["--cookies", str(cookie_file)])

    # yt-dlp 버전 확인
    version_result = subprocess.run(["yt-dlp", "--version"], capture_output=True, text=True)
    logger.info(f"[download_with_ytdlp] yt-dlp version: {version_result.stdout.strip()}")

    # 사용 가능한 포맷 확인
    logger.info("[download_with_ytdlp] Checking available formats...")
    format_cmd = base_cmd + ["-F", url]
    logger.debug(f"[download_with_ytdlp] Format command: {' '.join(format_cmd)}")
    format_result = subprocess.run(format_cmd, capture_output=True, text=True)
    logger.debug(f"[download_with_ytdlp] Format stdout:\n{format_result.stdout}")
    if format_result.stderr:
        logger.debug(f"[download_with_ytdlp] Format stderr:\n{format_result.stderr}")

    # 비디오 다운로드 (최고 품질 비디오+오디오 병합)
    logger.info("[download_with_ytdlp] Starting video download...")
    video_cmd = base_cmd + [
        "-f", "bestvideo[height<=720]+bestaudio/best[height<=720]",
        "--merge-output-format", "mp4",
        "-o", str(video_path),
        url,
    ]
    logger.debug(f"[download_with_ytdlp] Video command: {' '.join(video_cmd)}")

    result = subprocess.run(video_cmd, capture_output=True, text=True)
    logger.debug(f"[download_with_ytdlp] Video download returncode: {result.returncode}")
    logger.debug(f"[download_with_ytdlp] Video download stdout:\n{result.stdout}")
    if result.stderr:
        logger.debug(f"[download_with_ytdlp] Video download stderr:\n{result.stderr}")

    if result.returncode != 0:
        logger.error(f"[download_with_ytdlp] Video download failed with code {result.returncode}")
        raise RuntimeError(f"Video download failed: {result.stderr}")

    # temp_dir 내용 확인
    logger.info("[download_with_ytdlp] Checking temp_dir contents after video download...")
    all_files = list(temp_dir.iterdir())
    logger.debug(f"[download_with_ytdlp] Files in temp_dir: {[str(f) for f in all_files]}")
    for f in all_files:
        logger.debug(f"[download_with_ytdlp]   - {f.name}: {f.stat().st_size} bytes")

    # 파일 존재 확인 (확장자가 다를 수 있음)
    logger.debug(f"[download_with_ytdlp] Checking video_path exists: {video_path.exists()}")
    if not video_path.exists():
        logger.warning(f"[download_with_ytdlp] Expected video_path not found: {video_path}")
        # glob으로 찾기
        matches = list(temp_dir.glob(f"{safe_id}.*"))
        logger.debug(f"[download_with_ytdlp] Glob matches for {safe_id}.*: {[str(m) for m in matches]}")
        video_matches = [m for m in matches if m.suffix in (".mp4", ".webm", ".mkv")]
        logger.debug(f"[download_with_ytdlp] Video matches: {[str(m) for m in video_matches]}")
        if video_matches:
            video_path = video_matches[0]
            logger.info(f"[download_with_ytdlp] Using found video file: {video_path}")
        else:
            logger.error(f"[download_with_ytdlp] No video file found!")
            raise FileNotFoundError(f"Video file not found: {video_path}")

    logger.info(f"[download_with_ytdlp] Video file confirmed: {video_path} ({video_path.stat().st_size} bytes)")

    # ffmpeg로 비디오에서 오디오 추출 (비디오 파일 유지)
    logger.info("[download_with_ytdlp] Extracting audio from video with ffmpeg...")
    audio_cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "aac", "-b:a", "128k",
        str(audio_path),
    ]
    logger.debug(f"[download_with_ytdlp] Audio command: {' '.join(audio_cmd)}")

    result = subprocess.run(audio_cmd, capture_output=True, text=True)
    logger.debug(f"[download_with_ytdlp] Audio extraction returncode: {result.returncode}")
    logger.debug(f"[download_with_ytdlp] Audio extraction stdout:\n{result.stdout}")
    if result.stderr:
        logger.debug(f"[download_with_ytdlp] Audio extraction stderr:\n{result.stderr}")

    if result.returncode != 0:
        logger.error(f"[download_with_ytdlp] Audio extraction failed with code {result.returncode}")
        raise RuntimeError(f"Audio extraction failed: {result.stderr}")

    # temp_dir 내용 다시 확인
    logger.info("[download_with_ytdlp] Checking temp_dir contents after audio download...")
    all_files = list(temp_dir.iterdir())
    logger.debug(f"[download_with_ytdlp] Files in temp_dir: {[str(f) for f in all_files]}")
    for f in all_files:
        logger.debug(f"[download_with_ytdlp]   - {f.name}: {f.stat().st_size} bytes")

    # 파일 존재 확인
    logger.debug(f"[download_with_ytdlp] Checking audio_path exists: {audio_path.exists()}")
    if not audio_path.exists():
        logger.warning(f"[download_with_ytdlp] Expected audio_path not found: {audio_path}")
        matches = list(temp_dir.glob(f"{safe_id}*"))
        logger.debug(f"[download_with_ytdlp] Glob matches for {safe_id}*: {[str(m) for m in matches]}")
        audio_matches = [m for m in matches if m.suffix in (".m4a", ".mp3", ".wav", ".opus")]
        logger.debug(f"[download_with_ytdlp] Audio matches: {[str(m) for m in audio_matches]}")
        if audio_matches:
            audio_path = audio_matches[0]
            logger.info(f"[download_with_ytdlp] Using found audio file: {audio_path}")
        else:
            logger.error(f"[download_with_ytdlp] No audio file found!")
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

    logger.info(f"[download_with_ytdlp] Audio file confirmed: {audio_path} ({audio_path.stat().st_size} bytes)")

    # 쿠키 파일 삭제
    if cookie_file and cookie_file.exists():
        cookie_file.unlink()
        logger.debug("[download_with_ytdlp] Cookie file deleted")

    logger.info(f"[download_with_ytdlp] Download complete! video={video_path}, audio={audio_path}")
    return video_path, audio_path


async def transcribe_with_groq(audio_path: Path) -> dict:
    """Groq Whisper API로 자막 추출"""
    logger.info(f"[transcribe_with_groq] Starting transcription for: {audio_path}")
    logger.debug(f"[transcribe_with_groq] Audio file size: {audio_path.stat().st_size} bytes")
    logger.debug(f"[transcribe_with_groq] GROQ_API_KEY configured: {bool(settings.groq_api_key)}")

    async with httpx.AsyncClient(timeout=120.0) as client:
        with open(audio_path, "rb") as f:
            files = {"file": (audio_path.name, f, "audio/m4a")}
            data = {
                "model": "whisper-large-v3",
                "response_format": "verbose_json",
                "temperature": "0",
                "language": "ko",
            }
            logger.debug(f"[transcribe_with_groq] Request data: {data}")

            response = await client.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {settings.groq_api_key}"},
                files=files,
                data=data,
            )

        logger.debug(f"[transcribe_with_groq] Response status: {response.status_code}")
        if response.status_code != 200:
            logger.error(f"[transcribe_with_groq] Groq API error: {response.text}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Groq API error: {response.text}",
            )

        result = response.json()
        logger.info(f"[transcribe_with_groq] Transcription complete! Duration: {result.get('duration', 0)}s, Segments: {len(result.get('segments', []))}")
        logger.debug(f"[transcribe_with_groq] Text preview: {result.get('text', '')[:100]}...")
        return result


@app.get("/health")
async def health():
    logger.debug("[health] Health check called")
    return {"status": "ok"}


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(request: TranscribeRequest):
    """영상 다운로드 + 자막 추출"""
    logger.info("=" * 60)
    logger.info("[transcribe] New transcribe request received")
    logger.info(f"[transcribe] URL: {request.url}")
    logger.info(f"[transcribe] Cookies provided: {bool(request.cookies)}")
    logger.info("=" * 60)

    if not settings.groq_api_key:
        logger.error("[transcribe] GROQ_API_KEY not configured!")
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        logger.info(f"[transcribe] Created temp directory: {temp_path}")

        try:
            # 1. yt-dlp로 다운로드
            logger.info("[transcribe] Step 1: Downloading with yt-dlp...")
            video_path, audio_path = download_with_ytdlp(
                request.url, request.cookies, temp_path
            )
            logger.info(f"[transcribe] Step 1 complete: video={video_path}, audio={audio_path}")

            # 2. Groq Whisper로 자막 추출
            logger.info("[transcribe] Step 2: Transcribing with Groq Whisper...")
            whisper_result = await transcribe_with_groq(audio_path)
            logger.info("[transcribe] Step 2 complete")

            # 3. 영상 Base64 인코딩
            logger.info("[transcribe] Step 3: Encoding video to Base64...")
            with open(video_path, "rb") as f:
                video_bytes = f.read()
                video_base64 = base64.b64encode(video_bytes).decode("utf-8")
            logger.info(f"[transcribe] Step 3 complete: {len(video_bytes)} bytes -> {len(video_base64)} base64 chars")

            # 4. 응답 생성
            logger.info("[transcribe] Step 4: Building response...")
            response = TranscribeResponse(
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
            logger.info("[transcribe] Step 4 complete - returning response")
            logger.info("=" * 60)
            return response

        except subprocess.CalledProcessError as e:
            logger.exception("[transcribe] CalledProcessError occurred")
            raise HTTPException(
                status_code=500,
                detail=f"yt-dlp error: {e.stderr.decode() if e.stderr else str(e)}",
            )
        except Exception as e:
            logger.exception(f"[transcribe] Exception occurred: {type(e).__name__}: {e}")
            raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting yt-dlp Worker server...")
    logger.info(f"GROQ_API_KEY configured: {bool(settings.groq_api_key)}")
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
