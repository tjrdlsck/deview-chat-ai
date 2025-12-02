import logging
import os
from pathlib import Path
from huggingface_hub import HfApi
from dotenv import load_dotenv

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 💡 [핵심] .env 파일에서 환경 변수 로드 ---
load_dotenv()

# --- 설정 ---
LOCAL_MODEL_PATH = "./qwen3-8b-cs-interviewer-merged-rtn"
HF_USERNAME = "radi04"  # <-- 여기에 본인 ID를 입력하세요.
REPO_NAME = "qwen3-8b-cs-interviewer-merge-v1-150-q4"

def upload_merged_model():
    """로컬에 병합된 모델을 .env 파일의 토큰을 사용하여 Hugging Face Hub에 업로드합니다."""

    # --- 💡 [핵심] .env 파일에서 토큰 읽어오기 ---
    hf_token = os.getenv("HUGGINGFACE_TOKEN")

    if not hf_token:
        logger.error("Hugging Face 토큰을 찾을 수 없습니다.")
        logger.error(".env 파일에 'HUGGINGFACE_TOKEN=your_token' 형식으로 토큰을 설정했는지 확인해주세요.")
        return

    local_path = Path(LOCAL_MODEL_PATH)
    if not local_path.exists():
        logger.error(f"업로드할 모델 폴더 '{LOCAL_MODEL_PATH}'를 찾을 수 없습니다.")
        return

    repo_id = f"{HF_USERNAME}/{REPO_NAME}"
    
    try:
        logger.info(f"Hugging Face Hub에 모델 업로드를 시작합니다 (토큰 사용)...")
        logger.info(f"업로드 위치: {repo_id}")

        # --- 💡 [핵심] HfApi 생성 시 토큰 전달 ---
        api = HfApi(token=hf_token)

        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

        api.upload_folder(
            folder_path=str(local_path),
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload merged Qwen3-8B model"
        )
        
        logger.info("="*50)
        logger.info("✅ 업로드가 성공적으로 완료되었습니다!")
        logger.info(f"모델 주소: https://huggingface.co/{repo_id}")
        logger.info("="*50)

    except Exception as e:
        logger.error(f"업로드 중 오류 발생: {e}", exc_info=True)
        logger.error("토큰이 유효한지, 'write' 권한이 있는지 다시 확인해주세요.")

if __name__ == "__main__":
    upload_merged_model()