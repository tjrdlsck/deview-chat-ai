import os
import logging
from pathlib import Path
from huggingface_hub import HfApi, login
from dotenv import load_dotenv

# 1. 로깅(Logging) 설정: 정보를 출력하여 사용자에게 현재 상황을 명확히 전달합니다.
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# .env 파일을 환경 변수로 로드합니다.
load_dotenv() 


# --- 설정 변수 ---
# GGUF 파일 경로
# 단일 파일을 업로드하므로 이 변수를 사용합니다.
LOCAL_MODEL_PATH = "/workspace/model-Q4_K_M.gguf" 

# Hugging Face 사용자 이름과 저장소 이름
HF_USERNAME = "radi04" 
REPO_NAME = "Qwen3-4B-Deview-Finetune-v3" 
REPO_ID = f"{HF_USERNAME}/{REPO_NAME}"


def upload_single_file_model():
    """
    단일 GGUF 파일을 Hugging Face Hub에 업로드하는 함수입니다.
    """
    # [인증] 환경 변수에서 토큰을 가져옵니다.
    hf_token = os.getenv("HUGGINGFACE_TOKEN")

    if not hf_token:
        logger.error("Hugging Face 토큰을 찾을 수 없습니다. 환경 변수 'HUGGINGFACE_TOKEN' 형식으로 설정했는지 확인해주세요.")
        return

    local_path = Path(LOCAL_MODEL_PATH)

    if not local_path.exists() or local_path.is_dir():
        # 파일이 존재하지 않거나, 폴더인 경우 오류를 출력합니다.
        logger.error(f"업로드할 단일 모델 파일 경로를 찾을 수 없거나(Not Found), 폴더입니다(Is a Directory): {LOCAL_MODEL_PATH}")
        return

    try:
        logger.info("Hugging Face Hub에 로그인을 시작합니다 (토큰 사용)...")
        login(token=hf_token, add_to_git_credential=False)
        api = HfApi()

        # 저장소가 존재하지 않으면 자동으로 생성합니다.
        api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
        
        logger.info(f"업로드를 시작합니다: {LOCAL_MODEL_PATH} -> {REPO_ID}")

        # ****** [핵심 수정] upload_folder 대신 단일 파일용 upload_file 사용 ******
        # path_or_fileobj: 로컬에 있는 파일 경로
        # path_in_repo: 허브 저장소 내에 저장될 이름 (로컬 파일명과 동일하게 설정)
        api.upload_file(
            path_or_fileobj=LOCAL_MODEL_PATH,
            path_in_repo=local_path.name, # "model-Q4_K_M.gguf"가 됩니다.
            repo_id=REPO_ID,
            repo_type="model",
            commit_message=f"Upload GGUF Quantized Model: {local_path.name}",
        )
        # ******************************************************************

        logger.info("🎉 모델 업로드가 성공적으로 완료되었습니다!")
        logger.info(f"확인 링크: https://huggingface.co/{REPO_ID}/blob/main/{local_path.name}")

    except Exception as e:
        logger.error(f"업로드 중 치명적인 오류 발생: {e.__class__.__name__}: {e}")
        logger.error("토큰이 유효한지, 'write' 권한이 있는지 다시 확인해주세요.")


if __name__ == "__main__":
    upload_single_file_model()