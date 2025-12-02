import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import logging
from pathlib import Path

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 설정 ---
BASE_MODEL_ID = "Qwen/Qwen3-8B"
ADAPTER_PATH = "./qwen3-8b-cs-interviewer-renew"
MERGED_MODEL_PATH = "./qwen3-8b-cs-interviewer-merged"

def merge_and_save():
    """
    LoRA 어댑터와 기본 모델을 병합하고, 로컬에 없을 경우 자동으로 다운로드합니다.
    """
    merged_model_path = Path(MERGED_MODEL_PATH)
    
    # --- 💡 [핵심 추가] 이미 병합된 모델이 있는지 확인 ---
    # config.json 파일의 존재 여부로 확인
    if (merged_model_path / "config.json").exists():
        logger.info(f"이미 병합된 모델이 '{MERGED_MODEL_PATH}' 경로에 존재합니다. 병합을 건너뜁니다.")
        return

    logger.info(f"'{BASE_MODEL_ID}' 기본 모델을 로드합니다 (float16)...")
    logger.info("로컬에 모델이 없으면 자동으로 다운로드합니다.")
    
    # 병합 시에는 양자화 없이 float16으로 로드해야 합니다.
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)

    logger.info(f"'{ADAPTER_PATH}'에서 LoRA 어댑터를 로드합니다...")
    model_to_merge = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    logger.info("모델과 어댑터를 병합합니다...")
    # merge_and_unload()는 병합 후 어댑터를 메모리에서 해제하여 VRAM을 절약합니다.
    merged_model = model_to_merge.merge_and_unload()
    logger.info("병합 완료.")

    logger.info(f"병합된 모델을 '{MERGED_MODEL_PATH}' 경로에 저장합니다...")
    merged_model_path.mkdir(parents=True, exist_ok=True) # 저장 경로 생성
    merged_model.save_pretrained(MERGED_MODEL_PATH)
    tokenizer.save_pretrained(MERGED_MODEL_PATH)
    logger.info("저장 완료.")

if __name__ == "__main__":
    try:
        merge_and_save()
    except Exception as e:
        logger.error(f"병합 과정에서 오류 발생: {e}", exc_info=True)
        logger.error("VRAM이 충분한지, 어댑터 경로가 올바른지 확인해주세요.")