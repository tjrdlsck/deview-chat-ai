import os
import json
import torch
import logging
from datetime import datetime

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer

from huggingface_hub import login, create_repo
from dotenv import load_dotenv
import wandb

# --- 로깅 설정 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path="config.json"):
    """설정 파일을 로드하고 반환합니다."""
    logger.info(f"'{config_path}'에서 설정 파일을 로드합니다...")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"오류: '{config_path}' 파일을 찾을 수 없습니다.")
        exit()
    except json.JSONDecodeError:
        logger.error(f"오류: '{config_path}' 파일의 형식이 올바르지 않습니다.")
        exit()

def login_services():
    """Hugging Face와 W&B에 로그인합니다."""
    load_dotenv()
    
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    wandb_api_key = os.getenv("WANDB_API_KEY")

    if not hf_token:
        logger.warning("HUGGINGFACE_TOKEN 환경 변수를 찾을 수 없습니다.")
    else:
        try:
            login(token=hf_token)
            logger.info("Hugging Face 로그인 성공.")
        except Exception as e:
            logger.error(f"Hugging Face 로그인 실패: {e}")

    if not wandb_api_key:
        logger.warning("WANDB_API_KEY 환경 변수를 찾을 수 없습니다. W&B 로깅이 비활성화될 수 있습니다.")
    else:
        try:
            wandb.login(key=wandb_api_key)
            logger.info("W&B 로그인 성공.")
        except Exception as e:
            logger.error(f"W&B 로그인 실패: {e}")


def main():
    config = load_config()

    # --- 설정값 변수화 ---
    project_cfg = config['project_settings']
    qlora_cfg = config['qlora_settings']
    training_cfg = config['training_settings']
    dataset_cfg = config['dataset_settings']

    # --- 1. 로그인 및 환경 설정 ---
    login_services()
    
    hub_model_id = f"{project_cfg['hf_username']}/{project_cfg['new_model_name']}"

    # [수정] TrainingArguments에 전달하기 전에 커스텀 인자를 미리 제거하고 사용합니다.
    output_dir_prefix = training_cfg.pop("output_dir_prefix", "./") # pop으로 값을 가져오고, 없으면 기본값 './' 사용
    output_dir = f"{output_dir_prefix}{project_cfg['new_model_name']}"

    os.environ["WANDB_PROJECT"] = project_cfg['project_name']
    logger.info(f"W&B 프로젝트가 '{project_cfg['project_name']}' (으)로 설정되었습니다.")

    try:
        create_repo(project_cfg['new_model_name'], private=True, exist_ok=True)
        logger.info(f"HF Hub 리포지토리 '{hub_model_id}' 준비 완료.")
    except Exception as e:
        logger.warning(f"리포지토리 생성/확인 중 오류(무시 가능): {e}")


    # --- 2. 데이터셋 로드 및 분할 ---
    logger.info(f"'{project_cfg['dataset_id']}'에서 데이터셋을 로드합니다...")
    try:
        full_dataset = load_dataset(project_cfg['dataset_id'], split="train")
        dataset_split = full_dataset.train_test_split(
            test_size=dataset_cfg['test_size'],
            shuffle=True,
            seed=dataset_cfg['random_seed']
        )
        train_dataset = dataset_split["train"]
        eval_dataset = dataset_split["test"]
        logger.info(f"데이터셋 분할 완료: 학습 {len(train_dataset)}, 검증 {len(eval_dataset)}")
    except Exception as e:
        logger.error(f"데이터셋 로드에 실패했습니다: {e}"); exit()


    # --- 3. 모델, 토크나이저, QLoRA 설정 ---
    logger.info(f"'{project_cfg['base_model_id']}' 모델과 토크나이저를 로드합니다...")
    
    # BitsAndBytes (QLoRA) 설정
    compute_dtype = getattr(torch, qlora_cfg['bnb_4bit_compute_dtype'])
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=qlora_cfg['load_in_4bit'],
        bnb_4bit_quant_type=qlora_cfg['bnb_4bit_quant_type'],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=qlora_cfg['bnb_4bit_use_double_quant'],
    )

    # 기본 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        project_cfg['base_model_id'],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(
        project_cfg['base_model_id'],
        trust_remote_code=True,
        padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        logger.info(f"pad_token이 없어 eos_token '{tokenizer.eos_token}'으로 설정합니다.")
    
    logger.info("모델과 토크나이저 로드 완료.")

    # LoRA 설정
    peft_config = LoraConfig(
        r=qlora_cfg['lora_r'],
        lora_alpha=qlora_cfg['lora_alpha'],
        lora_dropout=qlora_cfg['lora_dropout'],
        bias="none",
        task_type="CAUSAL_LM",
        # target_modules 자동 탐지 기능은 Trainer가 내부적으로 처리하므로 생략 가능
    )

    # --- 4. 학습 설정 (TrainingArguments) ---
    max_seq_length = training_cfg.pop("max_seq_length", 2048) # <<< 이 줄 추가
    packing = training_cfg.pop("packing", False)             # <<< 이 줄 추가

    # --- 4. 학습 설정 (TrainingArguments) ---
    logger.info("TrainingArguments를 설정합니다...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        hub_model_id=hub_model_id,
        **training_cfg # <<< max_seq_length가 여기에 포함된 채로 전달되도록 둡니다.
    )

    # --- 5. SFTTrainer 생성 및 학습 시작 ---
    logger.info("SFTTrainer를 생성합니다...")
    trainer = SFTTrainer(
        model=model,
        args=training_args, # <<< 모든 설정은 이 'args'를 통해 전달됩니다.
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config
    )

    logger.info("학습을 시작합니다...")
    try:
        trainer.train(resume_from_checkpoint=True)
    except (FileNotFoundError, ValueError): # 체크포인트가 없을 때 발생하는 오류들
        logger.warning("체크포인트를 찾을 수 없습니다. 처음부터 학습을 시작합니다.")
        trainer.train()
    
    logger.info("학습이 완료되었습니다.")

    # --- 6. 최종 모델 저장 및 푸시 ---
    final_adapter_dir = f"{output_dir}-final"
    trainer.save_model(final_adapter_dir)
    logger.info(f"최종 LoRA 어댑터가 로컬 '{final_adapter_dir}'에 저장되었습니다.")

    if training_cfg['push_to_hub']:
        logger.info(f"최종 모델을 Hub '{hub_model_id}'에 푸시합니다...")
        try:
            trainer.push_to_hub(commit_message="End of training")
            logger.info("Hub 푸시 완료.")
        except Exception as e:
            logger.error(f"Hub 푸시 중 오류 발생: {e}")

    if wandb.run:
        wandb.finish()
    
    logger.info("모든 파이프라인이 성공적으로 완료되었습니다.")

if __name__ == "__main__":
    main()