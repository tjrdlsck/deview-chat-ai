### **AI 면접관 모델 파인튜닝 A-Z 가이드**

이 가이드는 `Qwen` 언어 모델을 기반으로 특정 면접 데이터셋을 학습시켜, 나만의 AI 기술 면접관을 만드는 전 과정을 안내합니다.

#### **Part 1: 사전 준비 (Hardware & Software Prerequisites)**

새로운 컴퓨터에서 가장 먼저 확인하고 준비해야 할 것들입니다.

**1. 하드웨어 요구사항**
*   **GPU**: **NVIDIA GPU가 필수적**입니다. (예: RTX 3090, RTX 4090, A100 등)
*   **VRAM (GPU 메모리)**:
    *   **최소 16GB 이상**: `Qwen3-4B`와 같은 작은 모델 학습에 권장됩니다.
    *   **권장 24GB 이상**: `Qwen3-8B` 모델을 안정적으로 학습시키기 위해 강력히 권장됩니다.

**2. 필수 소프트웨어 설치**
*   **Git**: 프로젝트 코드를 다운로드하기 위해 필요합니다. [Git 공식 홈페이지](https://git-scm.com/downloads)에서 설치하세요.
*   **Python**: **버전 3.9 이상**을 권장합니다. [Python 공식 홈페이지](https://www.python.org/downloads/)에서 설치하세요.
    *   ⚠️ 설치 시 **"Add Python to PATH"** 옵션을 반드시 체크해주세요.
*   **NVIDIA 드라이버**: GPU를 사용하기 위한 최신 드라이버를 설치하세요. [NVIDIA 드라이버 다운로드](https://www.nvidia.com/Download/index.aspx)
*   **CUDA Toolkit**: PyTorch가 GPU를 활용하기 위해 필요합니다.
    *   **가장 중요한 단계!** 먼저 PyTorch 홈페이지에서 **원하는 PyTorch 버전에 맞는 CUDA 버전을 확인**한 후, 해당 버전의 CUDA Toolkit을 [NVIDIA 개발자 사이트](https://developer.nvidia.com/cuda-toolkit-archive)에서 설치하세요. (예: PyTorch 2.1은 CUDA 11.8 또는 12.1을 지원)

---

#### **Part 2: 개발 환경 설정 (Environment Setup)**

이제 프로젝트를 실행할 격리되고 깨끗한 환경을 만듭니다.

**1. 프로젝트 코드 다운로드 (Clone)**
   *   터미널(Git Bash, PowerShell, cmd 등)을 열고 다음 명령어를 실행하세요.
     ```bash
     git clone https://github.com/your-repository-url.git  # <-- 이 부분은 실제 프로젝트의 Git 주소로 변경
     cd your-project-folder # 다운로드된 프로젝트 폴더로 이동
     ```

**2. 가상환경 생성 및 활성화**
   *   프로젝트 폴더 내에서 터미널에 다음 명령어를 입력하여 `venv`라는 가상환경을 만듭니다.
     ```bash
     python3.10 -m venv venv
     ```
   *   생성된 가상환경을 활성화합니다.
     *   **Windows:** `.\venv\Scripts\activate`
     *   **macOS/Linux:** `source venv/bin/activate`
   *   활성화되면 터미널 프롬프트 앞에 `(venv)`가 표시됩니다.

**3. 필수 라이브러리 설치**
   *   💡 **(중요!) PyTorch 먼저 설치하기:** CUDA 버전에 맞는 PyTorch를 [PyTorch 공식 홈페이지](https://pytorch.org/get-started/locally/)에서 찾아 먼저 설치합니다. 이는 GPU 호환성 문제를 예방하는 가장 좋은 방법입니다.
     *   *예시 (CUDA 11.8 환경):*
       ```bash
       pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
       ```
   *   **나머지 라이브러리 설치:** 프로젝트에 포함된 `requirements.txt` 파일을 이용하여 한 번에 설치합니다.
     ```bash
     pip install -r requirements.txt
     ```

---

#### **Part 3: 학습 설정 (Training Configuration)**

학습을 시작하기 전, 어떤 모델을 어떻게 학습시킬지 설정합니다.

**1. API 키 및 계정 준비**
   *   **Hugging Face**: 모델과 데이터셋을 다운로드하고, 학습된 모델을 업로드하기 위해 계정이 필요합니다.
     *   회원가입 후, **Access Tokens** 메뉴에서 `write` 권한을 가진 토큰을 발급받으세요.
   *   **WandB (선택사항)**: 학습 과정을 시각적으로 모니터링하고 싶다면 [Weights & Biases](https://wandb.ai/)에 가입하고 API 키를 발급받으세요.

**2. `.env` 파일 생성**
   *   프로젝트 최상단에 `.env` 라는 이름의 파일을 새로 만드세요.
   *   발급받은 키들을 아래 형식으로 파일에 저장합니다. (이 파일은 Git에 올라가지 않아 안전합니다.)
     ```
     HUGGINGFACE_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"
     WANDB_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
     ```

**3. 데이터셋 준비**
   *   학습시킬 데이터를 **JSON Lines (`.jsonl`) 형식**으로 준비합니다.
   *   준비된 데이터셋을 본인의 Hugging Face Hub에 업로드합니다.

**4. `config.json` 파일 수정**
   *   이 파일은 학습의 모든 것을 제어하는 **설계도**입니다.
   *   **[필수 수정]**
     *   `hf_username`: 본인의 Hugging Face ID로 변경
     *   `dataset_id`: 3번에서 업로드한 본인의 데이터셋 ID로 변경 (예: "my-username/my-interview-dataset")
     *   `new_model_name`: 학습 후 저장될 모델의 이름 지정 (예: "qwen3-8b-my-interviewer-v1")
   *   **[성능/환경에 따른 수정]**
     *   `base_model_id`: VRAM이 부족하다면 `"Qwen/Qwen3-8B"`를 `"Qwen/Qwen3-4B"` 등으로 변경
     *   `num_train_epochs`: 데이터셋이 작다면 `3`으로 늘려 더 학습시키기
     *   `per_device_train_batch_size`, `gradient_accumulation_steps`: VRAM 부족 오류 시 `batch_size`를 `1`로 줄이고 `accumulation_steps`를 `8`이나 `16`으로 늘리기

---

#### **Part 4: 파인튜닝 실행 및 후처리**

모든 준비가 끝났습니다. 이제 모델을 학습시키고 완성합니다.

**1. 학습 시작**
   *   모든 설정이 완료되었는지 확인 후, 터미널에서 다음 명령어를 실행합니다.
     ```bash
     python train.py
     ```
   *   스크립트가 실행되면 자동으로 `.env` 파일의 키로 로그인하고, `config.json` 설정에 따라 데이터셋과 기본 모델을 다운로드한 후 학습을 시작합니다. 학습 과정은 터미널과 WandB 대시보드에서 확인할 수 있습니다.
   *   ⚠️ 학습은 GPU 성능과 데이터셋 크기에 따라 수십 분에서 수 시간이 소요될 수 있습니다.

**2. 모델 병합 (Merge)**
   *   학습이 완료되면 결과물로 'LoRA 어댑터'가 생성됩니다. 추론(실제 사용) 시 더 빠르고 간편하게 사용하기 위해 기본 모델과 이 어댑터를 하나로 합칩니다.
   *   터미널에서 다음 명령어를 실행하세요.
     ```bash
     python merge_model.py
     ```
   *   `config.py` 파일에 정의된 `MERGED_MODEL_PATH` 경로에 병합된 모델이 저장됩니다.

**3. (선택) 병합된 모델 업로드**
   *   병합된 모델을 Hugging Face Hub에 올려두면, 나중에 API 서버 등 다른 환경에서 쉽게 다운로드하여 사용할 수 있습니다.
   *   `upload_model_to_hf.py` 파일을 열어 `HF_USERNAME`과 `REPO_NAME`을 원하는 값으로 수정합니다.
   *   터미널에서 다음 명령어를 실행하세요.
     ```bash
     python upload_model_to_hf.py
     ```

**축하합니다!** 이제 당신만의 AI 기술 면접관 모델이 완성되어 Hugging Face Hub에 업로드되었습니다. 이 모델을 사용하여 API 서버를 구축하거나 다양한 응용 프로그램을 만들 수 있습니다.