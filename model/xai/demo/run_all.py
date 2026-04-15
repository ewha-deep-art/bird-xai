"""
모든 모델 준비 + 시각화를 순서대로 실행

Usage:
  uv run python run_all.py              # 로컬 학습 + 시각화
  uv run python run_all.py --load       # HuggingFace 다운로드 + 시각화
  uv run python run_all.py --viz-only   # 저장된 모델로 시각화만
"""
import sys
import subprocess
from pathlib import Path

ROOT    = Path(__file__).parent
MODELS  = ROOT / 'models'
OUTPUTS = ROOT / 'outputs'

TRAIN_SCRIPTS = [
    ('CNN 학습',       ROOT / 'train' / 'train_cnn.py',       MODELS / 'mnist_cnn.keras'),
    ('GAN 학습',       ROOT / 'train' / 'train_gan.py',        MODELS / 'generator.weights.h5'),
    ('Diffusion 학습', ROOT / 'train' / 'train_diffusion.py',  MODELS / 'unet.weights.h5'),
]

LOAD_SCRIPTS = [
    ('CNN 로드 (CPU 학습)',     ROOT / 'loaders' / 'load_cnn.py',       MODELS / 'mnist_cnn.keras'),
    ('GAN 로드 (CPU 학습)',     ROOT / 'loaders' / 'load_gan.py',        MODELS / 'generator.weights.h5'),
    ('Diffusion 로드 (HF)',    ROOT / 'loaders' / 'load_diffusion.py',  MODELS / 'ddpm-mnist'),
]

# --viz-only 모드에서 확인할 체크포인트 목록
# (로컬 학습: unet.weights.h5 / HF 다운로드: ddpm-mnist/)
REQUIRED_CKPTS = [
    ('CNN',       MODELS / 'mnist_cnn.keras'),
    ('GAN',       MODELS / 'generator.weights.h5'),
    ('Diffusion', MODELS / 'unet.weights.h5', MODELS / 'ddpm-mnist'),
]

VIZ_SCRIPTS = [
    ('CNN 필터 & 활성화',              ROOT / 'viz' / '01_cnn_filters.py'),
    ('Gradient Attribution',          ROOT / 'viz' / '02_gradient_attribution.py'),
    ('Perturbation (Occlusion/LIME)', ROOT / 'viz' / '03_perturbation.py'),
    ('Embedding (t-SNE/UMAP)',        ROOT / 'viz' / '04_embedding.py'),
    ('GAN 시각화',                    ROOT / 'viz' / '05_gan.py'),
    ('Diffusion 시각화',              ROOT / 'viz' / '06_diffusion.py'),
    ('SHAP',                          ROOT / 'viz' / '07_shap.py'),
]

VIZ_ONLY = '--viz-only' in sys.argv
USE_LOAD = '--load'     in sys.argv

def run(label, script):
    print(f'\n{"━"*60}')
    print(f'  {label}')
    print(f'  {script.relative_to(ROOT)}')
    print('━'*60)
    result = subprocess.run([sys.executable, str(script)])
    if result.returncode != 0:
        print(f'[오류] {script.name} 실패 (returncode={result.returncode})')
        sys.exit(result.returncode)

# ─── 모델 준비 단계 ────────────────────────────────────────────────────────────
if VIZ_ONLY:
    print('모델 준비 단계 건너뜀 (--viz-only)')
    missing = []
    for entry in REQUIRED_CKPTS:
        name, *paths = entry
        if not any(p.exists() for p in paths):
            missing.append(name)
    if missing:
        print(f'[경고] 다음 모델이 없습니다: {", ".join(missing)}')
        print('       로컬 학습:    uv run python run_all.py')
        print('       HF 다운로드:  uv run python run_all.py --load')
        sys.exit(1)

elif USE_LOAD:
    print('\n[1/2] HuggingFace / CPU 로더로 모델 준비')
    for label, script, ckpt in LOAD_SCRIPTS:
        if ckpt.exists():
            print(f'  ✓ 이미 존재, 건너뜀: {label}')
        else:
            run(label, script)

else:
    print('\n[1/2] 로컬 학습')
    for label, script, ckpt in TRAIN_SCRIPTS:
        if ckpt.exists():
            print(f'  ✓ 이미 학습됨, 건너뜀: {label}')
        else:
            run(label, script)

# ─── 시각화 단계 ───────────────────────────────────────────────────────────────
print('\n[2/2] 시각화')
for label, script in VIZ_SCRIPTS:
    run(label, script)

print(f'\n{"━"*60}')
print(f'  완료! 이미지 저장 위치: outputs/')
print('━'*60)
