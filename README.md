# Autoencoders Robustness
Badanie odporności różnych architektur autoenkoderów (AE, U-Net, VQ-VAE) na szum i zdolności rekonstrukcji przy użyciu zbioru ImageNet.

## Spis treści
1. [Opis](#opis)
2. [Wymagania](#wymagania)
3. [Struktura projektu](#struktura-projektu)
4. [Sposób uruchomienia](#sposób-uruchomienia)
5. [Trening i ewaluacja](#trening-i-ewaluacja)
6. [Wyniki](#wyniki)
7. [Licencja](#licencja)

## Opis

Projekt **Autoencoders Robustness** bada odporność różnych architektur autoenkoderów na zakłócenia oraz ich zdolności rekonstrukcji obrazu. W eksperymentach porównano klasyczne autoenkodery konwolucyjne, modele U-Net, sieci resztkowe oraz warianty VQ-VAE (Vector Quantized Variational Autoencoder). Modele trenowane są w trzech trybach: na czystych danych, z zaszumionym wejściem oraz z zakłóceniami w przestrzeni latentnej. Do treningu i ewaluacji wykorzystano zestaw danych ImageNet oraz metryki MSE, PSNR i SSIM.

## Wymagania i instalacja

### Wymagania systemowe
- Python 3.10+
- GPU z CUDA (rekomendowane: ≥12 GB VRAM)
- CPU (rekomendowane: ≥30 core'ów)

### Wymagane biblioteki Python
Wszystkie zależności znajdują się w pliku `requirements.txt`. Główne biblioteki:
- `torch`
- `torchvision`
- `torchmetrics`
- `numpy`
- `pandas`
- `matplotlib`
- `tqdm`
- `scikit-image`

### Instalacja

1. Utwórz i aktywuj środowisko:
   ```bash
   python -m venv venv
   
   #Linux
   source venv/bin/activate   
   #Windows
   source venv\Scripts\activate
   
2. Zainstaluj wymagane biblioteki:
   ```bash
   pip install -r requirements.txt
   
3. Zweryfikuj, że torch widzi GPU:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   
## Struktura katalogów

.
├── configs/                # Pliki konfiguracyjne modeli i treningu (JSON)
│   ├── adversarial/
│   ├── conv/
│   ├── residual/
│   ├── unet/
│   ├── vqv/
│   └── vqv2/
│
├── models/                # Implementacje architektur AE (PY)
│   ├── adversarial/
│   ├── conv/
│   ├── residual/
│   ├── unet/
│   ├── vqv/
│   └── vqv2/
│
├── training/              # Skrypty treningowe
│   ├── train_clean.py
│   ├── train_noisy.py
│   └── train_noisy_latent.py
│
├── evaluation/            # Ewaluacja jakości rekonstrukcji i odporności
│   └── evaluate.py
│
├── data/                  # Przygotowanie i ładowanie zbioru danych
│   ├── prepare_subset.py
│   └── data_setter.py
│
├── utils/                 # Narzędzia pomocnicze (logowanie, metryki, rysowanie, seed, itd.)
│
├── results/               # Folder z wynikami (metryki TXT, wykresy,  obrazy PNG)
│
├── requirements.txt       # Lista wymaganych bibliotek Python
│
├── device_check.txt       # Skrypt do sprawdzania GPU
│
└── main.py                # Główny punkt wejścia – ładuje konfigurację i uruchamia trening lub ewaluację

## Sposób uruchomienia

### Argumenty:

- Argumenty:
   ```bash
   python main.py --mode <tryb> --model <model> --type <wariant> --input <dane> [--log]
   
### Opis:

1. --mode 

tryb (tryb działania):
- train – trening modelu,
- test – ewaluacja modelu (na zbiorze testowym),
- train_test – połączenie treningu i ewaluacji (jeden po drugim).

2. --model 

model (nazwa modelu lub grupa modeli), np.:
- conv_transpose_ae_512
- residual_ae_256
- vq_v_ae_512
- unet (dla całej grupy modeli models/unet/)
- all (dla wszystkich modeli)

3. --type 

wariant (rodzaj danych wejściowych):
- clean – czyste dane (bez szumu),
- noisy – dane z szumem Gaussowskim (na wejściu),
- noisy_latent – szum w przestrzeni latentnej,
- all – wykonuje trening lub test dla wszystkich trzech wariantów po kolei.

4. --input 

dane (źródło danych):
- subset – podzbiór ImageNet (np. wybrane klasy),
- full – pełny zbiór ImageNet (potrzebuje dużej ilości RAM/dysku).

5. --log (opcjonalnie)

(włącza szczegółowe logowanie procesu)

## Przykłady użycia

1. Trening pojedynczego modelu "conv_transpose_ae_512" na czystych danych z logami:
   ```bash
   python main.py --mode train --model conv_transpose_ae_512 --type clean --input subset --log
   
2. Trening całej grupy "residual" na wszystkich typach danych:
   ```bash
   python main.py --mode train --model residual --type all --input subset
   
3. Pełny trening + ewaluacja dla modelu vq_v_ae2_512
   ```bash
   python main.py --mode train_test --model vq_v_ae2_512 --type clean --input full
   
4. Ewaluacja wszystkich modeli z folderu unet na zaszumionych danych:
   ```bash
   python main.py --mode test --model unet --type noisy --input subset