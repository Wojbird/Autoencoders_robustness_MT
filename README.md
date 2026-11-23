# Autoencoders Robustness

Projekt porównuje jakość rekonstrukcji i odporność autoenkoderów dla obrazów RGB ImageNet2012 skalowanych do `224×224`.

## Protokół danych

Jedyna obowiązująca transformacja obrazu:

```python
transforms.Resize((224, 224))
transforms.ToTensor()
```

Zakres wejścia i wyjścia wynosi `[0,1]`. Modele kończą dekoder warstwą `Sigmoid`. Nie należy dodawać `transforms.Normalize` bez zmiany całego protokołu treningu i ewaluacji.

## Macierz modeli

Rodziny:

- `conv_transpose_ae` — klasyczny konwolucyjny AE,
- `residual_ae` — AE z blokami rezydualnymi,
- `u_shaped_ae` — symetryczny AE bez skip connections,
- `unet_ae` — właściwy U-Net ze skip connections,
- `adversarial_unet_ae` — generator U-Net trenowany z MSE i stratą adversarial,
- `vq_v_ae` — jednopoziomowy VQ-VAE.

Każda rodzina występuje dla `C ∈ {16, 32, 64, 128, 256}`. Centralny tensor ma rozmiar `C×7×7`.

Warianty treningu:

- `clean`,
- `noisy`,
- `noisy_latent` — tylko dla ciągłych reprezentacji; VQ-VAE jest wykluczony.

## Instalacja

Utwórz środowisko wirtualne. Następnie zainstaluj PyTorch i TorchVision za pomocą oficjalnego selektora: https://pytorch.org/get-started/locally/

Pozostałe zależności:

```bash
python -m pip install -r requirements.txt
```

Testy i narzędzia deweloperskie:

```bash
python -m pip install -r requirements-dev.txt
```

Opcjonalne logowanie W&B:

```bash
python -m pip install -r requirements-wandb.txt
```

## Dane

Pełny ImageNet można wskazać zmienną środowiskową `IMAGENET_DIR` albo plikiem `data/root.json`. Lokalny subset powinien znajdować się w `datasets/subset_imagenet/` i mieć strukturę zgodną z `torchvision.datasets.ImageFolder`.

## Trening

Pojedynczy model:

```bash
python main.py --mode train --model unet_ae_64 --type clean --input subset --log
```

Cała rodzina i wszystkie wspierane warianty:

```bash
python main.py --mode train --model conv --type all --input full --log
```

VQ-VAE automatycznie pomija `noisy_latent`:

```bash
python main.py --mode train --model vqv --type all --input full --log
```

## Ewaluacja podstawowa

```bash
python main.py --mode test --model unet_ae_64 --type clean --input subset --eval_on all
```

Metryki MSE, PSNR i SSIM są obliczane osobno dla każdego obrazu, a następnie uśredniane po próbkach.

## Ewaluacja odporności

Obsługiwane warunki:

- clean,
- Gaussian noise na wejściu,
- Gaussian noise w reprezentacji latentnej,
- impulse/salt-and-pepper,
- okluzja.

Rotacja nie należy do obowiązującego protokołu. VQ-VAE nie jest oceniany dla ciągłego ataku latentnego.

```bash
python robustness_main.py --mode all --model all --training-variant all
```

## Testy

```bash
python -m pytest tests/test_model_matrix.py -v
python param_counter.py --model all
```

Testy weryfikują macierz 6×5, budżety parametrów, zakres wyjść, działanie prawdziwego U-Netu oraz ograniczenia VQ-VAE.

## Ważna uwaga metodologiczna

Dla U-Netu centralny tensor `C×7×7` nie jest pełnym wąskim gardłem informacyjnym, ponieważ skip connections przekazują cechy z enkodera do dekodera. Dlatego współczynnik kompresji centralnego tensora nie powinien być interpretowany tak samo jak dla architektur bez skip connections.
