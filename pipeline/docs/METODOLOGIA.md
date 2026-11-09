# MAM-E lesion synthesis — lesões sintéticas em mamografias saudáveis

Implementação da metodologia de **MAM-E: Mammographic synthetic image generation
with diffusion models** (Montoya-del-Angel, Sam-Millan, Vilanova & Martí,
[arXiv:2311.09822v1](https://arxiv.org/abs/2311.09822) / *Sensors* 24(7):2076)
aplicada às mamografias saudáveis de [`sem_anomalia/`](../../sem_anomalia), com
**saída padronizada em 1024×1024**, **interface interativa** e **bateria de
testes de qualidade** sobre a região gerada.

O pipeline gera lesões do tipo **massa**, **nódulo** e **densidade/assimetria
focal** por *inpainting* em uma região escolhida pelo usuário.

---

## 1. O que foi reproduzido do artigo

| Item do artigo | Onde está |
|---|---|
| DICOM → PNG, VOI LUT, MONOCHROME1 | [`preprocessing.dicom_to_gray`](../mamelesion/preprocessing.py) |
| Intensidades uint16 → uint8 em [0, 255] | [`preprocessing.to_uint8`](../mamelesion/preprocessing.py) |
| RGB replicando o canal cinza | [`preprocessing.to_rgb`](../mamelesion/preprocessing.py) |
| Enquadramento `Resize(lado curto) + CenterCrop` | [`preprocessing.build_canvas`](../mamelesion/preprocessing.py) |
| Máscara binária de bounding box (255 dentro, 0 fora) | [`masks.mask_from_box`](../mamelesion/masks.py) |
| Prompt único `"a mammogram with a lesion"` | [`config.yaml`](../config.yaml) |
| SD v1-5 + DreamBooth, UNet de 9 canais, VAE congelado 512→64 | [`inpainting.LesionInpainter`](../mamelesion/inpainting.py) |
| 40 passos, guidance 4.0, DPMSolverMultistep | [`config.yaml`](../config.yaml) |
| Avaliação por CAD + XAI (gradcam, saliency, occlusion) | [`cad.py`](../mamelesion/cad.py) |

**Pesos**: `Likalto4/mammo_lesion-inpainting` — o checkpoint publicado pelos
próprios autores do artigo, já presente no cache local do Hugging Face. Nada é
treinado aqui: o *fine-tuning* DreamBooth descrito na seção 2.3 do artigo é o que
produziu esse checkpoint. Verificado no carregamento: `StableDiffusionInpaintPipeline`,
UNet com `in_channels=9` (latente + máscara + latente mascarado) e VAE com
`sample_size=64`, exatamente a arquitetura da figura 2 do artigo.

### Duas decisões que se afastam do artigo (deliberadas)

**Saída em 1024×1024 com composição híbrida.** O modelo opera em 512×512 (o VAE
comprime para o latente 64×64). O canvas de trabalho é construído em 1024 com o
*mesmo* enquadramento do artigo; o modelo recebe sua redução 2× exata; e apenas a
região mascarada volta para o canvas de 1024. Assim o restante do mamograma
permanece **bit a bit idêntico** à imagem real — que é o que se quer para aumento
de dataset. O modo `global` (imagem inteira passando pelo VAE, como no artigo)
continua disponível em `composition.mode`, e os testes mostram por que ele não é
o padrão: ele altera 50% dos pixels da imagem e derruba o SSIM para 0.82.

**Tipo de lesão por geometria, não por prompt.** O ajuste fino do artigo usou um
único prompt genérico para todas as imagens de treino, logo o condicionamento
textual **não** carrega o tipo da lesão. O controle de tipo aqui é feito pelos
parâmetros que de fato mudam a aparência: tamanho da região em milímetros, razão
de aspecto, *guidance* e ganho do sinal da lesão (presets em `lesion.types`).

---

## 2. Instalação

O ambiente já está instalado em `pipeline/.venv` (Python 3.10.12, torch
2.6.0+cu124, diffusers 0.40.0, transformers 5.17.0, gradio 6.27.0, OpenCV 5.0,
scikit-image, scikit-learn, matplotlib). Todos os comandos abaixo partem de
`/home/joao/MAE/pipeline`:

```bash
cd /home/joao/MAE/pipeline
.venv/bin/.venv/bin/python scripts/batch_generate.py --help
```

Para recriar do zero, e para a lista de versões exatas, veja
[README.md §1](../README.md#1-ambiente) e [`requirements.txt`](../requirements.txt).

Os pesos são resolvidos automaticamente a partir do cache local do Hugging Face
(`resolve_model_path`), sem exigir rede. Para apontar um diretório explícito,
use `paths.model_local_path` no `config.yaml`.

## 3. Uso

### Interface interativa

```bash
.venv/bin/python gui/app.py                                  # usa paths.input_dir do config.yaml
.venv/bin/python gui/app.py --input-dir ../sem_anomalia     # suas próprias imagens
.venv/bin/python gui/app.py --input-dir ../sem_anomalia --output-dir /dados/saida
.venv/bin/python gui/app.py --port 7870 --share              # outra porta / link público
```

Formatos aceitos na pasta de entrada: PNG, JPEG, DICOM (`.dcm`) e TIFF. DICOM
passa por VOI LUT, correção de MONOCHROME1 e usa o `PixelSpacing` real; PNG/JPEG
em 8 ou 16 bits são escalonados para [0, 255] como no artigo.

Fluxo: escolher a mamografia → desenhar a região com o pincel (ou **Sugerir
região**) → escolher o tipo de lesão → **Gerar lesão** → **Salvar 1024×1024**.

A interface mostra, a cada geração, o zoom lado a lado (original × sintética) e
as métricas de qualidade da região, incluindo o veredito do controle de
qualidade. Parâmetros expostos: tipo, *guidance*, passos, semente, ganho da
lesão, transferência de textura, tipo de mistura da borda e prompt.

### Geração em lote

```bash
.venv/bin/python scripts/batch_generate.py --per-image 2
.venv/bin/python scripts/batch_generate.py --types mass,density --per-image 3 --seed 7
.venv/bin/python scripts/batch_generate.py --set diffusion.guidance_scale=6 --no-qc
```

Salva em `outputs/`: `images/` (PNG 1024×1024), `masks/`, `metadata/` (JSON com
geometria, semente e todos os hiperparâmetros), `comparisons/` e `manifest.csv`.

O **portão de QC** mede cada geração e, se a lesão sair sem contraste (o modelo
às vezes apenas reconstrói o tecido) ou fora de faixa, regenera com outra
semente até `quality.qc_max_attempts`.

### Testes de qualidade

```bash
.venv/bin/python scripts/run_quality_tests.py --per-image 3 --xai-cases 9
```

Produz `outputs/quality_report/`: `RELATORIO_QUALIDADE.md`, `metrics.csv`,
`probes.json` e figuras.

### Como biblioteca

```python
from mamelesion import Config, prepare_case, synthesize, save_result
from mamelesion.inpainting import LesionInpainter
from mamelesion.masks import LesionBox

cfg = Config.load()
case = prepare_case("../sem_anomalia/6278_IM-0002_CCE_L.png", cfg)
result = synthesize(case, LesionInpainter(cfg), LesionBox(400, 700, 480, 780),
                    cfg, lesion_type="mass", seed=1337)
save_result(result, case, cfg)
```

---

## 4. Estrutura

```
pipeline/
├── README.md                guia de execução do pipeline
├── executar_pipeline.sh     roda tudo do início ao fim
├── config.yaml              todos os hiperparâmetros
├── requirements.txt         dependências com versões exatas
├── mamelesion/
│   ├── preprocessing.py     DICOM/PNG → canvas 1024, uint8, RGB, máscara da mama
│   ├── masks.py             caixas, máscaras, validação e amostragem em mm
│   ├── anatomy.py           mapa de plausibilidade anatômica da posição
│   ├── inpainting.py        wrapper do pipeline MAM-E (SD v1-5 inpainting)
│   ├── compose.py           volta do 512 para o 1024 (feather/poisson, textura)
│   ├── native.py            volta do canvas para a resolução original
│   ├── generator.py         orquestração e escrita em disco (GUI e lote simples)
│   ├── pipeline.py          pipeline de produção do dataset
│   ├── quality.py           métricas de imagem e critérios de plausibilidade
│   ├── cad.py               CAD clássico, anomalia DINOv2, XAI, sondas lineares
│   ├── report.py            figuras e relatório da bateria de qualidade
│   └── dataset_report.py    relatório técnico do lote
├── gui/app.py               interface Gradio
├── scripts/
│   ├── build_synthetic_dataset.py   geração do dataset (principal)
│   ├── verify_dataset.py            verificação independente do dataset
│   ├── inspect_lesion.py            depuração estágio por estágio
│   ├── batch_generate.py            lote experimental
│   └── run_quality_tests.py         bateria completa + relatório
├── docs/                    esta documentação
├── outputs/                 saídas de teste e inspeção
└── referencia/              artigo em PDF + repositório original dos autores
```

Qualquer campo do `config.yaml` pode ser sobrescrito na linha de comando:
`--set composition.detail_transfer=0.6 --set lesion.types.mass.guidance_scale=5`.
Referência completa em [PARAMETROS.md](PARAMETROS.md).

## 5. Hiperparâmetros que importam

| Parâmetro | Padrão | Efeito |
|---|---|---|
| `diffusion.guidance_scale` | 4.0 | intensidade/definição da lesão |
| `diffusion.num_inference_steps` | 40 | qualidade × tempo (~2 s/geração na RTX 5070) |
| `composition.detail_transfer` | 0.80 | devolve a granulação do detector na região |
| `composition.lesion_gain` | 1.0 | `<1` produz densidades sutis |
| `composition.feather_px` | 24 | largura da transição (limitada a 25% do lado da ROI) |
| `lesion.types.*.diameter_mm_range` | por tipo | tamanho clínico da lesão |
| `preprocessing.framing` | `center_crop` | `pad` preserva a mama inteira |

O valor de `detail_transfer` foi escolhido por varredura medida (0.0 → 1.0 em 15
gerações): com 0.80 a razão de ruído ROI/tecido real passa de 0.76 para 1.00 e a
de nitidez de 0.30 para 0.92, sem perda de contraste da lesão.

`mm/px` é estimado assumindo detector de 24×30 cm quando o PNG não traz
espaçamento; com DICOM, o `PixelSpacing` real é usado. Ajustável em
`preprocessing.mm_per_px`.

---

## 6. Resultados dos testes

O relatório completo (`outputs/quality_report/RELATORIO_QUALIDADE.md`) não está mais no disco; regenere com `.venv/bin/python scripts/run_quality_tests.py --per-image 3 --xai-cases 9` (baixa o DINOv2-base, ~350 MB). Ver
para os números completos. Resumo (41 gerações sobre as 5 mamografias):

- **Fidelidade**: 0 pixels alterados fora da região marcada (composição híbrida).
- **Contraste**: Weber médio 0.55 (massa), 0.35 (nódulo), 0.46 (densidade), todos
  positivos — a lesão é mais densa que o tecido vizinho.
- **Textura**: razões GLCM lesão/tecido real entre 0.92 e 1.42; ruído 0.93 e
  nitidez 0.92 (1.00 = idêntico ao tecido real).
- **Emenda**: excesso de gradiente no contorno de −0.02 em média (0 = borda
  indistinguível da mesma borda na imagem original).
- **CAD clássico** (LoG multiescala): a lesão fica no percentil 98–99 da resposta
  do detector; aparece no top-10 de candidatos em 62% (massa) a 23% (nódulo) dos
  casos.
- **Anomalia DINOv2** (ajustada só com tecido saudável): escore ~1.9× maior na
  região, lesão no percentil 99–100 do tecido.
- **Sondas lineares**: lesão × tecido real AUC 0.90 (detectável, desejado);
  fundo reconstruído × mesmo local na imagem real AUC 0.76, caindo para 0.66 sem
  o nível médio.

### Limitações medidas

1. **O gerador deixa assinatura.** A sonda pareada detecta o tecido reconstruído
   acima do acaso (AUC 0.76; 0.66 só com textura), com desvio sistemático de
   ~4 níveis de cinza. Um radiologista não percebe (o artigo reporta AUROC 0.49
   para leitor humano), mas um classificador treinado para isso percebe. As
   imagens servem para **aumento de dataset**, não como padrão-ouro em estudos
   de detecção de imagem sintética.
2. **Taxa de falha do modelo.** Cerca de 10% das gerações saem sem lesão
   (contraste ≤ 0) ou com densidade exagerada, sobretudo nos tipos `nodule` e
   `density`. Daí o portão de QC com reamostragem.
3. **Resolução efetiva.** A lesão nasce em 512 e é ampliada 2× para o canvas de
   1024; a transferência de textura recupera as estatísticas de ruído, mas a
   microestrutura da lesão continua nascendo na escala do modelo. Isso vale
   sobretudo para microcalcificações, que este pipeline não se propõe a gerar —
   limitação já apontada na conclusão do artigo.
4. **Escala em milímetros estimada.** Sem DICOM, `mm/px` vem de uma hipótese
   sobre o tamanho do detector; os diâmetros em mm são aproximados.
5. **Os mapas de XAI localizam, mas não dominam.** A lesão fica no percentil 97
   do mapa de *saliency* e 89 do Grad-CAM, porém o **pico global** do mapa nunca
   cai na ROI (0% dos casos): a linha da pele e as regiões mais densas do
   parênquima continuam sendo o que mais move o escore de anomalia. Grad-CAM
   sobre ViT é o localizador mais fraco dos três (razão ROI/tecido ≈ 1.0).

### Reprodutibilidade

Toda geração grava um JSON com semente, caixa, escala, prompt e todos os
parâmetros de difusão e composição — basta reexecutar com a mesma semente.

---

## 7. Citação

```bibtex
@Article{s24072076,
  AUTHOR = {Montoya-del-Angel, Ricardo and Sam-Millan, Karla and Vilanova, Joan C. and Martí, Robert},
  TITLE = {MAM-E: Mammographic Synthetic Image Generation with Diffusion Models},
  JOURNAL = {Sensors}, VOLUME = {24}, YEAR = {2024}, NUMBER = {7},
  ARTICLE-NUMBER = {2076}, DOI = {10.3390/s24072076}
}
```

> **Uso**: imagens sintéticas destinadas a pesquisa e aumento de dataset. Não
> devem ser usadas para diagnóstico nem misturadas a arquivos clínicos reais sem
> marcação explícita de origem sintética.
