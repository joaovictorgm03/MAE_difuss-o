# Referência do `config.yaml`

Todos os hiperparâmetros do pipeline, seção por seção: o que cada chave faz, a
faixa útil e a consequência de mexer nela.

## Como alterar um parâmetro

Três formas, em ordem de precedência (**`--set` vence `config.yaml`, que vence o
padrão do código**):

```bash
# 1. --set: alcança qualquer chave, sem editar arquivo. Repetível.
.venv/bin/python scripts/build_synthetic_dataset.py \
    --set diffusion.guidance_scale=6.0 \
    --set lesion.types.mass.diameter_mm_range=[15,30]

# 2. editar config.yaml: vira o novo padrão de todas as execuções

# 3. config alternativo, sem tocar no original
.venv/bin/python scripts/build_synthetic_dataset.py --config experimentos/sutil.yaml
```

O valor de `--set` é lido como YAML: `4.0` vira número, `[15,30]` vira lista,
`true`/`false` viram booleano, `null` vira `None`, e texto com espaço precisa de
aspas (`--set 'lesion.prompt=a dense mass'`).

> **Armadilha:** mudar o padrão de um *dataclass* no código Python **não tem
> efeito** se a chave existir no `config.yaml` — o YAML é aplicado por cima.
> Altere no YAML, ou remova a chave de lá.

---

## `paths` — entrada, saída e pesos

| Chave | Padrão | Efeito |
|---|---|---|
| `input_dir` | `../sem_anomalia` | pasta das mamografias saudáveis; relativa à raiz de `pipeline/` |
| `output_dir` | `outputs` | saída da GUI, do `batch_generate.py` e dos testes de qualidade |
| `model_id` | `Likalto4/mammo_lesion-inpainting` | checkpoint no Hugging Face — os pesos publicados pelos autores do artigo |
| `model_local_path` | `null` | caminho para um snapshot local; use para rodar **offline** |

Com `model_local_path: null`, os pesos são resolvidos no cache local do Hugging
Face (`~/.cache/huggingface/hub`) e, se não estiverem lá, baixados (~4 GB).

---

## `preprocessing` — da imagem bruta ao canvas

| Chave | Padrão | Faixa útil | Efeito |
|---|---|---|---|
| `canvas_size` | 1024 | 512–2048 | resolução de trabalho e da saída em `1024/`. Aumentar não aumenta o detalhe da lesão (o modelo continua em 512) |
| `model_resolution` | 512 | **não mexer** | resolução nativa do modelo; o VAE comprime para o latente 64×64 |
| `framing` | `center_crop` | `center_crop` \| `pad` | enquadramento do artigo × preservar a mama inteira. O pipeline de dataset usa recorte pela mama e ignora esta chave |
| `max_value` | `null` | 3500 (Siemens), 4095 (Hologic) | saturação na conversão uint16 → uint8; `null` usa o máximo da imagem |
| `breast_threshold` | 10 | 5–30 | limiar de segmentação do tecido. Suba se o fundo tiver ruído; desça se a mama estiver sendo cortada |
| `detector_long_side_mm` | 295.0 | 240–350 | tamanho assumido do detector, base da estimativa de mm/px em PNGs |
| `mm_per_px` | `null` | — | sobrescreve a estimativa acima com um valor conhecido |

Com DICOM, o `PixelSpacing` real é usado e estas duas últimas chaves são
ignoradas.

---

## `lesion` — o que se pede ao modelo

| Chave | Padrão | Efeito |
|---|---|---|
| `prompt` | `a mammogram with a lesion` | o prompt do treino. **Mudar quase não altera o resultado** — veja a nota abaixo |
| `negative_prompt` | `""` | prompt negativo |
| `lesion_type` | `mass` | tipo padrão quando nenhum é pedido |
| `mask_shape` | `rect` | `rect` (igual ao treino) \| `ellipse`. Elipse é fora da distribuição de treino |
| `snap_to_latent` | `true` | alinha a caixa à grade de 16 px do canvas (8 px no modelo, 1 px no latente) |
| `min_coverage` | 0.98 | fração mínima da caixa que precisa cair dentro da mama |

> **O prompt não carrega o tipo da lesão.** O ajuste fino usou a *mesma* string
> para todas as imagens de treino, então o condicionamento textual não distingue
> massa de nódulo. Quem controla a aparência são os presets abaixo.

### `lesion.types.<nome>` — os presets

Cada tipo tem seis campos. Padrões:

| Campo | `mass` | `nodule` | `density` | Efeito |
|---|---|---|---|---|
| `diameter_mm_range` | 10–25 | 5–12 | 20–40 | tamanho clínico sorteado para a lesão |
| `aspect_range` | 0,80–1,25 | 0,85–1,18 | 0,60–1,60 | razão de aspecto da caixa |
| `guidance_scale` | 4,0 | 4,5 | 3,5 | intensidade/definição |
| `num_inference_steps` | 40 | 40 | 40 | passos de difusão |
| `detail_transfer` | 0,80 | 0,75 | 0,85 | textura do detector reinjetada |
| `lesion_gain` | 1,00 | 1,00 | 0,60 | ganho do sinal (density é atenuado de propósito) |

> **O tamanho final é quantizado, e isso infla lesões pequenas.** A caixa é
> alinhada à grade do espaço latente (`snap_to_latent`): 16 px do canvas de 1024,
> com o canto inicial arredondado para baixo e o final para cima. Os lados
> possíveis são, portanto, 32, 48, 64 px… — cerca de 4,4 mm de passo nestas
> imagens. Um nódulo sorteado com 6 mm vira uma caixa de 32 px ≈ 8,8 mm. Medido
> no `lote_nodulos`: preset de 5–12 mm, diâmetros gravados de **8,6 a 17,6 mm**.
> O valor em `diametro_mm` (manifesto e metadados) é o da caixa **depois** do
> alinhamento; a faixa do preset vale para o sorteio, **antes** dele. Para
> desligar o alinhamento — ao custo de vazamento de meio-tom na borda da
> máscara — use `--set lesion.snap_to_latent=false`.

Alterar um preset:

```bash
--set lesion.types.mass.diameter_mm_range=[15,30]
--set lesion.types.nodule.guidance_scale=5.5
```

**Criar um tipo novo** — basta declará-lo no `config.yaml`; campos omitidos usam
o padrão do código:

```yaml
lesion:
  types:
    macro:                            # tipo novo, invocável com --types macro
      diameter_mm_range: [30.0, 50.0]
      aspect_range: [0.7, 1.4]
      guidance_scale: 3.0
      lesion_gain: 0.8
      description: massa grande de baixa densidade
```

---

### Tipos BI-RADS (`birads4`, `birads5`)

Presets do modo nativo com LoRA. Diferem dos outros num ponto: têm `prompt`
próprio, igual ao usado no ajuste (`a mammogram with a birads 5 lesion`). Com
`native_window.size_source: catalog` (padrão), o tamanho vem dos recortes reais
de treino e `diameter_mm_range`/`aspect_range` são ignorados. `detail_transfer`
é 0: na resolução nativa a granulação do detector já está na imagem.

---

## `native_window` — geração em resolução original

Liga a geração da lesão numa janela em resolução (quase) nativa, em vez do canvas
reduzido. É o modo usado com o LoRA BI-RADS; a razão e as medições estão em
[FINETUNING.md](FINETUNING.md).

| Chave | Padrão | Efeito |
|---|---|---|
| `enabled` | `false` | `true` liga o modo nativo. `--lora` e `--nativo` fazem o mesmo na CLI |
| `context_factor` | 2.0 | lado da janela / maior lado da lesão |
| `min_scale` | 0.75 | a lesão nunca é reduzida abaixo disso; acima, o contexto encolhe |
| `min_context` | 1.25 | contexto mínimo quando o teto de redução atua |
| `mask_margin_frac` | 0.10 | margem da máscara em volta da lesão |
| `feather_frac` | 0.08 | transição da composição (fração do lado da caixa) |
| `margin_texture_sigma` | 3.0 | devolve a textura do detector na margem regenerada em volta da lesão (σ em px do modelo); `0` desliga. Não depende do treino |
| `density_floor` | -3.0 | a lesão não escurece o tecido do hospedeiro além de 3 níveis de 0–255 na baixa frequência (evita o halo escuro aprendido dos recortes); `null` desliga |
| `paste_feather_frac` | 0.15 | transição da colagem nas amostras de **treino** |
| `lora_path` | `null` | pasta de um treino (usa o melhor checkpoint) ou de um checkpoint |
| `lora_scale` | 1.0 | intensidade do adaptador |
| `guidance` | `null` | `null` = `melhor_guidance` do treino ou, sem ela, a do preset |
| `size_source` | `catalog` | `catalog` = tamanhos dos recortes reais (exige `lora_path`); `preset` = faixa em mm |

> **A geometria gravada no treino prevalece.** Com `lora_path`, os seis
> primeiros campos são lidos de `treino.json`, e cada divergência com o
> `config.yaml` vai para o log. O LoRA só vale na geometria em que foi ajustado:
> mudar `context_factor` na inferência colocaria o modelo fora da distribuição
> de treino.

---

## `diffusion` — a inferência

| Chave | Padrão | Faixa útil | Efeito |
|---|---|---|---|
| `num_inference_steps` | 40 | 20–60 | qualidade × tempo. Abaixo de 25 a lesão perde definição; acima de 50 o ganho é marginal |
| `guidance_scale` | 4.0 | 3–7 | quanto o modelo obedece ao prompt. Alto = lesão mais marcada e mais artificial |
| `scheduler` | `dpmsolver` | `dpmsolver` \| `ddim` | amostrador. `dpmsolver` é o do repositório oficial |
| `seed` | 1337 | — | semente padrão; a CLI (`--seed`) tem precedência |
| `dtype` | `float16` | `float16` \| `float32` | precisão. Em CPU o código força `float32` |
| `num_images_per_prompt` | 1 | 1–4 | candidatos por chamada |
| `attention_slicing` | `true` | — | economiza VRAM; desligar acelera um pouco em GPU folgada |

---

## `composition` — como o resultado volta para o canvas

| Chave | Padrão | Faixa útil | Efeito |
|---|---|---|---|
| `mode` | `hybrid` | `hybrid` \| `global` | **`hybrid` é o que garante zero alteração fora da ROI.** `global` reproduz o artigo literalmente: a imagem inteira passa pelo VAE, altera ~50% dos pixels e derruba o SSIM para 0,82 |
| `blend` | `feather` | `feather` \| `poisson` \| `hard` | transição da borda. `hard` deixa emenda visível; `poisson` pode espalhar o nível médio |
| `feather_px` | 24 | 8–48 | largura da transição, limitada a 25% do lado da ROI |
| `detail_transfer` | 0.80 | 0,0–1,0 | reinjeta a granulação do detector na região. **Valor medido por varredura**: com 0,80 a razão de ruído ROI/tecido real vai de 0,76 para 1,00 e a de nitidez de 0,30 para 0,92 |
| `detail_sigma` | 1.5 | 1,0–2,5 | escala da textura reinjetada |
| `border_offset_correction` | `true` | — | corrige o degrau de nível médio na borda da ROI |
| `lesion_gain` | 1.0 | 0,4–1,3 | ganho do sinal da lesão. `<1` produz densidades sutis sem mudar a forma gerada |

---

## `output` — o que gravar (GUI e `batch_generate.py`)

| Chave | Padrão | Efeito |
|---|---|---|
| `size` | 1024 | lado da imagem salva |
| `save_grayscale` | `true` | grava em tons de cinza em vez de RGB |
| `save_mask` | `true` | grava a máscara da lesão |
| `save_metadata` | `true` | grava o JSON de reprodutibilidade |
| `save_comparison` | `true` | grava o painel de comparação |

No `build_synthetic_dataset.py` isso é controlado pelas opções de linha de
comando (`--no-canvas`, `--no-overlay`, `--no-anatomy-map`).

---

## `quality` — métricas e portão de QC

| Chave | Padrão | Efeito |
|---|---|---|
| `qc_enabled` | `true` | liga o portão de qualidade. Desligar (`--no-qc`) grava a melhor tentativa, boa ou não |
| `qc_max_attempts` | 3 | regerações com nova semente antes de trocar de posição |
| `qc_required` | `[weber_contrast, cnr, outside_changed_fraction, roi_delta_ratio]` | critérios **obrigatórios**; os demais são informativos |
| `ring_px` | 40 | anel de tecido de referência ao redor da ROI, usado no contraste e no CNR |
| `glcm_distances` | `[1, 3, 5]` | distâncias da matriz de co-ocorrência (textura) |
| `glcm_levels` | 64 | níveis de cinza na GLCM |
| `patch_size` | 96 | recorte centrado na lesão nas análises com DINOv2 |
| `probe_patch_size` | 40 | recorte da sonda de impressão digital; precisa caber dentro da área reconstruída |
| `n_control_patches` | 200 | recortes de controle amostrados no tecido saudável |
| `dinov2_id` | `facebook/dinov2-base` | modelo do detector de anomalia (baixa ~350 MB na 1ª vez) |
| `dinov2_local_path` | `null` | snapshot local do DINOv2, para rodar offline |
| `cad_scales_mm` | `[5, 8, 12, 18, 25]` | escalas do detector LoG multiescala |
| `cad_top_k` | 10 | quantos candidatos do CAD contam como "detectado" |

### Por que `roi_delta_ratio` está entre os obrigatórios

`weber_contrast` é medido sobre a fração da ROI que ficou **mais clara**, então
acusa contraste positivo mesmo quando o modelo *escureceu* a região — a falha em
que nenhuma lesão é inserida. `roi_delta_ratio` mede o saldo médio de densidade
em **toda** a ROI (mínimo 1%) e é o critério que barra esse caso. Sem ele,
10 a 15 de cada 66 gerações passavam invertidas.

---

## Parâmetros que **não** estão no `config.yaml`

O posicionamento anatômico tem seus próprios valores, em `mamelesion/anatomy.py`
(margem de 8 mm da pele, 6 mm do peitoral, percentil 45 de densidade mínima,
vizinhança glandular de 12 mm). Eles são gravados em cada `metadata/*.json` e no
`RELATORIO_TECNICO.md` do lote. Para mudá-los, edite `AnatomyConfig` no módulo —
veja [DATASET.md](DATASET.md) §3.
