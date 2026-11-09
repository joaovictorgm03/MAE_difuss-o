# Referência dos programas

Os programas do pipeline: o que cada um faz, o comando exato, todos os
parâmetros e o que produz.

**Todos os comandos partem de `/home/joao/MAE/pipeline` e usam `.venv/bin/python`.**

```bash
cd /home/joao/MAE/pipeline
```

Índice:

1. [`build_synthetic_dataset.py`](#1-buildsyntheticdatasetpy) — gera o dataset (principal)
2. [`verify_dataset.py`](#2-verifydatasetpy) — verifica um dataset gerado
3. [`inspect_lesion.py`](#3-inspectlesionpy) — depura uma imagem, estágio por estágio
4. [`gui/app.py`](#4-guiapppy) — interface interativa
5. [`batch_generate.py`](#5-batchgeneratepy) — lote experimental
6. [`run_quality_tests.py`](#6-runqualitytestspy) — bateria de qualidade
7. [Ajuste fino BI-RADS](#7-ajuste-fino-bi-rads) — `prepare_birads_data.py`, `train_lora_birads.py`,
   `select_checkpoint.py`, `evaluate_birads.py` e as duas análises
8. [`auditar_dataset.py`](#8-auditardatasetpy) — auditoria independente de qualidade de um lote
9. [`montar_dataset_final.py`](#9-montardatasetfinalpy) — monta o dataset de entrega a partir da auditoria
10. [`calibrar_contraste.py`](#10-calibrarcontrastepy) — casa o contraste das lesões com o das reais anotadas
11. [`fidelidade_real.py`](#11-fidelidaderealpy) — confere se cada lesão cai na distribuição das reais
12. [`painel_real_sintetico.py`](#12-painelrealsinteticopy) — painel visual real × sintético, mesmo janelamento
13. [Lote grande em partes](#13-lote-grande-em-partes) — `selecionar_hospedeiras.py`, `lote_em_partes.py`,
    `unificar_partes.py`, `verificar_entrega.py`, `relatorio_qualidade_lote.py`

---

## 1. `build_synthetic_dataset.py`

**O programa principal.** Lê uma pasta de mamografias saudáveis e grava um
dataset completo com uma (ou mais) lesão por imagem.

```bash
.venv/bin/python scripts/build_synthetic_dataset.py \
    --input-dir ../sem_anomalia \
    --output-dir ../mamas_com_lesao/meu_lote
```

### Parâmetros

| Parâmetro | Padrão | Efeito |
|---|---|---|
| `--input-dir PASTA` | `../sem_anomalia` (de `config.yaml`) | pasta com as mamografias saudáveis |
| `--output-dir PASTA` | `../mamas_com_lesao/lote_AAAAMMDD_HHMM` | onde gravar o dataset |
| `--per-image N` | 1 | lesões por imagem; com N>1 o sufixo vira `_com_lesao_01`, `_02`… |
| `--types LISTA` | `mass,nodule,density` | tipos usados, alternados entre as imagens |
| `--seed N` | 1337 | semente do lote; a mesma semente reproduz a saída inteira |
| `--canvas-size N` | 1024 | lado do canvas de trabalho e da saída em `1024/` |
| `--limit N` | — | processa só as N primeiras imagens (teste rápido) |
| `--placement-attempts N` | 4 | posições tentadas antes de desistir de uma lesão |
| `--dry-run` | — | valida a entrada e grava os mapas anatômicos **sem** rodar o modelo |
| `--no-qc` | — | desliga o portão de qualidade (grava a melhor tentativa) |
| `--no-canvas` | — | não gravar a saída em 1024 |
| `--no-overlay` | — | não gravar os painéis de comparação |
| `--no-anatomy-map` | — | não gravar as figuras de auditoria anatômica |
| `--config ARQUIVO` | `config.yaml` da raiz | usa outro arquivo de configuração |
| `--set CHAVE=VALOR` | — | sobrescreve qualquer campo do `config.yaml` (repetível) |
| `--verbose` / `-v` | — | log DEBUG também no console |

### Exemplos

```bash
# ensaio: valida a entrada e os mapas anatômicos, sem GPU
.venv/bin/python scripts/build_synthetic_dataset.py --dry-run --output-dir /tmp/ensaio

# só massas, 3 por imagem, semente fixa
.venv/bin/python scripts/build_synthetic_dataset.py \
    --types mass --per-image 3 --seed 42 --output-dir ../mamas_com_lesao/lote_massas

# lesões mais sutis e maiores, com guidance mais alto
.venv/bin/python scripts/build_synthetic_dataset.py \
    --set composition.lesion_gain=0.6 \
    --set diffusion.guidance_scale=6.0 \
    --set lesion.types.mass.diameter_mm_range=[18,32] \
    --output-dir ../mamas_com_lesao/lote_sutil
```

### Saída

`original/` · `1024/` · `masks/original/` · `masks/1024/` · `overlays/` ·
`anatomy/` · `metadata/` · `manifest.csv` · `RELATORIO_TECNICO.md` ·
`processing.log` · `processing.jsonl` · `summary.json`

### Códigos de saída

| Código | Significado |
|---|---|
| 0 | tudo processado |
| 1 | concluído com falhas parciais (veja `processing.log`) |
| 2 | nada produzido — erro de ambiente ou configuração |

### Custo

~3 a 5 s por imagem na RTX 4090, mais ~30 s para carregar o modelo. As 66
imagens levam de 4 a 6 minutos.

---

## 2. `verify_dataset.py`

Confere, **sem depender do pipeline**, as garantias que o relatório técnico
afirma. É a prova independente de que o lote está íntegro — rode sempre depois de
gerar e depois de qualquer movimentação de arquivos.

```bash
.venv/bin/python scripts/verify_dataset.py ../mamas_com_lesao/lote_misto \
    --input-dir ../sem_anomalia
```

| Parâmetro | Obrigatório | Efeito |
|---|---|---|
| `dataset` (posicional) | sim | pasta gerada por `build_synthetic_dataset.py` |
| `--input-dir PASTA` | sim | pasta das mamografias de origem |
| `--quiet` / `-q` | não | só o resumo final, sem uma linha por imagem |

Verifica, para cada lesão do manifesto:

1. a saída existe, abre, e tem **exatamente** as dimensões e a profundidade de
   bits do arquivo de origem;
2. **zero pixels alterados fora da máscara** da lesão;
3. há alteração de fato **dentro** da máscara (a lesão não é vazia);
4. `manifest.csv` e os metadados batem com os arquivos em disco.

Código de saída 0 quando tudo passa, 1 quando alguma checagem falha.

---

## 3. `inspect_lesion.py`

Depura **uma** imagem: grava 17 PNGs numerados, um por estágio do pipeline, em
`outputs/inspecao/<imagem>/`. As funções chamadas são as mesmas de
`pipeline.py`, na mesma ordem — o que se vê aqui é o que roda no lote.

```bash
.venv/bin/python scripts/inspect_lesion.py ../sem_anomalia/6278_IM-0005_CCD_R.png \
    --type nodule --seed 7
```

| Parâmetro | Padrão | Efeito |
|---|---|---|
| `imagem` (posicional) | — | a mamografia a inspecionar |
| `--type TIPO` | `nodule` | `mass`, `nodule` ou `density` |
| `--seed N` | — | troca a geração |
| `--box x1,y1,x2,y2` | — | fixa a posição no canvas em vez de sortear |
| `--canvas-size N` | 1024 | lado do canvas |
| `--out PASTA` | `outputs/inspecao/` | onde gravar |
| `--etapas` | — | só imprime a tabela, não grava imagens |
| `--config ARQUIVO` | `config.yaml` | outro arquivo de configuração |

Estágios mais úteis:

| # | Arquivo | O que mostra |
|---|---|---|
| 08 | `score_plausibilidade` | onde a lesão **pode** cair (produto dos quatro fatores) |
| 11 | `entrada_modelo` | 512×512 com a região apagada — **o que o modelo vê** |
| 13 | `diferenca_bruta` | vermelho = densidade adicionada, azul = removida |
| 16 | `delta_lesao` | o que de fato vai para a resolução original |
| 17 | `nativo_antes_depois` | recorte antes/depois na resolução nativa |

**O estágio 13 é o diagnóstico principal**: se sair predominantemente azul, o
modelo removeu densidade em vez de adicionar — a falha que o critério
`roi_delta_ratio` do QC passou a barrar.

---

## 4. `gui/app.py`

Interface Gradio para desenhar a região à mão e gerar interativamente. Exige
`gradio` (já instalado no venv).

```bash
.venv/bin/python gui/app.py                                    # usa paths.input_dir
.venv/bin/python gui/app.py --input-dir ../sem_anomalia        # outra pasta
.venv/bin/python gui/app.py --input-dir /dados --output-dir /saida
.venv/bin/python gui/app.py --port 7870 --share                # outra porta / link público
```

Fluxo: escolher a mamografia → desenhar a região com o pincel (ou **Sugerir
região**) → escolher o tipo → **Gerar lesão** → **Salvar 1024×1024**.

Mostra a cada geração o zoom lado a lado (original × sintética) e as métricas de
qualidade da região, incluindo o veredito do QC. Parâmetros expostos na tela:
tipo, *guidance*, passos, semente, ganho da lesão, transferência de textura, tipo
de mistura da borda e prompt.

Formatos aceitos: PNG, JPEG, TIFF e DICOM (`.dcm`, com VOI LUT, correção de
MONOCHROME1 e `PixelSpacing` real).

> A GUI salva em `outputs/`, **não** em `mamas_com_lesao/` — ela é para
> exploração, não para produzir o dataset final.

---

## 5. `batch_generate.py`

Lote **experimental**, anterior ao pipeline de produção: posição aleatória dentro
da mama (sem mapa anatômico) e saída apenas em 1024. Serve para comparar
hiperparâmetros rapidamente, não para gerar o dataset.

```bash
.venv/bin/python scripts/batch_generate.py --per-image 2
.venv/bin/python scripts/batch_generate.py --types mass,density --per-image 3 --seed 7
.venv/bin/python scripts/batch_generate.py --set diffusion.guidance_scale=6 --no-qc
```

| Parâmetro | Padrão | Efeito |
|---|---|---|
| `--input-dir` / `--output-dir` | do `config.yaml` | entrada e saída |
| `--types LISTA` | `mass,nodule,density` | tipos gerados |
| `--per-image N` | 1 | lesões por imagem **e por tipo** |
| `--seed N` | do `config.yaml` | semente base |
| `--no-qc` | — | desativa o portão de qualidade |
| `--config` / `--set` | — | como no programa principal |

Saída em `outputs/`: `images/`, `masks/`, `metadata/`, `comparisons/`,
`manifest.csv`.

**Diferenças para `build_synthetic_dataset.py`:** sem validação de entrada, sem
posicionamento anatômico, sem saída na resolução nativa, sem log estruturado e
sem recuperação de falha por imagem.

---

## 6. `run_quality_tests.py`

Bateria completa de validação metodológica: métricas de imagem, CAD clássico,
detector de anomalia DINOv2, mapas de XAI e sondas lineares.

```bash
.venv/bin/python scripts/run_quality_tests.py --per-image 3 --xai-cases 9
```

| Parâmetro | Padrão | Efeito |
|---|---|---|
| `--output-dir PASTA` | `outputs/quality_report` | onde gravar o relatório |
| `--types LISTA` | `mass,nodule,density` | tipos avaliados |
| `--per-image N` | 1 | gerações por imagem |
| `--seed N` | do `config.yaml` | semente |
| `--xai-cases N` | — | quantos casos recebem mapas de XAI (*occlusion* é caro) |
| `--no-qc` | — | avalia todas as gerações, inclusive as que o QC rejeitaria |
| `--config` / `--set` | — | como nos demais |

Produz `RELATORIO_QUALIDADE.md`, `metrics.csv`, `probes.json` e figuras.

> **Baixa ~350 MB na primeira execução**: o detector de anomalia usa
> `facebook/dinov2-base`, que não está no cache local. Sem rede, aponte um
> caminho local em `quality.dinov2_local_path`.

Os números da última execução estão resumidos em [METODOLOGIA.md](METODOLOGIA.md). O
relatório completo que ela gerou (`outputs/quality_report/`) não está mais no disco;
rodar o comando acima o recria.

---

## 7. Ajuste fino BI-RADS

Guia completo, com o porquê de cada decisão: [FINETUNING.md](FINETUNING.md).

| Ordem | Programa | GPU | Faz |
|---|---|---|---|
| 1 | `scripts/prepare_birads_data.py` | não | audita recortes, divide por paciente, cataloga tamanhos |
| — | `scripts/analise_escala_vae.py` | sim | mede detalhe preservado por fator de contexto |
| — | `scripts/analise_colagem.py` | não | mede bordas e contraste da colagem |
| — | `scripts/analise_remendo.py LOTE` | não | mede se a margem em volta das lesões de um lote tem a textura do tecido original |
| — | `scripts/desenhar_bbox.py LOTE [--mascara]` | não | cria `LOTE/bbox/`: cada imagem com a caixa da lesão desenhada, mais `caixas.csv` |
| 2 | `scripts/train_lora_birads.py` | sim | treina o LoRA, valida a cada 500 passos |
| 3 | `scripts/select_checkpoint.py` | sim | reavalia checkpoints × guidance na validação e escolhe |
| 4 | `scripts/evaluate_birads.py` | sim | compara canvas × nativo × nativo+LoRA no teste |
| 5 | `scripts/build_synthetic_dataset.py --lora` | sim | produção |

### `prepare_birads_data.py`

| Parâmetro | Padrão | Efeito |
|---|---|---|
| `--classe CLASSE=PASTA` | obrigatório | pasta de recortes de uma classe (`birads4`, `birads5`); repetível |
| `--hospedeiros PASTA` | obrigatório | mamografias saudáveis; usa só as de 16 bits da mesma fonte |
| `--out PASTA` | obrigatório | saída (ex.: `treinos/dados_birads`) |
| `--seed N` | 20260916 | semente da divisão; trocar muda o conjunto de teste |
| `--fracoes a,b,c` | `0.70,0.15,0.15` | treino, validação, teste (por paciente) |
| `--fracoes-hospedeiros a,b,c` | `0.60,0.20,0.20` | o mesmo para os hospedeiros |

Código de saída 1 se detectar paciente em duas partições.

### `train_lora_birads.py`

| Parâmetro | Padrão | Efeito |
|---|---|---|
| `--run PASTA` | obrigatório | pasta do treino; não sobrescreve |
| `--retomar` | — | continua do último checkpoint salvo |
| `--dados PASTA` | `treinos/dados_birads` | saída da fase 0 |
| `--passos N` | 4000 | passos de otimização |
| `--lote N` | 4 | amostras por passo (5,1 GB de GPU) |
| `--lr X` | 1e-4 | taxa de aprendizado (cosseno, aquecimento de `--aquecimento` passos) |
| `--rank N` / `--alpha N` | 16 / 16 | tamanho do LoRA |
| `--peso-fora X` | 0.10 | peso da perda fora da máscara |
| `--min-snr X` | 5.0 | ponderação Min-SNR; 0 desliga |
| `--p-birads4 X` | 0.30 | frequência de BI-RADS 4 nas amostras (natural: 0,20) |
| `--p-vazio X` / `--p-generico X` | 0.10 / 0.20 | prompts vazio e genérico (ver `PromptPolicy`) |
| `--validar-cada N` | 500 | intervalo de validação e checkpoint |
| `--val-sementes N` / `--val-passos N` | 2 / 30 | custo da validação |
| `--val-limite N` | — | valida só N recortes (teste de fumaça) |
| `--workers N` | 6 | processos de amostragem (criados com `spawn`) |

### `select_checkpoint.py`

| Parâmetro | Padrão | Efeito |
|---|---|---|
| `--run PASTA` | obrigatório | treino a reavaliar |
| `--guidance a,b` | `4.0` | valores de guidance a varrer |
| `--sementes N` / `--passos N` | 2 / 30 | lesões por recorte e passos de difusão |
| `--checkpoints a,b` | todos | limita a checkpoints específicos |
| `--sem-base` | — | não avalia o modelo base |

Atualiza `melhor_checkpoint` e `melhor_guidance` em `treino.json`.

### `evaluate_birads.py`

| Parâmetro | Padrão | Efeito |
|---|---|---|
| `--run PASTA` | obrigatório | treino (usa o melhor checkpoint e o `split.json` dele) |
| `--particao` | `test` | `val` durante o desenvolvimento; `test` uma vez, no fim |
| `--bracos` | `atual,nativo_base,nativo_lora` | quais comparar |
| `--sementes N` | 3 | lesões sintéticas por recorte real |
| `--checkpoint PASTA` | melhor | outro checkpoint |
| `--lora-escala X` | 1.0 | intensidade do adaptador |
| `--out PASTA` | `<run>/avaliacao_<particao>` | saída |

### Opções novas de `build_synthetic_dataset.py`

| Parâmetro | Efeito |
|---|---|
| `--lora PASTA` | liga o modo nativo com o adaptador; tipos padrão passam a `birads5:0.83,birads4:0.17` |
| `--types TIPO:PESO,...` | pesos por tipo (qualquer modo). Sem pesos, os tipos se alternam igualmente, como antes |
| `--nativo` | modo nativo sem LoRA (exige `--set native_window.size_source=preset`) |
| `--lora-escala X` | intensidade do adaptador |

### Testes

```bash
.venv/bin/python -m unittest discover -s tests -t . -v
```

---

## 8. `auditar_dataset.py`

Relê do disco o que o lote **entregou** (`original/` + `masks/original/`) e as
mamografias de origem, e **recalcula todas as métricas do zero** — nenhum número
gravado pelo pipeline é reaproveitado. É mais rigoroso que o portão de QC da
geração, que só olha contraste, CNR, saldo de densidade e alteração fora da
máscara: das 520 lesões de `lote_variacoes_v1`, as 160 que esta auditoria
reprovou tinham **todas** passado no portão interno.

```bash
.venv/bin/python scripts/auditar_dataset.py ../mamas_com_lesao/lote_variacoes_v1
```

| Parâmetro | Obrigatório | Efeito |
|---|---|---|
| `lote` (posicional) | sim | pasta do lote (precisa de `metadata/` e `original/`) |
| `--saida PASTA` | não | onde gravar o relatório (padrão: `<lote>/auditoria`) |
| `--ring-mm N` | não | largura do anel de tecido de referência (padrão: 12 mm) |
| `--limit N` | não | audita só as N primeiras mamas (teste) |

27 critérios em cinco grupos, cada um com severidade **bloqueante** (reprova) ou
**aviso** (registra e mantém). O dicionário `CRITERIOS`, no topo do programa, é a
definição autoritativa: grupo, severidade, faixa aceita e descrição de cada um.

| Grupo | O que checa |
|---|---|
| integridade | dimensões e `dtype` iguais aos do original, zero pixels alterados fora da máscara, máscara de uma peça só, caixa do metadado igual à da máscara |
| artefato | degrau na borda da máscara, excesso de gradiente na costura, halo escuro, saturação introduzida, periodicidade de grade do VAE, textura da lesão contra o anel vizinho |
| anatomia | lesão dentro do tecido, distância da pele **medida na resolução nativa**, sobreposição com o peitoral, truncamento na borda, score do mapa de plausibilidade |
| clínica | contraste de Weber, CNR, saldo de densidade na ROI, diâmetro em mm, tamanho dentro do catálogo de recortes reais |
| duplicação | supressão não-máxima entre as variantes da mesma mama (IoU das pegadas e correlação dos recortes) |

Dois critérios merecem nota:

* **`grade_z`** é calibrado contra o próprio hospedeiro: mede a periodicidade de
  grade do espaço latente no recorte da lesão e compara com a mesma medida em 24
  recortes de tecido **real** da mesma mama (z robusto, mediana e MAD). Não
  depende de limiar absoluto, que varia com o equipamento.
* **`textura_roi`** é a média geométrica de ruído e nitidez em relação ao anel.
  Separar os dois não funciona: uma massa densa real é mais homogênea que o
  parênquima e reprovaria num piso de ruído sozinho, enquanto o artefato de véu
  (ou bloco liso) derruba ruído **e** nitidez juntos.

Saídas em `<lote>/auditoria/`: `RELATORIO_QUALIDADE.md` (contagem por critério,
distribuições, lista das reprovadas com o valor que as reprovou),
`auditoria.csv` (uma linha por lesão, todas as medidas) e `auditoria.json` (o
veredito critério a critério). Código de saída 0 se tudo passou, 1 se alguma
lesão reprovou, 2 se o lote não pôde ser lido.

---

## 9. `montar_dataset_final.py`

Monta a pasta de entrega a partir de um lote **já auditado**: copia só as lesões
aprovadas, escreve o manifesto e as anotações em três formatos.

```bash
.venv/bin/python scripts/montar_dataset_final.py ../mamas_com_lesao/lote_variacoes_v1 \
    --saida ../dataset_escolhido
```

| Parâmetro | Obrigatório | Efeito |
|---|---|---|
| `lote` (posicional) | sim | lote com `auditoria/auditoria.csv` |
| `--saida PASTA` | não | destino (padrão: `../dataset_escolhido`) |
| `--max-por-mama N` | não | limita variantes por mamografia, maior confiança primeiro |
| `--split ARQUIVO` | não | divisão de hospedeiros a honrar (padrão: `treinos/birads_lora_v1/split.json`) |
| `--sem-bbox` | não | não gerar `dataset_bbox/` |
| `--fidelidade ARQUIVO` | não | `fidelidade.csv` de [`fidelidade_real.py`](#11-fidelidaderealpy); entra só a lesão aprovada nos **dois** critérios. `auto` = `<lote>/fidelidade/fidelidade.csv` |
| `--calibracao ARQUIVO` | não | `calibracao_contraste.csv`, para gravar no manifesto o ganho aplicado. `auto` = `<lote>/calibracao_contraste.csv` |

Produz `datasetfinal_MAE/` (imagens na resolução e profundidade de bits
originais, máscaras, `manifest.csv`, e a auditoria completa em `qualidade/`) e
`dataset_bbox/` (as mesmas imagens com a caixa desenhada, mais `caixas.csv`,
`coco.json` e anotações YOLO). A partição train/val/test é **por paciente** e
respeita a divisão já usada no treino do LoRA.

Para redesenhar só as caixas de um lote inteiro, sem montar dataset:
`scripts/desenhar_bbox.py <lote> [--mascara]`.

---

## 10. `calibrar_contraste.py`

Ajusta o contraste de cada lesão do lote para a **distribuição das lesões reais
anotadas**, sem gastar GPU. A composição nativa é linear no ganho
(`window.window_delta`: `delta = alpha * gain * diff`), então
`saída(g) = origem + g · (saída(1,0) − origem)` reproduz exatamente o que o
pipeline teria escrito com `lesion_gain = g`. Onde o delta é zero, qualquer
ganho continua zero: **a garantia de zero pixels alterados fora da máscara
sobrevive**, e `verify_dataset.py` continua passando depois da calibração.

```bash
.venv/bin/python scripts/calibrar_contraste.py ../mamas_com_lesao/lote_variacoes_v2 \
    --reais /home/joao/medgan2/dados/birrads_5
```

| Parâmetro | Obrigatório | Efeito |
|---|---|---|
| `lote` (posicional) | sim | pasta do lote (precisa de `metadata/` e `original/`) |
| `--reais PASTA` | não | recortes reais que definem o alvo (padrão: `/home/joao/medgan2/dados/birrads_5`) |
| `--entrada PASTA` | não | mamografias de origem (padrão: `<lote>/../../sem_anomalia`) |
| `--modo {densidade,ganho}` | não | o que reescalar (padrão `densidade`) |
| `--sigma-textura N` | não | σ em px nativos que separa densidade de textura (padrão 3,0) |
| `--simular` | não | só mede e grava o CSV, sem reescrever as imagens |

### Os dois modos, e por que o padrão é `densidade`

| modo | o que faz |
|---|---|
| `densidade` | `delta' = delta + (g−1)·blur(delta, σ)·máscara` — escala só a baixa frequência (a densidade da massa) e deixa a textura fina em amplitude cheia |
| `ganho` | `delta' = g·delta` — idêntico a rodar o pipeline com `lesion_gain = g` |

O modo `ganho` é o caminho óbvio e **custa textura**: um escalar encolhe o ruído
e a nitidez da lesão junto com a densidade. Medido em 130 lesões contra a faixa
das reais (AUC 0,5 = indistinguível):

| descritor | sem calibrar | modo `ganho` | modo `densidade` |
|---|---:|---:|---:|
| `crel` | 0,68 | 0,51 | **0,52** |
| `conspic` | 0,70 | 0,47 | **0,48** |
| `sigma_ratio` | 0,48 | 0,37 | **0,46** |
| `grad_borda` | 0,45 | 0,32 | **0,39** |
| `radialidade` | 0,13 | 0,14 | 0,14 |

Os dois corrigem o contraste igual; só o `densidade` não estraga a granulação e
a borda no caminho. A correção é multiplicada pela **máscara binária** porque o
blur espalha energia para fora dela — sem isso, a garantia de zero pixels
alterados fora da máscara cairia. Como o pipeline já entrega o delta suavizado
até zero na borda (`feather_frac`), ali o blur também é ~0 e o corte não cria
degrau; `verify_dataset.py` confirma depois.

**Por que o descritor é geométrico.** O alvo é o `crel` de
`descritores.medir(recorte, None)`: elipse central de 55% do lado como núcleo,
anel em volta como tecido. Um segmentador adaptativo não serve para comparar:
ele encontra uma região clara em qualquer tecido e mede contraste ≈ 0,27 mesmo
numa mama **sem lesão nenhuma** — medido, varrendo o ganho até zero. Com a
elipse fixa, o mesmo nulo cai para 0,03.

**Casamento por quantil, não pela mediana.** A lesão de posto *r* entre as
sintéticas recebe o quantil (r+0,5)/n das reais. A mediana **e** a dispersão
passam a bater, e a ordem relativa é preservada.

**Três estados possíveis** na coluna `estado` do CSV:

| estado | o que houve |
|---|---|
| `ok` | ganho resolvido dentro de [0,15; 2,50] |
| `ganho_no_limite` | o alvo exigiria ganho fora da faixa; ficou no limite |
| `contraste_nao_controlavel` | `crel(g)` sai plano ou decrescente — a elipse do descritor não cobre o que ficou mais denso. O ganho fica em 1,0 e quem julga é `fidelidade_real.py`. Sem essa guarda, resolver a equação jogaria o ganho ao teto e **pioraria** a lesão (medido: 15 lesões com alvo 0,130 terminando em 0,316) |

Grava `<lote>/calibracao_contraste.csv` com ganho, `crel` antes, alvo e depois de
cada lesão. **Rode antes de `auditar_dataset.py`**, para a auditoria medir o que
será entregue de fato.

---

## 11. `fidelidade_real.py`

Complementa `auditar_dataset.py`. Aquele confere **faixas absolutas** (contraste
entre 0,05 e 1,50, diâmetro entre 5 e 60 mm); este confere se a lesão cai na
**distribuição das lesões reais anotadas**, que é outra pergunta — uma lesão de
contraste 1,2 passa na faixa absoluta e ainda assim é duas vezes mais conspícua
que qualquer BI-RADS 5 do conjunto real.

```bash
.venv/bin/python scripts/fidelidade_real.py ../mamas_com_lesao/lote_variacoes_v2 \
    --reais /home/joao/medgan2/dados/birrads_5 \
    --referencia ../dataset_escolhido/datasetfinal_MAE
```

| Parâmetro | Obrigatório | Efeito |
|---|---|---|
| `lote` (posicional) | sim | pasta do lote |
| `--reais PASTA` | não | recortes reais que definem as faixas p5–p95 |
| `--referencia PASTA` | não | `datasetfinal_MAE` de um lote já entregue, para detectar duplicação **entre** lotes |
| `--iou-max N` | não | sobreposição máxima com uma lesão do lote de referência (padrão 0,50) |

Os dois lados são medidos com o mesmo descritor e o mesmo enquadramento (janela
em que a lesão ocupa 55% do lado). Seis descritores, com severidade fixada pela
separabilidade medida no lote anterior:

| descritor | severidade | por quê |
|---|---|---|
| `crel`, `conspic` | bloqueante | AUC 0,68 e 0,69 — a distribuição sintética está deslocada para cima; é o que `calibrar_contraste.py` corrige |
| `sigma_ratio`, `grad_borda` | aviso | AUC 0,50 e 0,47 — a mediana já bate; o que sai da faixa sai pela cauda, e a faixa real é estreita (±6% em `grad_borda`) |
| `psd_core` | aviso | AUC 0,55, faixa larga |
| `radialidade` | aviso | AUC 0,14 — deslocamento **sistemático e conhecido**: as sintéticas são menos espiculadas que as reais (1,30 contra 1,61). É limitação do gerador, não ruído de medida |

Além dos critérios individuais, `fora_da_faixa_real` (mais de 2 descritores fora
ao mesmo tempo) é bloqueante: um desvio isolado é cauda, três desvios
independentes é lesão fora da distribuição real.

Saídas em `<lote>/fidelidade/`: `fidelidade.csv` e `FIDELIDADE.md`.

---

## 12. `painel_real_sintetico.py`

Painel visual com lesões reais em cima e sintéticas embaixo, **no mesmo
enquadramento e com o mesmo janelamento de exibição** (percentis 0,5 e 99,5 do
próprio recorte).

```bash
.venv/bin/python scripts/painel_real_sintetico.py ../mamas_com_lesao/lote_variacoes_v2
```

O janelamento é o ponto: um recorte real aberto sozinho é auto-janelado e a
lesão salta; a mesma lesão dentro da mamografia inteira parece sutil. Sem
igualar isso, a sintética sempre parece ter menos contraste que a real,
independentemente do que os números digam — foi o que motivou este programa.


---

## 13. Lote grande em partes

Para milhares de lesões (o `birads_1` tem 2000 mamografias), com execução que
sobrevive à queda do SSH, retoma de onde parou e entrega o dataset em etapas.
Foi o que produziu `../dataset_escolhido_birrad1/`.

```bash
# 1. hospedeiras e incidência de cada uma (CPU, ~30 s)
.venv/bin/python scripts/selecionar_hospedeiras.py ../birads_1 \
    --saida ../mamas_com_lesao/lote_birads1_v1/_controle --excluir-de ../sem_anomalia

# 2. geração + cadeia de qualidade + entregas, desacoplado da sessão
L=../mamas_com_lesao/lote_birads1_v1
setsid nohup .venv/bin/python scripts/lote_em_partes.py --entrada ../birads_1 --lote $L \
    --hospedeiras $L/_controle/incluidas.txt --incidencias $L/_controle/incidencias.csv \
    --saida-dataset ../dataset_escolhido_birrad1 --por-imagem 4 --tamanho-parte 60 \
    --calibracao fixo --ganho-fixo 0.85 --entrega-a-cada 500 --semente 20260923 \
    > $L/_controle/nohup.log 2>&1 < /dev/null &

# acompanhar
tail -f $L/_controle/orquestrador.log
cat $L/_controle/entregas.json
```

Rodar **o mesmo comando** de novo retoma: etapa com marcador em
`_controle/estado.json` é pulada; parte com geração interrompida é refeita
inteira (a semente é por parte, então sai igual — verificado byte a byte).

### `selecionar_hospedeiras.py`

O nome do arquivo pode mentir. No `birads_1` a MLO esquerda foi exportada como
`L_CC_1` (e `_2`, `_3`); pelo nome, o pipeline tratava como CC e **não excluía o
peitoral**, e a auditoria, com a mesma função, não perceberia. O programa lê o
rótulo gravado na imagem (`CCE`/`CCD` = 3 letras, `MLOE`/`MLOD` = 4; a razão
largura/altura separa sem sobreposição) e exclui: rótulo fora do padrão (texto
clínico de cirurgia, phantom, outro equipamento), incidência divergente do nome,
duplicata exata e hospedeira já usada numa entrega (`--excluir-de`). Grava
`hospedeiras.csv` (decisão e motivo por imagem), `incidencias.csv`,
`incluidas.txt` e `excluidas.png`.

### Opções novas nos programas existentes

| Programa | Opção | Efeito |
|---|---|---|
| `build_synthetic_dataset.py` | `--incidencias CSV` | tabela `arquivo,view,lateralidade` que prevalece sobre o nome; a fonte fica em `anatomia.view_source` |
| `build_synthetic_dataset.py` | `--lista TXT` | processa só os nomes listados; nome ausente é erro, não imagem pulada |
| `auditar_dataset.py` | `--incidencias CSV` | a MESMA tabela; senão o peitoral é remedido pela incidência do nome |
| `montar_dataset_final.py` | `--vincular` | hardlink em vez de cópia (a entrega não duplica os PNG de 16 bits) |
| `montar_dataset_final.py` | `--particao-estavel` | partição por hash do ID do paciente: não muda quando a entrega cresce |
| `montar_dataset_final.py` | `--entregas JSON`, `--reusar-bbox PASTA`, `--incidencias CSV` | colunas `entrega`/`parte`/`view_fonte`; reaproveita a imagem com caixa já desenhada **se** caixa e tipo forem os mesmos da entrega anterior |
| `montar_dataset_final.py` | `--processos N` (6) | desenha as caixas em paralelo (~2,3 s por imagem de 16 bits) |

Mudanças de comportamento:

* **`auditar_dataset.py` tem um critério novo, `nucleo_acima_tecido`** (artefato,
  bloqueante, ≤ 0,25): fração do núcleo da lesão mais clara que o p99,5 do tecido
  original em volta. Pega a *mancha branca chapada*, que passava em todos os outros
  critérios. Limiar posto olhando as aprovadas do piloto ordenadas pela medida.
  O sorteio dos controles do `grade_z` passou de `hash()` (muda por processo) para
  `crc32`: a auditoria agora é reprodutível.
* **`fidelidade_real.py` tem um critério bloqueante novo, `granulado`**:
  `sigma_ratio` > 1,8 (textura "sal e pimenta" dentro da lesão). Entre a faixa
  das reais (~1,13) e 1,8 continua aviso. Posto olhando a entrega 1: das 16
  entregues acima de 1,8, 15 eram granulado nítido; perto de 1,6 havia lesão boa
  e ruim lado a lado. As 16 saíram na entrega 2, listadas em
  `qualidade/removidas_desde_entrega_anterior.csv`.
* **`calibrar_contraste.py` sai com código 2** se alguma lesão não tiver origem,
  saída ou máscara legível. Antes ela ficava sem calibrar, com estado `nao_medida`
  e código 0.

### O que cada entrega faz

A cada `--entrega-a-cada` lesões aprovadas (auditoria **e** fidelidade), e no fim:
`unificar_partes.py` junta as partes prontas num lote único por hardlink;
`montar_dataset_final.py` monta em `<saida>.novo`; `verificar_entrega.py` confere
cada arquivo contra a origem (dimensões, bits, zero pixel fora da máscara, caixa
dentro da máscara); `relatorio_qualidade_lote.py` escreve `RELATORIO_QUALIDADE.md`,
`rastreabilidade.csv` (toda lesão gerada, com o veredito nas quatro verificações —
artefatos, anatomia, restrições, plausibilidade — e o critério que reprovou), os
painéis e os LEIA-ME; só então a pasta nova **troca** a antiga. Uma entrega que
falha deixa a anterior no lugar. A pasta de entrega só é substituída se tiver o
marcador `.origem_lote` deste lote. A troca são duas renomeações: a pasta some por
milissegundos, e um terminal que estava *dentro* dela fica apontando para a pasta
antiga já apagada (`cd` de novo resolve).

### Calibração neste lote

`--calibracao fixo --ganho-fixo 0.85`, escolhida comparando políticas nas mesmas
91 lesões de um piloto com os mesmos limiares (`_controle/experimento_calibracao.csv`):
1,25 (do v5) aprovou 26%, sem calibração 37%, 0,85 43%, quantil por lesão 36%.
Nestas hospedeiras as lesões já saem mais contrastadas que as reais (o quantil
pede ganho mediano 0,89); o 1,25 foi medido com hospedeiras de `sem_anomalia`.

---

## Ordem de execução recomendada

```bash
cd /home/joao/MAE/pipeline

# 1. ensaio sem GPU: confira as figuras em anatomy/ antes de gastar tempo
.venv/bin/python scripts/build_synthetic_dataset.py --dry-run --output-dir /tmp/ensaio

# 2. teste pequeno de verdade
./executar_pipeline.sh lote_teste --limit 5

# 3. se o teste passou, o lote completo (gera + verifica)
./executar_pipeline.sh lote_final

# 4. auditoria independente do lote (27 critérios, sem GPU)
.venv/bin/python scripts/auditar_dataset.py ../mamas_com_lesao/lote_final

# 5. dataset de entrega, só com o que a auditoria aprovou
.venv/bin/python scripts/montar_dataset_final.py ../mamas_com_lesao/lote_final \
    --saida ../dataset_escolhido

# 6. opcional: bateria de qualidade sobre a metodologia
.venv/bin/python scripts/run_quality_tests.py --per-image 3 --xai-cases 9
```
