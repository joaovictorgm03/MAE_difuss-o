# Pipeline de lesões sintéticas em mamografias

Insere **uma lesão sintética anatomicamente plausível** em cada mamografia
saudável, preservando o resto da imagem **bit a bit**. O motor é o checkpoint
`Likalto4/mammo_lesion-inpainting` — Stable Diffusion v1-5 *inpainting* ajustado
por DreamBooth pelos autores do artigo **MAM-E**
([arXiv:2311.09822](https://arxiv.org/abs/2311.09822) / *Sensors* 24(7):2076).
Aqui só se faz **inferência**; nada é treinado.

```
  ../sem_anomalia/              ESTE DIRETÓRIO (pipeline/)           ../mamas_com_lesao/
  ┌────────────────┐    ┌──────────────────────────────────┐    ┌────────────────────┐
  │ 66 mamografias │ →  │ 1. valida e enquadra pela mama   │ →  │ original/  (nativa)│
  │ PNG 8/16 bits  │    │ 2. mapa de plausibilidade        │    │ 1024/              │
  │ CC, MLO, OE,RL │    │ 3. sorteia a caixa da lesão      │    │ masks/  overlays/  │
  └────────────────┘    │ 4. difusão (inpainting 512×512)  │    │ metadata/ anatomy/ │
                        │ 5. portão de QC (regera se ruim) │    │ manifest.csv       │
                        │ 6. volta à resolução original    │    │ RELATORIO_TECNICO  │
                        └──────────────────────────────────┘    └────────────────────┘
```

---

## Estrutura do projeto

```
/home/joao/MAE/
├── sem_anomalia/            ENTRADA  — 66 mamografias saudáveis (PNG, 141 MB)
│
├── pipeline/                CÓDIGO E DOCUMENTAÇÃO  ← você está aqui
│   ├── README.md              este guia
│   ├── executar_pipeline.sh   roda tudo do início ao fim
│   ├── config.yaml            todos os hiperparâmetros
│   ├── requirements.txt       dependências com versões exatas
│   ├── .venv/                 ambiente Python já instalado
│   ├── mamelesion/            biblioteca (20 módulos)
│   ├── scripts/               13 programas de linha de comando
│   ├── gui/app.py             interface interativa (Gradio)
│   ├── tests/                 testes das garantias críticas (unittest)
│   ├── treinos/               dados divididos, treinos LoRA, avaliações
│   ├── docs/                  documentação detalhada
│   ├── outputs/               (criada sob demanda) GUI, lote experimental, testes de qualidade, inspeção
│   └── referencia/            artigo em PDF + repositório original dos autores
│
├── mamas_com_lesao/         SAÍDA  — lotes gerados
│   ├── lote_misto/            66 lesões: massa, nódulo e densidade alternados
│   ├── lote_nodulos/          66 lesões: só nódulos
│   ├── lote_birads_lora/      63 lesões BI-RADS 4/5 (janela nativa + LoRA)
│   ├── lote_variacoes_v1/     520 lesões: 8 variações por mama
│   └── lote_birads1_v1/       hospedeiras BI-RADS 1 (../birads_1), 4 variações por mama, em partes
│
├── dataset_escolhido/       ENTREGA — as 360 variações aprovadas na auditoria
│   ├── datasetfinal_MAE/      resolução e bits originais + manifesto + auditoria
│   └── dataset_bbox/          as mesmas com caixa desenhada + COCO/YOLO/CSV
│
└── dataset_escolhido_birrad1/  ENTREGA sobre o birads_1, crescendo a cada ~500 aprovadas
                               (RELATORIO_QUALIDADE.md na raiz; ver docs/PROGRAMAS.md §13)
```

---

## 1. Ambiente

**Nesta máquina já está tudo pronto.** O ambiente fica em `pipeline/.venv` e o
checkpoint (4 GB) está no cache do Hugging Face. Teste de fumaça:

```bash
cd /home/joao/MAE/pipeline
.venv/bin/python -c "import torch, diffusers; print(torch.__version__, torch.cuda.is_available())"
# esperado: 2.6.0+cu124 True
```

Versões validadas: Python 3.10.12 · torch 2.6.0+cu124 · diffusers 0.40.0 ·
transformers 5.17.0 · OpenCV 5.0.0 · scikit-image 0.25.2 · scikit-learn 1.7.2 ·
gradio 6.27.0 · GPU RTX 4090. A lista completa está em `requirements.txt`.

<details>
<summary><b>Recriar o ambiente do zero</b> (outra máquina, ou se o venv quebrar)</summary>

```bash
cd /home/joao/MAE/pipeline
python3 -m virtualenv .venv                    # esta máquina não tem ensurepip
.venv/bin/pip install torch==2.6.0 torchvision==0.21.0 \
      --index-url https://download.pytorch.org/whl/cu124
.venv/bin/pip install -r requirements.txt
```

O checkpoint é baixado automaticamente na primeira execução (~4 GB) e
reaproveitado do cache depois. Sem GPU o código cai para CPU em float32
automaticamente — funciona, mas leva minutos por imagem em vez de segundos.
</details>

> **Nunca use `python` do sistema.** Todos os comandos deste guia usam
> `.venv/bin/python`, sempre a partir de `/home/joao/MAE/pipeline`.

---

## 2. Rodar o pipeline completo

Um comando, do início ao fim:

```bash
cd /home/joao/MAE/pipeline
./executar_pipeline.sh meu_lote
```

Isso gera `../mamas_com_lesao/meu_lote/` com as 66 imagens, máscaras, metadados,
manifesto e relatório — e em seguida **verifica** o resultado. Leva de 4 a 6
minutos na RTX 4090 (~3 a 5 s por imagem, mais ~30 s de carga do modelo na
primeira execução).

Variações de uso:

```bash
./executar_pipeline.sh lote_massas  --types mass            # só massas
./executar_pipeline.sh lote_teste   --limit 5               # teste rápido (5 imagens)
./executar_pipeline.sh lote_duplo   --per-image 2           # 2 lesões por imagem
./executar_pipeline.sh lote_forte   --set diffusion.guidance_scale=6.0
./executar_pipeline.sh --ajuda                              # todos os parâmetros
```

O script **recusa sobrescrever um lote existente** — escolha outro nome ou apague
a pasta antes. Se o nome do lote for omitido, ele usa `lote_AAAAMMDD_HHMM`.

### Lesões BI-RADS 4/5 (modo nativo + LoRA)

Um adaptador ajustado nos recortes reais de `birads_4/` e `_download_drive/` gera
lesões suspeitas numa janela em resolução original, em vez do canvas reduzido:

```bash
./executar_pipeline.sh lote_birads --lora treinos/birads_lora_v1
```

Como o adaptador foi treinado, por que a arquitetura mudou e o que as medições
mostram: **[docs/FINETUNING.md](docs/FINETUNING.md)**.

---

## 3. As três etapas, uma a uma

`executar_pipeline.sh` só encadeia os comandos abaixo. Rode-os separadamente
quando quiser controlar cada etapa:

| # | Etapa | Comando | Produz |
|---|---|---|---|
| 1 | **Ensaio sem modelo** *(opcional, recomendado)* | `.venv/bin/python scripts/build_synthetic_dataset.py --input-dir ../sem_anomalia --output-dir ../mamas_com_lesao/meu_lote --dry-run` | `anatomy/` — os mapas de onde a lesão **pode** cair, sem gastar GPU |
| 2 | **Geração** | `.venv/bin/python scripts/build_synthetic_dataset.py --input-dir ../sem_anomalia --output-dir ../mamas_com_lesao/meu_lote` | o dataset completo |
| 3 | **Verificação** | `.venv/bin/python scripts/verify_dataset.py ../mamas_com_lesao/meu_lote --input-dir ../sem_anomalia` | confirma dimensões, bit depth, zero alteração fora da máscara |

A etapa 1 é barata (segundos) e mostra, em `anatomy/`, a região válida em verde e
o peitoral excluído em vermelho. Vale olhar essas figuras antes de rodar o lote
inteiro: é onde se percebe uma segmentação ruim antes de gastar 6 minutos de GPU.

---

## 4. Os programas, na ordem em que se usam

| Ordem | Programa | Para quê | Obrigatório? |
|---|---|---|---|
| 1 | `scripts/build_synthetic_dataset.py` | **gera o dataset** — é o programa principal | **sim** |
| 2 | `scripts/verify_dataset.py` | confere o dataset gerado, sem depender do pipeline | **sim** (é a prova de integridade) |
| 3 | `scripts/auditar_dataset.py` | **auditoria independente**: relê o lote do disco e recalcula 27 critérios de qualidade | antes de entregar um lote |
| 4 | `scripts/montar_dataset_final.py` | monta o dataset de entrega com o que a auditoria aprovou | ao fechar um dataset |
| — | `scripts/inspect_lesion.py` | depurar **uma** imagem: grava 17 PNGs, um por estágio | quando algo sai errado |
| — | `scripts/lote_em_partes.py` | **lote grande** (milhares de lesões) em partes retomáveis, desacoplado do SSH, com entregas incrementais | para o `birads_1` |
| — | `scripts/desenhar_bbox.py` | desenha a caixa de cada lesão sobre as imagens do lote | conferência visual |
| — | `gui/app.py` | desenhar a região à mão e gerar interativamente | exploração manual |
| — | `scripts/batch_generate.py` | lote **experimental** (posição aleatória, saída só em 1024) | comparar hiperparâmetros |
| — | `scripts/run_quality_tests.py` | bateria de métricas + CAD + XAI, gera relatório de qualidade | validação metodológica |
| — | `scripts/prepare_birads_data.py` → `train_lora_birads.py` → `select_checkpoint.py` → `evaluate_birads.py` | ajuste fino BI-RADS | só para (re)treinar o adaptador |

Os comandos exatos, todos os parâmetros e as saídas de cada um estão em
**[docs/PROGRAMAS.md](docs/PROGRAMAS.md)**.

---

## 5. Parâmetros: o que mudar e como

Há **três formas** de alterar qualquer parâmetro, da mais pontual à mais
permanente:

```bash
# a) opção de linha de comando (o que muda com mais frequência)
.venv/bin/python scripts/build_synthetic_dataset.py --types mass --per-image 2 --seed 42

# b) --set CHAVE=VALOR: alcança QUALQUER campo do config.yaml, sem editar arquivo
.venv/bin/python scripts/build_synthetic_dataset.py \
     --set diffusion.guidance_scale=6.0 \
     --set composition.lesion_gain=0.7 \
     --set lesion.types.mass.diameter_mm_range=[15,30]

# c) editar config.yaml: vira o novo padrão de todas as execuções
```

A precedência é **`--set` > `config.yaml` > padrão do código**. Cuidado: mudar um
valor no código Python não tem efeito nenhum se a chave existir no `config.yaml`.

### Os dez parâmetros que mais importam

| Parâmetro | Padrão | Efeito | Como mudar |
|---|---|---|---|
| `--types` | `mass,nodule,density` | tipos gerados, alternados entre as imagens | `--types mass,nodule` |
| `--per-image` | 1 | lesões por imagem | `--per-image 3` |
| `--seed` | 1337 | semente do lote; a mesma semente reproduz a saída inteira | `--seed 42` |
| `--limit` | — | processa só as N primeiras (teste) | `--limit 5` |
| `--placement-attempts` | 4 | posições tentadas antes de desistir da lesão | `--placement-attempts 6` |
| `diffusion.guidance_scale` | 4.0 | intensidade/definição da lesão | `--set diffusion.guidance_scale=6` |
| `diffusion.num_inference_steps` | 40 | qualidade × tempo | `--set diffusion.num_inference_steps=25` |
| `composition.lesion_gain` | 1.0 | `<1` deixa a lesão mais sutil | `--set composition.lesion_gain=0.6` |
| `composition.detail_transfer` | 0.80 | devolve a granulação do detector à região | `--set composition.detail_transfer=0.9` |
| `lesion.types.<tipo>.diameter_mm_range` | por tipo | tamanho clínico da lesão | `--set lesion.types.mass.diameter_mm_range=[12,20]` |

**Referência completa de todas as chaves do `config.yaml`**, com faixa útil e
consequência de cada uma: **[docs/PARAMETROS.md](docs/PARAMETROS.md)**.

---

## 6. O que sai, e o que é cada arquivo

```
../mamas_com_lesao/meu_lote/
├── original/            ← O RESULTADO FINAL: PNG na resolução e bits do original
├── 1024/                  o mesmo resultado no canvas quadrado de 1024×1024
├── masks/original/        máscara da lesão, na resolução nativa (branco = lesão)
├── masks/1024/            máscara no canvas
├── overlays/              painel original | sintética | diferença, com zoom
├── anatomy/               mapa de plausibilidade (auditoria visual da posição)
├── metadata/              1 JSON por imagem: caixa, semente, mm, métricas, QC
├── manifest.csv           1 linha por lesão — abre direto no pandas/Excel
├── RELATORIO_TECNICO.md   validação do lote em números
├── processing.log         log legível da execução
├── processing.jsonl       o mesmo log em JSON, um evento por linha
└── summary.json           contagens finais
```

Para treinar um detector, use `original/` + `masks/original/` (ou as caixas do
`manifest.csv`) — ou, melhor, o dataset já auditado em
[`../dataset_escolhido/`](../dataset_escolhido). Filtrar o que passou no QC:

```bash
.venv/bin/python -c "
import pandas as pd
m = pd.read_csv('../mamas_com_lesao/lote_misto/manifest.csv')
print(m[m.qc_aprovado].groupby('tipo_lesao')[['weber_contrast','confianca']].mean())
"
```

---

## 7. Quando algo dá errado

| Sintoma | Causa | O que fazer |
|---|---|---|
| `ModuleNotFoundError: mamelesion` | rodou de outra pasta ou com o Python do sistema | `cd /home/joao/MAE/pipeline` e use `.venv/bin/python` |
| `CUDA out of memory` | outra coisa ocupando a GPU | `nvidia-smi`; ou `--set diffusion.dtype=float32` na CPU |
| Lesão invisível na saída | o modelo reconstruiu o tecido sem inserir lesão | é o que o QC barra; aumente `--placement-attempts` ou `--set quality.qc_max_attempts=5` |
| Lote com lesões descartadas | QC reprovou em todas as tentativas | veja o motivo: `grep 'descartada' ../mamas_com_lesao/<lote>/processing.log` |
| Quero ver por que uma imagem falhou | — | `.venv/bin/python scripts/inspect_lesion.py ../sem_anomalia/<arquivo>.png --type nodule --seed 7` e olhe o estágio 13 |
| `verify_dataset.py` acusa pixels alterados fora da máscara | bug de composição — **não use o lote** | reporte; o esperado é sempre zero |

---

## 8. Documentação detalhada

| Documento | Conteúdo |
|---|---|
| **[docs/DOCUMENTACAO_TECNICA.md](docs/DOCUMENTACAO_TECNICA.md)** | **visão completa do estado atual**: fluxo de uma lesão, difusão passo a passo, todos os filtros e por que existem, parâmetros ajustáveis, limites |
| **[docs/PROGRAMAS.md](docs/PROGRAMAS.md)** | cada programa: comando exato, todos os parâmetros, saídas, códigos de saída |
| **[docs/PARAMETROS.md](docs/PARAMETROS.md)** | `config.yaml` chave por chave: efeito, faixa útil, o que quebra |
| **[docs/DATASET.md](docs/DATASET.md)** | como a posição é escolhida, como a resolução nativa é preservada, QC, rastreabilidade |
| **[docs/METODOLOGIA.md](docs/METODOLOGIA.md)** | o que foi reproduzido do artigo, decisões que se afastam dele, resultados medidos e limitações |
| **[docs/FINETUNING.md](docs/FINETUNING.md)** | ajuste fino BI-RADS 4/5: dados, arquitetura em janela nativa, decisões medidas, passo a passo, resultados |
| **[../mamas_com_lesao/LEIA-ME.md](../mamas_com_lesao/LEIA-ME.md)** | o que há em cada lote já gerado |
| **[../dataset_escolhido/LEIA-ME.md](../dataset_escolhido/LEIA-ME.md)** | o dataset de entrega: como foi auditado, o que reprovou e por quê |
| `referencia/2311.09822v1.pdf` | o artigo MAM-E |
| `referencia/mam-e/` | repositório original dos autores (código de treino) |

---

## 9. Limites do que este pipeline entrega

1. **O tipo de lesão não é semântico.** O checkpoint foi ajustado com um único
   prompt (`a mammogram with a lesion`) para todas as imagens de treino: o texto
   não carrega o tipo. Os presets `mass`/`nodule`/`density` mudam apenas
   diâmetro, razão de aspecto, *guidance* e ganho.
2. **O gerador deixa assinatura.** Uma sonda linear separa tecido reconstruído de
   tecido real com AUC 0,76. Serve para **aumento de dataset**, não como
   padrão-ouro em estudos de detecção de imagem sintética.
3. **Escala em mm estimada.** Sem `PixelSpacing` (caso dos PNGs), `mm/px` vem da
   hipótese de um detector de 24×30 cm.
4. **Sem microcalcificações.** A lesão nasce em 512×512; a microestrutura fica na
   escala do modelo.
5. **Detecção do peitoral é heurística**, não um modelo treinado — por isso as
   figuras em `anatomy/` existem.

> Imagens sintéticas destinadas a pesquisa e aumento de dataset. **Não devem ser
> usadas para diagnóstico** nem misturadas a arquivos clínicos reais sem marcação
> explícita de origem sintética.
