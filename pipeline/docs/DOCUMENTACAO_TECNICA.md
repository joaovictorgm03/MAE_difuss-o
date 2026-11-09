# Documentação técnica — geração de lesões sintéticas em mamografias BI-RADS 1

Estado em 2026-09-29. Descreve o pipeline como ele está hoje e como produziu os
lotes `dataset_birrad1_lotes/lote_01` a `lote_06`: da mamografia saudável ao
arquivo entregue, o processo de difusão, cada filtro de qualidade (e por que ele
existe), os parâmetros que mudam a lesão gerada e os limites do que os dados
permitem afirmar.

Todos os comandos partem de `/home/joao/MAE/pipeline` e usam `.venv/bin/python`.
Documentos vizinhos, com mais detalhe em cada tema: [`PROGRAMAS.md`](PROGRAMAS.md)
(todos os programas e opções), [`PARAMETROS.md`](PARAMETROS.md) (`config.yaml`
chave a chave), [`FINETUNING.md`](FINETUNING.md) (treino do LoRA BI-RADS 4/5),
[`METODOLOGIA.md`](METODOLOGIA.md) (reprodução do artigo MAM-E).

---

## 1. Visão geral

O pipeline insere **uma lesão suspeita (BI-RADS 4 ou 5) sintética** numa
mamografia saudável e devolve a imagem **na resolução e profundidade de bits
originais**, alterada só dentro de uma máscara: fora dela, cada pixel é bit a bit
idêntico ao original.

```
 birads_1/ (2000 PNG 16 bits)
      │
      ▼
 1. SELEÇÃO DE HOSPEDEIRAS ── rótulo gravado na imagem → incidência real, exclusões
      │
      ▼
 2. POSIÇÃO ── mapa de plausibilidade anatômica → centro e tamanho da lesão
      │
      ▼
 3. JANELA NATIVA ── recorte da imagem original em volta do local (≤ 512 px no modelo)
      │
      ▼
 4. DIFUSÃO ── SD 1.5 inpainting (MAM-E) + LoRA BI-RADS 4/5, 40 passos, guidance 2,5
      │
      ▼
 5. COMPOSIÇÃO ── textura nativa na margem, piso de densidade, transição suave
      │                → delta somado à imagem ORIGINAL, zero fora da máscara
      ▼
 6. PORTÃO DE QC (na geração) ── lesão invertida/apagada/forte demais → nova semente/posição
      │
      ▼
 7. CALIBRAÇÃO DE CONTRASTE ── casamento por quantil com 92 lesões reais
      │
      ▼
 8. AUDITORIA (31 critérios) + FIDELIDADE ÀS REAIS (9 critérios)
      │
      ▼
 9. SELEÇÃO E ENTREGA ── N imagens estratificadas por contraste × tamanho,
                          verificação contra a origem, relatório, anotações
```

Números do lote atual (política `v4`, 6 lotes): 4545 lesões geradas e auditadas,
~50% aprovadas nas duas avaliações, **2060 entregues** (5 × 400 + 60, escolhidas
entre as aprovadas), de 1420 mamografias distintas e 700 pacientes.

---

## 2. Dados de entrada

| Pasta | Conteúdo | Papel |
|---|---|---|
| `birads_1/` | 2000 PNG 16 bits, BI-RADS 1, 748 pacientes | **hospedeiras**: onde a lesão é inserida |
| `birads_4/` | 16 recortes de lesão BI-RADS 4 (16 bits) | treino do LoRA e **alvo de contraste** |
| `_download_drive/` | 76 recortes de lesão BI-RADS 5 | treino do LoRA e **alvo de contraste** |
| `birads_5/` | vazia | — |

Os recortes reais são justos em volta da lesão e da mesma fonte (mesmo
equipamento, 16 bits) das mamografias do `birads_1`. Por isso o contraste das
lesões geradas pode ser casado com o deles (seção 7).

### 2.1 Seleção de hospedeiras — `scripts/selecionar_hospedeiras.py`

**O nome do arquivo não é confiável.** Nesta fonte, a MLO esquerda foi
exportada como `L_CC_1` (e `L_CC_2`, `L_CC_3`): o rótulo gravado na imagem diz
`MLOE` e o músculo peitoral está visível. Pelo nome, o pipeline trataria 479
imagens MLO como CC e **não excluiria o peitoral**.

O programa lê o rótulo de incidência gravado na própria imagem (texto claro fora
da mama): `CCE`/`CCD` têm 3 letras, `MLOE`/`MLOD` têm 4, e a razão
largura/altura do rótulo separa as duas sem sobreposição (2,7–2,9 contra
4,07–4,29). A incidência resultante vai para `incidencias.csv`, que o gerador e a
auditoria leem com `--incidencias` (tabela prevalece sobre o nome).

| Exclusão | Imagens | Motivo |
|---|---:|---|
| `rotulo_nao_padrao` | 44 | texto clínico gravado (segmentectomia, quadrantectomia, mastectomia, cirurgia, radio/quimio, cicatriz), phantom de calibração, rótulo de outro equipamento |
| `incidencia_divergente` | 7 | rótulo e nome discordam |
| `ja_usada_em_entrega` | 8 | já foram hospedeiras em entregas antigas |
| **aceitas** | **1941** | 972 CC, 969 MLO, 744 pacientes |

---

## 3. Posição da lesão — `mamelesion/anatomy.py`, `mamelesion/window.py`

A mamografia é enquadrada pela mama num canvas de 1024 px, e sobre ele se
constrói um **mapa de plausibilidade** em [0, 1], produto de quatro fatores:

| Fator | Regra |
|---|---|
| pele | zero a menos de 8 mm da linha de pele (`min_skin_distance_mm`), rampa até 20 mm |
| peitoral | zero no músculo peitoral + 6 mm de margem (`pectoral_margin_mm`), só em MLO |
| densidade | percentil de intensidade dentro da mama: zero abaixo de 45 (gordura), rampa até 70; **v3/v4:** o peso cai do percentil 60 ao 100 até 0,4 (`density_upper_percentile`, `density_upper_floor`) |
| vizinhança glandular | fração de tecido denso num raio de 12 mm ≥ 0,2 (`min_glandular_fraction`; era 0,35) |

O centro é sorteado com probabilidade proporcional ao mapa. A caixa da lesão é
aceita se o score médio dela for ≥ 0,35 e se ela estiver inteira na mama, **sem
tocar o peitoral + margem nem ficar a menos de 8 mm da pele** — essas duas
exclusões são checadas explicitamente.

**Por que a caixa pode conter gordura (`min_box_score = 0`).** Com o padrão
antigo (todo pixel da caixa com score ≥ 0,15), só caixas inteiras dentro do
tecido denso eram aceitas, e mudar o peso do centro não mudava nada: as lesões
ficavam no percentil ~85 de densidade, mascaradas. Liberando a caixa, a
densidade mediana no local caiu para ~73 e as lesões até o percentil 75 subiram
de 11% para 63%, sem nenhuma violação de peitoral ou pele.

**Tamanho e proporção** vêm do **catálogo de lesões reais** do treino do LoRA
(`native_window.size_source: catalog`): cada lesão sorteia o tamanho em pixels de
um recorte real. A proporção entre BI-RADS 5 e 4 é a dos 92 recortes
(`birads5:0.83,birads4:0.17`).

---

## 4. O processo de difusão

### 4.1 O modelo

| Componente | O que é |
|---|---|
| Base | `Likalto4/mammo_lesion-inpainting`: Stable Diffusion 1.5 **inpainting**, ajustado por DreamBooth pelos autores do MAM-E (arXiv:2311.09822) em mamografias com lesões anotadas |
| Adaptador | LoRA `treinos/birads_lora_v1`, posto 16 nas camadas de atenção do UNet (3,2 M parâmetros), treinado em 59 recortes BI-RADS 4/5 colados em tecido saudável; checkpoint `passo_002500`. VAE e codificador de texto congelados |
| Prompt | `a mammogram with a birads 5 lesion` / `... birads 4 lesion` |
| Amostrador | DPM-Solver multistep, **40 passos**, guidance livre de classificador **2,5** (escolhida na validação do LoRA), float16 na GPU |

### 4.2 Por que uma janela nativa, e não a mamografia inteira

O modelo trabalha em 512×512 px. Reduzir a mamografia inteira a esse tamanho
reduz a lesão ~6×: uma BI-RADS típica viraria ~48 px, e o detalhe que a define
(espículas, microcalcificações) desaparece — correlação do detalhe fino após o
VAE de **0,06**. Numa janela recortada em resolução (quase) nativa, **0,81**.

`window.plan_window` recorta um quadrado centrado na lesão com **2× o maior lado
dela** (`context_factor`). Se esse quadrado passa de 512 px ele é reduzido, mas
nunca abaixo de 0,75× (`min_scale`): quando o contexto exigiria mais, é o
contexto que encolhe, até 1,25× (`min_context`). No lote real:

| Escala da lesão no modelo | Lesões | Lado típico da lesão |
|---|---:|---:|
| 1,0 (pixels nativos) | 32% | 220 px |
| 0,75–1,0 | 58% | 383 px |
| < 0,75 (lesões > ~550 px, contexto já no mínimo) | 10% | 623 px |

A geometria (`context_factor`, `min_scale`, `min_context`, `mask_margin_frac`) é
**a mesma do treino do LoRA** e é imposta a partir do `treino.json`: mudá-la põe o
modelo fora da distribuição em que foi ajustado.

### 4.3 Uma geração, passo a passo — `nativegen.generate_native`

1. **Recorte.** A janela é lida da imagem de 16 bits, dividida pelo máximo da
   imagem, reamostrada a 512 px (`INTER_AREA`) e quantizada em 8 bits — a entrada
   do modelo é sempre 8 bits.
2. **Máscara.** Um retângulo = pegada da lesão + 10% de margem
   (`mask_margin_frac`), **alinhado à grade de 8 px** do espaço latente
   (`lesion_box_model`). É retangular porque o MAM-E foi treinado com máscaras
   retangulares (bbox das anotações): outra forma de máscara é fora da
   distribuição.
3. **Codificação.** O VAE comprime a janela 8× por lado: latente de 64×64×4. O
   UNet de inpainting recebe 9 canais: o latente ruidoso (4), o latente da imagem
   com a região mascarada apagada (4) e a máscara reduzida (1).
4. **Remoção de ruído.** Partindo de ruído gaussiano (semente fixa por lesão), o
   UNet prevê o ruído em 40 passos do DPM-Solver. A cada passo roda duas vezes —
   com o prompt e com prompt vazio — e combina
   `ε = ε_vazio + g · (ε_prompt − ε_vazio)`, com g = 2,5. Guidance maior deixa a
   lesão mais marcada e menos variada; na validação, 2,5 superou 4,0 em
   semelhança com as reais.
5. **Decodificação.** O VAE devolve 512×512 px. O resultado é convertido para
   cinza.

Fora da máscara, a saída do modelo é descartada: o que interessa é a diferença
que ele fez dentro dela.

### 4.4 Composição — `mamelesion/window.py`

A saída do modelo nunca substitui a imagem. Calcula-se um **delta** e ele é
**somado à imagem original**:

| Etapa | Função | Por quê |
|---|---|---|
| Textura nativa na margem | `restore_native_margin` (σ 3 px) | a máscara vai 10% além da lesão, e o VAE alisa esse tecido (ruído 0,82 do original). Fora do corpo da lesão fica só a baixa frequência da mudança; a textura fina volta a ser a do detector (0,96) |
| Piso de densidade | `floor_low_frequency` (−3 níveis) | o LoRA aprendeu "massa clara + halo escuro" com recortes justos e passava a **escurecer** o tecido em volta; uma lesão real só soma atenuação. Escurecimento ≥ 8 níveis: 22% → 0% |
| Correção de nível | `border_offset` | remove o degrau de nível entre a borda da máscara e o tecido |
| Transição suave | `feather_alpha` (8% do lado) | sem aresta na borda da máscara; estreita de propósito para não apagar espículas |
| Volta à resolução nativa | `writeback` | delta reamostrado por **bilinear** (Lanczos espalha ringing para fora), máscara por vizinho mais próximo, delta **zerado fora da máscara** antes da soma, na profundidade original (16 bits) |

Resultado: `saída = original + delta`, com **zero pixels alterados fora da
máscara** — conferido em toda lesão, três vezes (seção 6).

---

## 5. Portão de QC durante a geração — `mamelesion/quality.py`

Cada geração é medida na janela antes de ser gravada. Se reprova, gera de novo
com outra semente (até 3 vezes, `quality.qc_max_attempts`) e depois em outra
posição (até 4, `--placement-attempts`); esgotado, a lesão é descartada e
registrada.

| Critério | Faixa | Pega |
|---|---|---|
| `weber_contrast` | 0,05 a 1,50 | lesão invisível ou densa demais |
| `cnr` | 0,3 a 12 | lesão que some no ruído ou salta demais |
| `roi_delta_ratio` | ≥ 0,01 | **lesão invertida**: o modelo só redesenhou o tecido ou retirou densidade (era 10–15% das gerações) |
| `outside_changed_fraction` | 0 | qualquer pixel alterado fora da máscara |

É a primeira barreira. Tudo que passa por ela é auditado de novo, do disco.

---

## 6. Filtros de qualidade

Depois de gravadas, as lesões passam por três programas independentes, que
**relêem os arquivos do disco e recalculam tudo** sem usar números do gerador.
Uma lesão só é entregue se passa em **todos** os critérios bloqueantes.

### 6.1 Integridade — `verify_dataset.py` e `verificar_entrega.py`

Rodam após a geração, de novo após a calibração e de novo sobre a pasta
entregue: dimensões e `dtype` iguais aos da origem, **zero pixels alterados fora
da máscara**, algum alterado dentro, caixa do manifesto dentro da máscara.
Qualquer falha tira a parte inteira da entrega.

### 6.2 Auditoria — `scripts/auditar_dataset.py` (31 critérios, 22 bloqueantes)

**Artefatos**

| Critério | Bloqueia se | O que pega |
|---|---|---|
| `textura_roi` | fora de 0,38–1,60 | véu liso ou bloco sem textura: ruído e nitidez da lesão caem juntos em relação ao tecido vizinho (assinatura do VAE). Calibrado olhando os extremos: 0,358 bloco visível, 0,402 densidade plausível |
| `grade_z` | > 6 | periodicidade de grade do espaço latente, comparada com 24 recortes de tecido real **da mesma mama** (z robusto) |
| `halo_u8` | fora de −3,5–12 | halo escuro em volta da lesão |
| `degrau_borda_u8` | \|x\| > 12 | degrau de nível na borda da máscara |
| `costura_excesso` | > 0,60 | costura visível (gradiente extra na borda) |
| `saturacao_frac` | > 2% | lesão saturada no topo da escala |
| `bloco_retangular` | = 1 | lesão compacta (preenche > 70% do retângulo que a envolve) **com cantos cheios** (≥ 0,33): herança das máscaras retangulares do MAM-E. Medido no **acréscimo de densidade** (saída − original), não na imagem, para o tecido do entorno não contaminar. Oval lisa (plausível) tem cantos 0,11–0,15 e passa |
| avisos | — | `ruido_roi_anel`, `nitidez_roi_anel`, `circularidade`, `preenchimento_pegada`, `nucleo_acima_tecido`, `extensao_acrescimo`, `cantos_acrescimo`, `contato_borda` — registrados por lesão, não reprovam |

**Consistência anatômica**

| Critério | Bloqueia se |
|---|---|
| `distancia_pele_mm` | < 8 mm, **medida na resolução nativa** (a medida no canvas saía otimista: 12,4 mm onde a real era 6,3) |
| `peitoral_frac` | > 0 (com a incidência da tabela, não do nome) |
| `cobertura_tecido` | < 98% da lesão sobre tecido mamário |
| `toca_borda_img` | lesão truncada pela borda da imagem |

**Restrições de dados**

| Critério | Bloqueia se |
|---|---|
| `fora_da_mascara_px` | > 0 |
| `dimensoes_ok`, `mascara_componentes`, `caixa_confere`, `mascara_area_px` | imagem ≠ origem; máscara vazia ou em pedaços; caixa do metadado ≠ caixa da máscara |
| `diametro_mm` | fora de 5–60 mm |
| `iou_max_variantes`, `similaridade_max` | sobreposição > 0,5 ou correlação > 0,95 com outra variante da mesma mama (supressão não-máxima: fica a de maior confiança) |

**Plausibilidade clínica (faixas absolutas)**

| Critério | Bloqueia se |
|---|---|
| `weber_nativo` | fora de 0,05–1,50 |
| `cnr_nativo` | fora de 0,3–12 |
| `roi_delta_ratio` | fora de 0,01–1,0 |

### 6.3 Fidelidade às lesões reais — `scripts/fidelidade_real.py` (9 critérios)

Compara cada lesão com a **distribuição** das 92 lesões reais de `birads_4/` +
`_download_drive/` (faixa p5–p95), com o mesmo descritor dos dois lados.

| Critério | Severidade | O que pega |
|---|---|---|
| `crel_afastado` | **bloqueante** | contraste núcleo × coroa de tecido **afastada** (0,75–0,95 do lado) fora da faixa das reais. É a mesma régua da calibração (seção 7) |
| `granulado` | **bloqueante** | `sigma_ratio` > 1,8: textura "sal e pimenta" dentro da lesão. Das 16 entregues acima de 1,8 num lote anterior, 15 eram granulado nítido; entre 1,58 e 1,66 havia lesão boa e ruim lado a lado, por isso o teto não é mais baixo |
| `fora_da_faixa_real` | **bloqueante** | 3 ou mais descritores fora da faixa ao mesmo tempo |
| `crel`, `conspic` (anel imediato) | bloqueante **só por baixo** | lesão apagada. Por cima vira aviso: no recorte real justo, o anel imediato ainda é lesão e deprime o contraste real — reprovar por cima reprovava lesão com contraste de lesão real |
| `sigma_ratio`, `grad_borda`, `psd_core`, `radialidade` | aviso | granulação, borda, espectro e espículas fora da faixa das reais |

### 6.4 Critérios que foram **desligados**, e por quê

- **`nucleo_acima_tecido`** (núcleo mais claro que o p99,5 do tecido em volta)
  foi criado para barrar "manchas brancas chapadas" e **rebaixado a aviso**:
  aplicado às 92 lesões reais, reprovaria **47%** delas — lesão densa real é mais
  clara que o entorno. A "mancha chapada" que o motivou era, em boa parte,
  efeito do janelamento de exibição (percentis do tecido original estouram
  qualquer lesão clara). Exibidas como um recorte real, as lesões mostravam
  textura.
- **`contato_borda`** (lesão encostando nos lados da caixa, medido na imagem):
  não separava bloco de lesão boa — o tecido claro do entorno encostava na caixa.
  Ficou como aviso; a medida que funciona é a do acréscimo (`bloco_retangular`).

Regra que ficou: **limiar novo só entra depois de olhar as imagens dos dois
extremos, e critério de artefato é testado também contra as lesões reais.**

---

## 7. Calibração de contraste — `scripts/calibrar_contraste.py`

A composição é linear no ganho, então o contraste de uma lesão pode ser
reescalado depois, sem GPU, sem tocar em pixel fora da máscara:

```
delta' = delta + (g − 1) · blur(delta, σ = 3 px) · máscara     (modo "densidade")
```

Só a baixa frequência (a densidade da massa) é escalada; a textura fina fica em
amplitude cheia. Escalar o delta inteiro custaria granulação e nitidez.

**Casamento por quantil (política atual).** O contraste de cada lesão é medido
na coroa afastada. A lesão de posto *r* entre as n do lote recebe como alvo o
quantil (r + 0,5)/n das 92 reais. Mediana **e** dispersão passam a bater, e a
ordem relativa entre as lesões é preservada. O ganho fica entre 0,15 e **2,0**
(`--ganho-max`; acima disso a lesão satura). Lesões cujo contraste não responde
ao ganho ficam com ganho 1,0 e são julgadas pela fidelidade.

| Contraste (coroa afastada), p25 / mediana / p75 | |
|---|---|
| lesões reais birads_4 + 5 | 0,339 / 0,508 / 0,737 |
| política antiga (ganho fixo 0,85) | 0,316 / 0,477 / 0,635 — faltava a cauda alta |
| **política atual** | ~0,40 / ~0,51 / ~0,70 |

O ganho fixo 1,25 usado em lotes antigos foi medido com hospedeiras de outra
fonte; nestas, deixava as lesões fortes demais (26% aprovadas, 7% nas MLO).

---

## 8. Seleção e entrega — `montar_dataset_final.py`, `lote_em_partes.py`

- **Uma lesão por mamografia em cada lote**, de mamografias distintas; entre
  lotes, uma mamografia só reaparece depois de todas as outras terem sido
  usadas, sempre com lesão, posição e semente novas. Nos 5 lotes de 400: 1360
  mamografias, 789 com 1 lesão, 502 com 2, 69 com 3.
- **Seleção estratificada** (`--selecionar N`): as aprovadas são divididas em
  células de contraste (5 faixas de `crel_afastado`) × tamanho (3 faixas de
  diâmetro), e cada célula contribui na proporção dela — cada lote cobre de
  sutil a evidente e de pequena a grande.
- **Partição por paciente, estável** (`--particao-estavel`): train/val/test sai
  de um hash do ID do paciente (70/15/15), então o mesmo paciente cai na mesma
  partição em qualquer lote. Pacientes com recorte no treino do LoRA ficam na
  partição do recorte.
- **Nomes únicos** (`--rotulo-saida _L0k`): lotes podem ser juntados sem colisão.

### 8.1 O que cada lote entregue contém

```
lote_0k/
├── RELATORIO_QUALIDADE.md      as verificações, lesão a lesão, com o que foi feito
├── LEIA-ME.md
├── datasetfinal_MAE/           PARA TREINAR
│   ├── imagens/                  PNG 16 bits, resolução da mamografia de origem
│   ├── mascaras/                 região alterada (branco); fora dela, pixel = origem
│   ├── manifest.csv              caixa x1,y1,x2,y2 (px nativos), partição, tipo,
│   │                             incidência, medidas de auditoria e fidelidade
│   └── qualidade/                rastreabilidade.csv (toda lesão gerada, aprovada ou
│                                 não, e o critério que reprovou), auditoria.csv,
│                                 fidelidade.csv, calibracao_contraste.csv,
│                                 verificacao_entrega.csv, paineis/
└── dataset_bbox/               PARA OLHAR
    ├── imagens/                  8 bits com a caixa desenhada
    └── anotacoes/                caixas.csv, coco.json, yolo/, classes.txt
```

---

## 9. Parâmetros que mudam a lesão gerada

**Seguros** — mudam a lesão sem sair da distribuição do modelo:

| Para | Parâmetro | Onde | Padrão atual |
|---|---|---|---|
| proporção BI-RADS 5 / 4 | `--types birads5:P,birads4:Q` | gerador | 0,83 / 0,17 |
| contraste | `--calibracao quantil\|fixo\|nenhuma`, `--ganho-fixo`, `--ganho-max`, `--reais PASTA` | orquestrador | quantil, teto 2,0, 92 reais |
| lesões mais/menos no tecido denso | `--anatomia density_upper_percentile=X`, `density_upper_floor`, `min_glandular_fraction`, `min_box_score` | gerador | 60, 0,4, 0,2, 0 |
| distância mínima da pele / peitoral | `--anatomia min_skin_distance_mm=`, `pectoral_margin_mm=` | gerador | 8 mm, 6 mm |
| variedade entre gerações | `--seed` / `--semente` (por parte) | gerador / orquestrador | fixa por lote |
| lesões por mamografia | `--per-image` / `--por-imagem` | ambos | 1 |
| quantas e quais entregar | `--selecionar N` | orquestrador | 400 |
| intensidade do LoRA | `--lora-escala` | gerador | 1,0 |

**Com cuidado** — mudam o modelo; exigem olhar o resultado:

| Parâmetro | Efeito | Risco |
|---|---|---|
| `native_window.guidance` (`--set`) | guidance > 2,5: lesão mais marcada, menos variada | a validação escolheu 2,5; 4,0 foi pior em semelhança com as reais |
| `diffusion.num_inference_steps` | < 40 é mais rápido e mais ruidoso | não validado abaixo de 40 |
| `native_window.size_source: preset` + `lesion.types.birads5.diameter_mm_range` | tamanhos fora do catálogo real | lesão fora da faixa do treino |
| `margin_texture_sigma`, `density_floor` | 0 / `null` desligam as correções da seção 4.4 | voltam o remendo liso e o halo escuro |

**Não mudar sem retreinar** — o LoRA foi ajustado nesta geometria e ela é
imposta pelo `treino.json`: `context_factor`, `min_scale`, `min_context`,
`mask_margin_frac`, a forma retangular da máscara e o prompt.

**Os filtros** também são parâmetros (faixas em `auditar_dataset.CRITERIOS` e
`fidelidade_real.CRITERIOS`/`GRANULADO_MAX`), mas cada um foi posto olhando
imagens dos dois extremos. Mexer neles muda o que é considerado artefato.

---

## 10. Como rodar

```bash
# hospedeiras e incidência (uma vez)
.venv/bin/python scripts/selecionar_hospedeiras.py ../birads_1 \
    --saida ../mamas_com_lesao/<lote>/_controle --excluir-de ../sem_anomalia

# um lote, desacoplado da sessão SSH e retomável (rodar de novo continua)
L=../mamas_com_lesao/<lote>
setsid nohup .venv/bin/python scripts/lote_em_partes.py --entrada ../birads_1 --lote $L \
    --hospedeiras $L/_controle/incluidas.txt --incidencias $L/_controle/incidencias.csv \
    --saida-dataset ../dataset_birrad1_lotes/lote_07 --selecionar 400 --entrega-a-cada 100000 \
    --por-imagem 1 --tamanho-parte 85 --semente 20267925 --rotulo-saida _L07 --politica v4 \
    --calibracao quantil --ganho-max 2.0 --reais <pasta com os recortes reais> --contraste-afastado \
    --anatomia density_upper_percentile=60 --anatomia min_glandular_fraction=0.2 \
    --anatomia min_box_score=0.0 > $L/_controle/nohup.log 2>&1 < /dev/null &

tail -f $L/_controle/orquestrador.log     # acompanhar
```

O orquestrador divide as mamografias em partes. Cada parte passa por
`gerar → verificar → calibrar → verificar → auditar → fidelidade`, com um
marcador por etapa em `_controle/estado.json`. A GPU gera uma parte enquanto a
CPU pós-processa a anterior. Uma etapa só conta se deixou a saída completa e
nova: o código de saída não basta, porque exceção do Python e "alguma lesão
reprovada" saem ambos com 1. A entrega é montada em `<saída>.novo`, verificada,
e só então troca a pasta anterior. Um lote de 850–920 mamografias leva ~50 min
numa RTX 4090. O disco da máquina é compartilhado: confira `df -h` antes.

Testes (110, incluindo as garantias acima):

```bash
.venv/bin/python -m unittest discover -s tests -t . -v
```

---

## 11. Limites — o que os dados permitem afirmar

1. **O rótulo BI-RADS 4/5 não é validado por classe.** O LoRA viu 12 recortes
   BI-RADS 4; não há evidência de que gere algo distinto por classe.
2. **Assinatura do gerador.** Um classificador em features profundas separa as
   lesões geradas das reais (AUC 0,94); a mesma lesão real só reconstruída pelo
   VAE já dá 0,97. Servem para **aumento de dataset**, não como substituto
   indistinguível de casos reais.
3. **Menos espiculadas que as reais** (`radialidade` mediana 1,36 contra 1,61).
4. **Formato.** O MAM-E foi treinado com máscaras retangulares; o filtro de bloco
   pega só os casos claros (compacta + cantos cheios). Parte das lesões ainda tem
   contorno arredondado-retangular.
5. **A caixa é a pegada, não o contorno.** A máscara é a região redesenhada (com
   margem), não a segmentação da lesão.
6. **Escala em mm estimada** (detector de 29,5 cm no lado maior): os PNG não
   trazem `PixelSpacing`.
7. **Lesões grandes perdem detalhe fino**: acima de ~550 px são geradas a menos
   de 0,75× (10% das lesões).

> Imagens sintéticas para pesquisa e aumento de dataset. **Não devem ser usadas
> para diagnóstico** nem misturadas a arquivos clínicos reais sem marcação de
> origem sintética.

---

## 12. Histórico das decisões

| Versão | O que mudou | Motivo |
|---|---|---|
| canvas (lotes antigos) | lesão gerada na mamografia reduzida a 1024 | reprodução do artigo |
| janela nativa + LoRA | lesão gerada em resolução ~nativa, LoRA BI-RADS 4/5 | no canvas o detalhe da lesão não sobrevive; o modelo base não gera lesão na janela |
| `L_CC_1` = MLO | tabela de incidências pelo rótulo da imagem | o nome mentia; peitoral não era excluído |
| v1 | ganho fixo 0,85 | o 1,25 herdado deixava as lesões fortes demais nestas hospedeiras |
| v2 | contraste casado por quantil com as 92 reais; fidelidade na coroa afastada | faltava a cauda de contraste alto; a régua do anel imediato era enviesada |
| v3 | caixa pode conter gordura | as lesões ficavam presas no tecido mais denso |
| v4 | 1 lesão por mamografia por lote, seleção estratificada, filtro de bloco retangular, granulado bloqueante | variedade, evitar que o modelo decore mama ou formato |
