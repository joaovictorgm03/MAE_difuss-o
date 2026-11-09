# Ajuste fino para lesões BI-RADS 4/5

Como o checkpoint MAM-E foi ajustado para gerar lesões parecidas com os recortes
reais de `birads_4/` e `_download_drive/` (BI-RADS 5), por que a arquitetura é
esta, e como reproduzir cada etapa.

Todos os comandos partem de `/home/joao/MAE/pipeline` e usam `.venv/bin/python`.

---

## 1. O problema, em números

**O modelo não via a lesão.** No modo canvas, a mamografia inteira chega ao
modelo reduzida ~6× (mediana 6,11 nas imagens de 16 bits). Uma lesão BI-RADS
típica vira ~48 px, e o detalhe que define BI-RADS 4/5 (espículas, distorção,
microcalcificações) não sobrevive. Medido nos recortes reais, passando pelo VAE
congelado do checkpoint:

| Escala | SSIM | Correlação do detalhe fino |
|---|---|---|
| Canvas atual (lesão ~48 px no modelo) | 0,60 | **0,06** |
| Janela em resolução nativa | 0,83 | **0,81** |

Com correlação 0,06, nenhum ajuste do UNet recupera o que o VAE já perdeu.
**Por isso o fine-tuning vem junto com uma mudança de arquitetura: a lesão é
gerada numa janela em resolução (quase) nativa, não no canvas.**

**O modelo base não gera lesão nessa janela.** Com a máscara grande da janela
nativa, o checkpoint original só reconstrói tecido comum: peitoral, pele,
gordura. O ajuste fino é necessário, não um refinamento.

---

## 2. Dados

| Origem | Classe | Recortes | Pacientes |
|---|---|---|---|
| `_download_drive/` (`birrads_5_1`, `birrads_5_2`) | BI-RADS 5 | 76 | 45 |
| `birads_4/` | BI-RADS 4 | 16 | 15 |
| `sem_anomalia/` (só os 15 de 16 bits) | hospedeiros saudáveis | 15 | 13 |

Recortes PNG de 16 bits, justos em volta da lesão (lado maior mediano de 367 px,
máximo de 997 px), sem a mamografia inteira de origem. Os 15 hospedeiros de 16
bits têm a mesma convenção de nome e profundidade dos recortes: mesma fonte. Os
outros 51 de `sem_anomalia` (8 bits, `IM-xxxx`) são de outro equipamento e
**não** entram no treino.

### Divisão por paciente

Até 4 recortes vêm do mesmo paciente (a mesma lesão em CC e MLO, por exemplo).
Dividir por recorte vazaria a lesão para o teste. `split_by_patient` divide por
paciente e estratifica por classe:

| Partição | Recortes | BI-RADS 4 (pacientes) | BI-RADS 5 (pacientes) | Hospedeiros |
|---|---|---|---|---|
| treino | 59 | 12 (11) | 47 (31) | 9 |
| validação | 18 | 2 (2) | 16 (7) | 3 |
| teste | 15 | 2 (2) | 13 (7) | 3 |

Semente 20260916. Detalhes em `treinos/dados_birads/AUDITORIA.md`.

---

## 3. Arquitetura

```
recorte real ──► aumento ──► posição anatômica ──► janela nativa ──► colagem suave ──► (imagem, máscara)
(treino)        (espelhos,    (mapa de            (contexto 2×,     (superelipse,       │
                 escala ±10%)  plausibilidade)     teto 0,75)        nível por plano)    ▼
                                                                                 UNet + LoRA
mamografia saudável ─► tamanho do catálogo real ─► posição anatômica ─► janela nativa ─► inpainting ─► delta ─► original
(inferência)                                        (MESMA função)      (MESMA função)    (LoRA)       (só dentro da máscara)
```

Um princípio guia tudo: **treino, avaliação e pipeline usam as mesmas funções**
de geometria, posição e escala de intensidade (`mamelesion/window.py`). Se um
lado usasse outro fator de contexto ou outra grade, o modelo rodaria fora da
distribuição em que foi ajustado, e nenhum erro apareceria.

| Módulo | Papel |
|---|---|
| `mamelesion/window.py` | janela nativa, caixa da máscara na grade do latente, volta do delta à imagem (zero fora da máscara), posicionamento anatômico |
| `mamelesion/birads_data.py` | leitura e auditoria dos recortes, divisão por paciente, catálogo de tamanhos |
| `mamelesion/paste.py` | amostra de treino: recorte colado em tecido saudável |
| `mamelesion/finetune.py` | dataset, política de prompts, LoRA (injetar, gravar, carregar), leitura do treino |
| `mamelesion/nativegen.py` | uma geração nativa completa, QC na janela, painel de comparação |
| `mamelesion/lesion_eval.py` | métricas de semelhança real × sintético e critério de escolha |

### Por que colagem

O ajuste de inpainting precisa de pares (imagem com lesão, máscara) com contexto
real ao redor. Sem as mamografias originais, o recorte é colado numa janela de
tecido saudável **da mesma fonte**. Isso reproduz exatamente o uso na inferência:
contexto saudável, lesão sob a máscara.

**O que a colagem não ensina:** como a lesão deforma o tecido *além* do recorte
(retração, distorção à distância), porque o entorno é de outra paciente. Com as
mamografias inteiras, esse limite desapareceria.

### Por que LoRA

Os autores ajustaram UNet e codificador de texto inteiros com dezenas de
milhares de imagens. Com 59 recortes, isso decoraria as lesões. O LoRA (posto
16, camadas de atenção do UNet, 3,2 M parâmetros) restringe a mudança; VAE e
codificador de texto ficam congelados.

---

## 4. Decisões medidas

Cada parâmetro abaixo foi escolhido por medição, não por padrão de mercado. Os
scripts de medição estão no repositório.

### Contexto da janela — `scripts/analise_escala_vae.py`

Mais contexto dá ao modelo mais tecido para condicionar a geração, mas reduz
mais a lesão. Correlação do detalhe fino após o VAE, nos 59 recortes de treino:

| Política | Mediana | p10 (lesões grandes) | Área mascarada (mediana) |
|---|---|---|---|
| contexto 1,5 | 0,746 | 0,598 | 57% |
| contexto 2,0 | 0,697 | 0,477 | 36% |
| **contexto 2,0 + teto de redução 0,75** | **0,726** | **0,635** | **36%** |

Com o teto, a lesão nunca é reduzida abaixo de 0,75×. Quando o contexto exigiria
mais redução, é o contexto que encolhe (até 1,25×), não a lesão. A lesão típica
mantém contexto 2,0; só as maiores trocam contexto por detalhe.

**Correção (2026-09-23):** o teto só vale enquanto o contexto pode encolher. Com o
contexto já no mínimo (1,25×), uma lesão acima de ~550 px de lado não cabe em
512 px a 0,75× e é reduzida mais. No `lote_birads1_v1`, 32% das lesões nasceram em
pixels nativos (escala 1,0), 58% entre 0,75 e 1,0 e 10% abaixo de 0,75 (mínimo 0,46;
lado mediano 623 px). A escala de cada lesão fica em `generation.janela.scale_native_to_model`.

### Colagem — `scripts/analise_colagem.py`

A primeira versão (alpha retangular, nível por mediana) deixava blocos
retangulares visíveis: o fundo do recorte tem gradiente próprio. O modelo
aprenderia que lesão é um retângulo.

| Colagem | Excesso de borda (mediana / p90) | Contraste preservado |
|---|---|---|
| retângulo + mediana | 1,024 / **1,248** | 1,069 |
| superelipse + plano, transição 0,15 | 0,998 / **1,002** | 1,054 |

(1,0 = a colagem não criou aresta.) A superelipse não tem arestas retas, e o
ajuste por plano corrige nível e gradiente do fundo.

### Textura nativa na margem da lesão — `scripts/analise_remendo.py`

A máscara vai 10% além da lesão, e o tecido dessa margem é redesenhado pelo
modelo e passa pelo VAE. Uma inspeção visual notou a região da lesão mais lisa
que o entorno, e a medição confirmou: ruído 0,82 e textura fina 0,80 do tecido
original, sem degrau de nível nem aresta.

`window.restore_native_margin` mantém, fora do corpo da lesão, só a baixa
frequência do que o modelo mudou (σ 3 px na entrada do modelo), devolvendo a
textura fina do detector. Dentro da lesão, a mudança fica inteira. É composição,
não treino: `native_window.margin_texture_sigma`, e `0` desliga.

| No lote de produção (1,00 = igual ao original) | Antes | Depois |
|---|---|---|
| Ruído na margem (16 / 8 bits) | 0,82 / 0,83 | **0,94 / 0,96** |
| Textura fina na margem | 0,80 / 0,81 | **0,90 / 0,89** |
| Contraste local 3–12 px na margem | 0,93 / 0,89 | 0,90 / 0,86 |
| Degrau na borda da máscara | 0,97 | 0,99 |

Na validação, a lesão não mudou além do ruído da medida (KID 0,90 → 0,94 com
desvio ~0,1; textura W1 0,56 → 0,58; contraste idêntico). O contraste de
estruturas médias (3–12 px) na margem continua ~10% abaixo: são estruturas que o
modelo redesenha, e baixá-las exigiria σ maior (que apaga a lesão) ou margem de
máscara menor (retreino).

### Piso de densidade — lesão não escurece o tecido

Numa inspeção visual (`6278_IM-0005_CCD_R`), o tecido em volta da lesão estava
mais escuro que o original. O modelo aprendeu com os recortes BI-RADS 5, justos
e cercados da gordura escura da própria paciente, a desenhar "massa clara + halo
escuro", e passou a **retirar** densidade do hospedeiro. Fisicamente, uma lesão
soma atenuação.

`window.floor_low_frequency` limita a mudança de baixa frequência (σ 3 px) a no
mínimo −3 níveis de 0–255; a textura fina mantém o sinal, e a gordura que já
existe continua escura porque vem da imagem original.
`native_window.density_floor`; `null` desliga.

| Periferia da lesão escurecida ≥ 8 níveis (mediana) | Antes | Depois |
|---|---|---|
| Lote, 16 bits | 9% | 0% |
| Lote, 8 bits | 15% (15 de 50 lesões com ≥ 25%) | 0% |
| Caso `6278_IM-0005_CCD_R` | 37% (5% dos pixels até −29 níveis) | 0% (−0,2) |

Na validação, a lesão ficou igual dentro do ruído (KID 0,94 → 0,91, textura
0,58 → 0,58, contraste 0,83 → 0,81).

### Posicionamento em três níveis

A regra anatômica do modo canvas exige tecido denso em **todos** os pixels da
caixa. Para lesões reais de 300 px, isso reprovava 35% das posições, porque a
periferia sempre pega gordura (o que é anatomicamente normal). Os níveis ficaram
assim:

| Nível | Exige | Uso (amostras de treino) |
|---|---|---|
| `anatomia` | toda a caixa plausível (regra do canvas) | 62% |
| `anatomia_relaxada` | centro em tecido denso; caixa dentro da mama, fora do peitoral, ≥ 8 mm da pele | 35% |
| `fallback` | só o centro | 4%, **desligado no treino e no pipeline** |

Sem o último recurso, lesões acima de 600 px são aceitas em ~50% das tentativas e
ficam levemente sub-representadas no treino.

### Critério de escolha de checkpoint

A primeira validação escolhia pelo KID sobre features DINOv3. **Esse critério
estava errado.** O modelo base, que não gera lesão nenhuma, teve KID *menor* que
o LoRA (1,40 contra 1,84 no passo 500): tecido fibroso qualquer se parece com
lesão para features genéricas. O critério atual (`lesion_eval.select_best`) é em
cascata, e cada etapa só decide quando a diferença excede o ruído dela:

1. **elegível:** razão de contraste núcleo/periferia sintético/real entre 0,5 e
   1,5 (a lesão existe) e memorização ≤ 0,15;
2. **AUC** de um classificador real × sintético, com empate dentro do IC95 do melhor;
3. **KID**, com empate dentro de um desvio do melhor;
4. **distância de textura** (mediana W1);
5. razão de contraste mais próxima de 1.

As etapas 2 e 3 ganharam empate estatístico depois que o AUC saturou em
0,98–1,00 para todos os candidatos (seção 6).

---

## 5. Passo a passo

### Fase 0 — auditoria e divisão (CPU, segundos)

```bash
.venv/bin/python scripts/prepare_birads_data.py \
    --classe birads5=../_download_drive --classe birads4=../birads_4 \
    --hospedeiros ../sem_anomalia --out treinos/dados_birads
```

Gera `split.json`, `catalogo_tamanhos.json`, `AUDITORIA.md` e folhas de contato
por partição. **Aborta se houver vazamento de paciente.**

### Análises que justificam os parâmetros (opcional)

```bash
.venv/bin/python scripts/analise_escala_vae.py --fatores 1.5,2.0 --min-scale 0.75   # GPU, ~1 min
.venv/bin/python scripts/analise_colagem.py                                         # CPU, ~1 min
```

### Fase 3 — treino (GPU, ~35 min)

```bash
# teste de fumaça primeiro (~3 min)
.venv/bin/python scripts/train_lora_birads.py --run treinos/fumaca --passos 40 \
    --validar-cada 20 --val-sementes 1 --val-limite 4 --val-passos 10

# treino completo
.venv/bin/python scripts/train_lora_birads.py --run treinos/birads_lora_v1

# se cair no meio
.venv/bin/python scripts/train_lora_birads.py --run treinos/birads_lora_v1 --retomar
```

Valida no passo 0 (modelo base) e a cada 500 passos. O script **recusa
sobrescrever** um treino existente.

### Escolha do checkpoint (GPU, ~2 min por checkpoint × guidance)

```bash
.venv/bin/python scripts/select_checkpoint.py --run treinos/birads_lora_v1 --guidance 2.5,4.0
```

Reavalia todos os checkpoints na **validação** com o critério da seção 4 e grava
`melhor_checkpoint` e `melhor_guidance` em `treino.json`.

### Fase 4 — avaliação final no teste (GPU, ~5 min)

```bash
.venv/bin/python scripts/evaluate_birads.py --run treinos/birads_lora_v1 --particao test
```

Compara três braços sobre os **mesmos** recortes, hospedeiros, posições e
sementes: `atual` (canvas), `nativo_base` (janela nativa sem LoRA) e
`nativo_lora`. **Rode no teste uma vez só, no fim.** Tudo que é comparação
durante o desenvolvimento vai em `--particao val`.

### Produção

```bash
./executar_pipeline.sh lote_birads --lora treinos/birads_lora_v1
# ou
.venv/bin/python scripts/build_synthetic_dataset.py --lora treinos/birads_lora_v1 \
    --output-dir ../mamas_com_lesao/lote_birads
```

`--lora` liga o modo nativo com os tipos `birads5:0.83,birads4:0.17`, a proporção
dos 92 recortes reais (76/16). Sem pesos, os tipos se alternariam em 50/50. Os
tamanhos vêm do catálogo real do treino. A geometria e a guidance gravadas no treino prevalecem
sobre o `config.yaml` (cada divergência vai para o log). As garantias do modo
canvas continuam valendo: zero pixels fora da máscara e mesma profundidade de
bits, conferidas por `verify_dataset.py`.

---

## 6. Resultados (`treinos/birads_lora_v1`)

Treino: 4.000 passos, lote 4, LoRA posto 16, ~35 min na RTX 4090, sem nenhuma
amostra em posicionamento de último recurso.

### Escolha do checkpoint (validação)

Todos os checkpoints de 1.500 a 4.000, com guidance 2,5 e 4,0, mais o modelo base
como referência (`treinos/birads_lora_v1/selecao/SELECAO.md`):

- **O modelo base não é elegível:** razão de contraste 0,22–0,24. Ele não gera lesão.
- **O AUC saturou** em 0,98–1,00 para todos, empatado dentro do IC95.
- **O KID empatou** dentro de um desvio para 9 dos 12 candidatos (0,81–0,93).
- **A textura decidiu:** **passo 2.500, guidance 2,5**, W1 0,42, razão de contraste 0,84.
- **Guidance 2,5 foi melhor que 4,0** em KID e textura em todos os checkpoints.

### Avaliação final (teste, uma vez)

15 recortes reais de pacientes nunca vistos (2 BI-RADS 4, 13 BI-RADS 5), 3
hospedeiros nunca vistos, 45 lesões por braço, mesmas posições e sementes em
todos (`treinos/birads_lora_v1/avaliacao_test/AVALIACAO.md`):

| Braço | KID ↓ | AUC real × sint. | Razão de contraste (1 = ideal) | Textura W1 ↓ | Pixels alterados fora da máscara |
|---|---|---|---|---|---|
| `atual` (canvas, antes deste trabalho) | 1,07 ± 0,22 | 0,99 [0,98–0,99] | 0,77 | 0,81 | 0 |
| `nativo_base` (janela nativa, sem LoRA) | 1,72 ± 0,18 | 1,00 [0,99–1,00] | 0,28 | 0,89 | 0 |
| **`nativo_lora`** | **0,64 ± 0,11** | **0,94 [0,89–0,97]** | 0,76 | **0,72** | **0** |
| *referência: as mesmas reais pelo VAE* | 0,30 | 0,97 [0,92–0,99] | 0,97 | 0,39 | — |
| *piso: reais × reais de outros pacientes* | 0,07 | 0,59 | — | — | — |

Memorização 0,00 nos três braços (esperado sem cópia: 0,05). A avaliação foi
rodada duas vezes com as mesmas sementes: os três braços saíram idênticos.

### Como ler

1. **O LoRA é o melhor dos três em todas as medidas de semelhança.** KID 40% menor
   que o pipeline atual e textura mais próxima da real. É o único braço cujo
   intervalo do AUC não encosta em 1,0. Visualmente, gera espículas radiais e
   pontos parecidos com microcalcificações, que o modo canvas não produz.
2. **O AUC do classificador mede o VAE, não a lesão.** A *mesma* lesão real,
   só reconstruída pelo VAE congelado, já é separada com AUC 0,97 (0,98 na
   validação). O LoRA (0,94) está nesse nível: ajustar o UNet não reduz mais o
   AUC. Descer daí exigiria mexer no decodificador do VAE.
3. **Em KID e textura ainda há distância real até o limite do VAE:** 0,64 e 0,72
   contra 0,30 e 0,39. É a margem que mais dados ou um treino melhor podem
   reduzir.
4. **O modelo base confirma a necessidade do ajuste:** sem LoRA, a janela nativa
   não gera lesão (contraste 0,28).

### Análises que não viraram mudança

`scripts/analise_separabilidade.py` testou compor só o sinal de baixa frequência
da lesão, preservando o ruído nativo do detector. O AUC caiu pouco (0,99 → 0,95)
e a textura piorou muito (0,42 → 0,71). A composição ficou como estava.

### Correções de percurso registradas

- A primeira validação escolhia por KID e preferia o modelo sem lesão. Criada a
  guarda de contraste.
- A primeira seleção comparava AUCs saturados pelo valor exato. Criado o empate
  estatístico em cascata (`lesion_eval.select_best`), definido só com a validação.
- A primeira referência do VAE colava o recorte com transição superelíptica e
  trocava os cantos por tecido do hospedeiro, misturando colagem e VAE; o LoRA
  saía *menos* separável que ela. Refeita com o recorte inteiro. Só a linha de
  referência mudou; os braços foram reproduzidos idênticos.
- O primeiro lote de produção saiu com 33% de BI-RADS 4 para um peso de 17%: o
  `crc32` não é uniforme nos 66 nomes reais (KS p=0,003). O sorteio com pesos
  passou a usar SHA-256; o lote refeito saiu 55/11 (83/17). O teste de regressão
  usa os nomes reais, porque nomes sintéticos não reproduziam o defeito.
- `executar_pipeline.sh` pulava a verificação de integridade quando o lote tinha
  falha parcial (código 1). Agora verifica sempre e só interrompe no código 2.

---

## 7. Limitações

1. **Dados pequenos.** 59 recortes de treino, 15 de teste. Os intervalos são
   largos, e BI-RADS 4 tem só 2 recortes no teste: nenhuma conclusão por classe
   é possível.
2. **O prompt de classe não foi validado.** O modelo recebe `birads 4` ou
   `birads 5`, mas com 12 recortes de BI-RADS 4 no treino não há evidência de
   que gere algo distinto por classe.
3. **Só recortes.** A colagem não ensina como a lesão deforma o tecido além da
   caixa.
4. **Uma fonte avaliada.** A semelhança com BI-RADS real foi medida só nas
   mamografias de 16 bits da mesma fonte dos recortes. Nas 51 de 8 bits de
   `sem_anomalia` o pipeline gera com aviso (`fonte_igual_ao_treino: false` nos
   metadados). No lote de produção, contraste, ruído e nitidez medianos saíram
   parecidos nas duas fontes (Weber nativo 1,01 × 0,94), o que descarta falha
   grosseira, mas não prova semelhança com lesões reais nessa fonte.
5. **Lesões fortes demais em hospedeiros escuros de 8 bits.** Nas imagens em que o
   QC reprova, o Weber mediano é 2,08 (limite 1,50). Quais imagens ficam sem
   lesão varia entre execuções, porque posição e semente dependem das tentativas
   gastas nas imagens anteriores.
6. **Estruturas claras pré-existentes dentro da máscara são redesenhadas.** Em
   `6278_IM-0005_CCD_R`, uma calcificação redonda do original virou um anel.
7. **Um recorte de treino contém um clipe cirúrgico**
   (`1400_32156_L_CC_1_box_000.png`), que o modelo pode aprender a gerar. Manter
   ou excluir num retreino é decisão pendente.
8. **Assinatura do VAE.** Um classificador em features profundas separa as
   lesões geradas das reais (AUC 0,94). Servem para aumento de dataset, não como
   substituto indistinguível de casos reais.

---

## 8. O que fica gravado

```
treinos/
├── dados_birads/                  fase 0 (+ análises)
└── birads_lora_v1/
    ├── treino.json                hiperparâmetros, geometria, ambiente, checkpoints, escolhido
    ├── split.json, catalogo_tamanhos.json
    ├── log.jsonl                  perda por passo
    ├── checkpoints/passo_NNNNNN/  pytorch_lora_weights.safetensors (~13 MB), validacao.json, grade.png
    ├── selecao/                   SELECAO.md, grades por checkpoint × guidance
    └── avaliacao_test/            AVALIACAO.md, avaliacao.json, grade.png
```

Cada lesão gerada em produção grava em `metadata/*.json`, no campo `generation`:
modo, prompt, adaptador, guidance, tamanho, janela, caixa e posicionamento.

---

## 9. Testes

```bash
.venv/bin/python -m unittest discover -s tests -t . -v
```

63 testes cobrem as garantias que falham em silêncio:

- **janela:** contém a lesão, fica dentro da imagem e alinha a caixa à grade;
- **writeback:** zero pixels fora da máscara em 8 e 16 bits, com delta aleatório também fora dela;
- **divisão:** nenhum paciente em duas partições, estratificação, determinismo;
- **colagem:** máscara cobre a lesão, nível real, reprodutível;
- **posicionamento:** o nível relaxado nunca toca o peitoral;
- **geração nativa completa**, com modelo falso: zero pixels fora da máscara de ponta a ponta;
- **seleção:** um gerador sem lesão perde mesmo com KID e AUC melhores (o erro real da primeira validação);
- **coerência dos padrões** de geometria entre `config.py`, `config.yaml`, `PasteConfig` e as funções;
- **pesos de tipo** reproduzem a proporção pedida, e sem pesos a alternância original fica igual.

Os testes de janela foram checados por mutação: um defeito de arredondamento da
grade que a margem mascarava passou a ser pego.
