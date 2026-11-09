# Geração do dataset sintético — `build_synthetic_dataset.py`

Pipeline de produção que lê uma pasta de mamografias saudáveis e devolve um
conjunto novo, com **uma lesão sintética por imagem em posição anatomicamente
plausível**, em duas resoluções, com métricas de qualidade, log rastreável e
relatório técnico.

Complementa — não substitui — o [`README.md`](../README.md), que é o guia de
execução, e o [`METODOLOGIA.md`](METODOLOGIA.md), que descreve a reprodução do
artigo MAM-E. Todos os comandos partem de `/home/joao/MAE/pipeline`.

---

## 1. Execução

```bash
# o ambiente já está em pipeline/.venv — veja README.md §1 para recriá-lo

# lote padrão: uma lesão por imagem, tipos alternados
cd /home/joao/MAE/pipeline
.venv/bin/python scripts/build_synthetic_dataset.py \
    --input-dir ../sem_anomalia \
    --output-dir ../mamas_com_lesao/meu_lote

# ou, gerando e verificando de uma vez:
./executar_pipeline.sh meu_lote
```

Os pesos (`Likalto4/mammo_lesion-inpainting`, ~4 GB) são baixados do Hugging Face
na primeira execução e reaproveitados do cache depois.

### Opções

| Opção | Padrão | Efeito |
|---|---|---|
| `--per-image N` | 1 | lesões por imagem; com N>1 o sufixo vira `_com_lesao_01`, `_02`… |
| `--types` | `mass,nodule,density` | tipos usados, alternados entre as imagens |
| `--seed` | 1337 | semente do lote; fixa reproduz a saída inteira |
| `--limit N` | — | processa só as N primeiras (teste rápido) |
| `--dry-run` | — | valida entrada e gera os mapas anatômicos **sem** rodar o modelo |
| `--no-qc` | — | desliga o portão de qualidade (grava a melhor tentativa) |
| `--placement-attempts N` | 4 | posições diferentes a tentar antes de desistir de uma lesão |
| `--no-canvas` / `--no-overlay` / `--no-anatomy-map` | — | não gravar essas saídas |
| `--set CHAVE=VALOR` | — | sobrescreve qualquer campo do `config.yaml` (repetível) |
| `--verbose` | — | log DEBUG também no console |

Códigos de saída: **0** tudo processado · **1** concluído com falhas parciais ·
**2** nada produzido (erro de ambiente ou configuração).

---

## 2. Estrutura de saída

```
mamas_com_lesao/meu_lote/
├── original/                  PNG na resolução e profundidade de bits originais
│   └── 10180_53352_L_CC_com_lesao.png
├── 1024/                      mesmo resultado no canvas quadrado de 1024
├── masks/
│   ├── original/              máscara da lesão na resolução original
│   └── 1024/                  máscara no canvas
├── overlays/                  painel original | sintética | diferença, com zoom
├── anatomy/                   mapa de plausibilidade (auditoria visual)
├── metadata/                  um JSON por imagem de entrada, inclusive as que falharam
├── manifest.csv               uma linha por lesão
├── processing.log             log legível de toda a execução
├── processing.jsonl           mesmo log, um evento JSON por linha
├── summary.json               contagens finais
└── RELATORIO_TECNICO.md       relatório de validação
```

### O que vai em cada metadado

`metadata/<imagem>.json` traz, por lesão: tipo, semente, número de tentativas,
região anatômica descrita, caixa **nos dois sistemas de coordenadas** (canvas e
resolução original), diâmetro em mm, indicadores anatômicos (distância à pele,
posto de densidade, sobreposição com o peitoral, distância ao mamilo), todas as
métricas de qualidade, veredito do QC e confiança. Também grava os
hiperparâmetros de difusão, composição e anatomia usados — o suficiente para
reexecutar.

---

## 3. Como a posição da lesão é escolhida

[`mamelesion/anatomy.py`](../mamelesion/anatomy.py) constrói um **mapa de plausibilidade** em [0, 1], produto
de quatro fatores, e sorteia o centro da lesão proporcionalmente a esse mapa:

| Fator | Regra |
|---|---|
| `f_pele` | zero a menos de 8 mm da linha de pele; cresce até 20 mm |
| `f_peitoral` | zero dentro do músculo peitoral + 6 mm de margem |
| `f_densidade` | posto percentual da intensidade local dentro da mama; gordura (abaixo do percentil 45) é descartada |
| `f_glandular` | fração de tecido denso numa vizinhança de 12 mm — separa parênquima de densidade isolada |

A caixa só é aceita se **todos** os seus pixels tiverem score acima do mínimo,
não apenas o centro. Isso é o que impede uma lesão de atravessar a borda do
peitoral ou vazar pela linha da pele.

A borda do peitoral é encontrada por busca da reta de maior gradiente entre a
borda superior da mama e a lateral da parede torácica (método `borda_reta`), com
dois planos de contingência (`componente`, `geometrico`). O método usado fica
registrado por imagem — a confiança na exclusão depende de qual deles decidiu.

As figuras em `anatomy/` mostram, para cada imagem, a região válida em verde e o
peitoral excluído em vermelho. **Vale conferi-las antes de confiar no lote.**

---

## 4. Como a resolução original é preservada

O modelo opera em 512×512 e o canvas de trabalho é 1024×1024, mas a saída sai no
tamanho original de cada arquivo. A volta é feita somando o *delta* da lesão à
imagem original em sua profundidade nativa:

```
saída = original + upsample(delta_da_lesão)
```

Duas consequências:

- **Fora da região marcada, nenhum pixel muda** — nem um bit, inclusive nos PNGs
  de 16 bits. Verificável: compare a saída com a entrada usando a máscara.
- **A textura nativa do detector é preservada dentro da lesão**, porque a imagem
  original entra inteira na soma. A lesão aparece como uma variação de densidade
  somada ao tecido real, que é o modelo físico correto para uma massa.

O reamostrador do delta é bilinear, não Lanczos: os lóbulos negativos do Lanczos
produzem *ringing* na fronteira e espalham alterações de alguns níveis de cinza
por milhares de pixels fora da região marcada.

### Enquadramento

`build_breast_canvas` recorta pela caixa envolvente da mama antes de
redimensionar. O `center_crop` do artigo descartaria metade da mama nestas
imagens (razão de aspecto até 1:2,8) e o `pad` desperdiçaria o canvas com fundo
preto.

---

## 4.1 Acompanhar o pipeline passo a passo

Para ver o que acontece entre a mamografia saudável e o arquivo final numa
imagem só:

```bash
.venv/bin/python scripts/inspect_lesion.py ../sem_anomalia/6278_IM-0005_CCD_R.png --type nodule --seed 7
```

Grava 17 PNGs numerados em `outputs/inspecao/<imagem>/`, um por estágio, e
imprime o número que importa em cada ponto:

| # | Arquivo | O que mostra |
|---|---|---|
| 03 | `canvas` | recorte pela mama, 1024×1024 |
| 04 | `mascara_mama` | segmentação do tecido |
| 05 | `distancia_pele` | distância à linha de pele (amarelo = longe) |
| 06 | `peitoral` | músculo detectado, em vermelho |
| 07 | `posto_densidade` | amarelo = fibroglandular, roxo = gordura |
| 08 | `score_plausibilidade` | produto dos quatro fatores — onde a lesão **pode** cair |
| 09 | `caixa` | posição sorteada |
| 11 | `entrada_modelo` | 512×512 com a região apagada — **o que o modelo vê** |
| 12 | `saida_modelo` | saída crua da difusão |
| 13 | `diferenca` | delta bruto: vermelho = densidade adicionada, azul = removida |
| 14 | `alpha` | transição suave da borda |
| 15 | `composto_canvas` | resultado em 1024 |
| 16 | `delta_lesao` | o que de fato vai para a resolução original |
| 17 | `nativo_antes_depois` | recorte antes/depois na resolução original |

O estágio **13** é o mais útil para diagnóstico: se ele sair predominantemente
azul, o modelo removeu densidade em vez de adicionar — a falha que o critério
`roi_delta_ratio` passou a barrar.

Opções: `--box x1,y1,x2,y2` fixa a posição em vez de sortear, `--seed` troca a
geração, `--etapas` só imprime a tabela sem gravar imagens.

As funções chamadas são as mesmas de `pipeline.py`, na mesma ordem — o script
não reimplementa nada, então o que você vê aqui é o que roda no lote.

## 5. Controle de qualidade

Cada geração é medida por [`mamelesion/quality.py`](../mamelesion/quality.py) e passa por um portão:

- **Critérios obrigatórios**: contraste de Weber positivo, CNR na faixa, zero
  pixels alterados fora da ROI, e **saldo líquido de densidade** na ROI acima de
  1% (`roi_delta_ratio`). Falhou, regenera com outra semente (até
  `quality.qc_max_attempts`); esgotado, tenta outra posição (até
  `--placement-attempts`); esgotado, a lesão é descartada e registrada.
- **Critérios informativos**: textura GLCM, ruído, nitidez, emenda, forma.

A **confiança** de cada lesão é `0,55 × fração de critérios aprovados +
0,45 × score anatômico médio`, reduzida em 40% se o QC reprovou. É um número
para *ordenar* gerações e priorizar revisão manual, não uma probabilidade
calibrada.

---

## 6. Rastreabilidade

`processing.log` registra início e fim de cada imagem, avisos, coordenadas,
métricas e status. `processing.jsonl` traz os mesmos eventos como objetos JSON,
um por linha, com os campos estruturados — para filtrar com `jq` ou carregar em
pandas:

```bash
jq -r 'select(.qc_ok==false) | "\(.arquivo) \(.tipo) \(.qc_falhas)"' processing.jsonl
```

Falha em uma imagem **não interrompe o lote**: o erro entra no log e nos
metadados daquela imagem, e o processamento segue.

---

## 7. Limitações

1. **Escala em mm estimada.** Sem `PixelSpacing` (o caso dos PNGs), `mm/px` vem
   da hipótese de um detector de 24×30 cm. Diâmetros e distâncias em mm são
   aproximados na mesma proporção.
2. **A lesão nasce em 512.** A textura nativa é preservada, mas a
   microestrutura da própria lesão nasce na escala do modelo.
   Microcalcificações não são geradas.
3. **Nomenclatura geométrica, não BI-RADS.** As localizações são descritas como
   terço posterior/médio/anterior e faixa superior/média/inferior. Mapear para
   quadrantes exigiria lateralidade e orientação de aquisição, que os PNGs não
   trazem.
4. **O gerador deixa assinatura.** Serve para aumento de dataset, não como
   padrão-ouro em estudos de detecção de imagem sintética.
5. **Detecção do peitoral é heurística**, não um modelo treinado.
6. **O tipo de lesão não é semântico.** O checkpoint foi ajustado com um único
   prompt (`a mammogram with a lesion`) para todas as imagens de treino, então o
   condicionamento textual não carrega o tipo. Os presets `mass`/`nodule`/
   `density` mudam apenas diâmetro, razão de aspecto, guidance e ganho — não
   peça ao dataset que distinga tipos que o modelo nunca aprendeu.

---

> Imagens sintéticas destinadas a pesquisa e aumento de dataset. Não devem ser
> usadas para diagnóstico nem misturadas a arquivos clínicos reais sem marcação
> explícita de origem sintética.
