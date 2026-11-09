#!/usr/bin/env bash
# =============================================================================
# Pipeline completo: mamografias saudaveis -> mamografias com lesao sintetica
# -----------------------------------------------------------------------------
# Roda as tres etapas na ordem correta e para no primeiro erro:
#
#   1. build_synthetic_dataset.py  gera o lote (entrada -> saida)
#   2. verify_dataset.py           confere a integridade do que foi gerado
#   3. resumo                      imprime onde ficou cada coisa
#
# USO
#   ./executar_pipeline.sh                      # lote datado, tipos alternados
#   ./executar_pipeline.sh lote_massas --types mass
#   ./executar_pipeline.sh lote_teste --limit 5 --verbose
#   ./executar_pipeline.sh lote_denso --types density --per-image 2 --seed 42
#
# O primeiro argumento e o NOME DO LOTE (vira uma pasta em mamas_com_lesao/).
# Todos os argumentos seguintes sao repassados a build_synthetic_dataset.py --
# veja `./executar_pipeline.sh --ajuda` ou docs/PROGRAMAS.md para a lista.
# =============================================================================
set -euo pipefail

AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../pipeline
RAIZ="$(dirname "$AQUI")"                               # .../MAE
PY="$AQUI/.venv/bin/python"
ENTRADA="$RAIZ/sem_anomalia"
SAIDA_BASE="$RAIZ/mamas_com_lesao"

if [ "${1:-}" = "--ajuda" ] || [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    # imprime o cabecalho de comentarios ate a primeira linha de codigo
    awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"
    echo
    echo "Parametros repassados ao gerador:"
    "$PY" "$AQUI/scripts/build_synthetic_dataset.py" --help | sed -n '/^options:/,$p'
    exit 0
fi

# O primeiro argumento so e o nome do lote se nao comecar com "-"; assim
# `./executar_pipeline.sh --limit 5` usa o nome datado e repassa a opcao.
if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then
    LOTE="$1"
    shift
else
    LOTE="lote_$(date +%Y%m%d_%H%M)"
fi

# --- checagens de ambiente antes de gastar GPU -------------------------------
[ -x "$PY" ] || { echo "erro: venv nao encontrado em $AQUI/.venv" >&2
                  echo "      crie com: python3 -m virtualenv .venv && \\" >&2
                  echo "      .venv/bin/pip install -r requirements.txt" >&2; exit 2; }
[ -d "$ENTRADA" ] || { echo "erro: pasta de entrada nao existe: $ENTRADA" >&2; exit 2; }
N_ENTRADA=$(find "$ENTRADA" -maxdepth 1 -type f \
            \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \
               -o -iname '*.tif' -o -iname '*.tiff' -o -iname '*.dcm' \) | wc -l)
[ "$N_ENTRADA" -gt 0 ] || { echo "erro: nenhuma imagem em $ENTRADA" >&2; exit 2; }

SAIDA="$SAIDA_BASE/$LOTE"
if [ -e "$SAIDA" ]; then
    echo "erro: '$SAIDA' ja existe. Escolha outro nome de lote ou apague a pasta." >&2
    exit 2
fi

echo "=============================================================="
echo " Pipeline de lesoes sinteticas"
echo "   entrada : $ENTRADA ($N_ENTRADA imagem(ns))"
echo "   saida   : $SAIDA"
echo "   extras  : ${*:-nenhum}"
echo "=============================================================="

# --- 1. geracao ---------------------------------------------------------------
echo
echo ">>> [1/3] Gerando o lote (modelo de difusao, ~3-5 s por imagem)..."
# codigo 1 = concluido com falhas parciais: o que foi gerado ainda precisa ser
# verificado. So o codigo 2 (nada produzido / configuracao invalida) interrompe.
GERACAO=0
"$PY" "$AQUI/scripts/build_synthetic_dataset.py" \
      --input-dir "$ENTRADA" --output-dir "$SAIDA" "$@" || GERACAO=$?
if [ "$GERACAO" -ge 2 ]; then
    echo "erro: geracao falhou (codigo $GERACAO); nada a verificar" >&2
    exit "$GERACAO"
fi
[ "$GERACAO" -eq 1 ] && echo "aviso: lote com falhas parciais -- veja $SAIDA/processing.log"

# --- 2. verificacao independente ---------------------------------------------
echo
echo ">>> [2/3] Verificando a integridade do lote..."
"$PY" "$AQUI/scripts/verify_dataset.py" "$SAIDA" --input-dir "$ENTRADA" --quiet

# --- 3. resumo ----------------------------------------------------------------
echo
echo ">>> [3/3] Pronto$( [ "$GERACAO" -eq 1 ] && echo ' (com falhas parciais)')."
echo "   imagens finais   : $SAIDA/original/     (resolucao e bits originais)"
echo "   versao 1024      : $SAIDA/1024/"
echo "   mascaras         : $SAIDA/masks/original/"
echo "   comparacoes      : $SAIDA/overlays/"
echo "   relatorio        : $SAIDA/RELATORIO_TECNICO.md"
echo "   manifesto        : $SAIDA/manifest.csv"
echo "   log              : $SAIDA/processing.log"
exit "$GERACAO"
