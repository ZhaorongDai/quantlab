#!/usr/bin/env bash
# The daily data update (scripts/data_update/update.py, config/data_update.yaml) from cron.
#
# cron calls this every hour (Debian cron has no CRON_TZ); it checks the time in
# New York itself, so daylight-saving changes need no edit:
#
#   0 * * * * $HOME/projects/quantlab2/scripts/data_update/update_daily.sh
#
# At 06 ET on a weekday it runs the update, which retries the vendor download every
# 15 minutes until 08:30 ET while the day's bar is not published, and writes
# <data-dir>/update_status.json; the paper trading (quantlab-ibkr live_daily.sh) waits
# for "done" on its day. `update_daily.sh now` runs it at once.
#
# Environment (files readable by the owner only):
#   ~/.config/quantlab/sharadar.env  SHARADAR_API_KEY
# Overrides: DATA_DIR, QUANTLAB_DIR, CPUS, UPDATE_FLAGS (e.g. "--dry-run").
set -u

DATA_DIR=${DATA_DIR:-/data/quantlab}
QUANTLAB_DIR=${QUANTLAB_DIR:-$HOME/projects/quantlab2}
CPUS=${CPUS:-64-114}
PY=$QUANTLAB_DIR/.venv/bin/python

ny() { TZ=America/New_York date "$@"; }
log_dir=$DATA_DIR/logs/data_update
mkdir -p "$log_dir"
log_file=$log_dir/$(ny +%F).log

run_update() {
    exec 9> "$log_dir/.lock"
    flock -n 9 || { echo "$(ny '+%F %T %Z') already running" >> "$log_file"; return 0; }
    [ -r "$HOME/.config/quantlab/sharadar.env" ] && { set -a; . "$HOME/.config/quantlab/sharadar.env"; set +a; }
    echo "$(ny '+%F %T %Z') data update: start ${UPDATE_FLAGS:-}" >> "$log_file"
    # shellcheck disable=SC2086
    (cd "$QUANTLAB_DIR" && QUANTLAB_DATA_DIR=$DATA_DIR OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        MKL_NUM_THREADS=1 taskset -c "$CPUS" "$PY" scripts/data_update/update.py \
        --data-dir "$DATA_DIR" --download-dir "$DATA_DIR/downloads" ${UPDATE_FLAGS:-}) >> "$log_file" 2>&1
    local status=$?
    echo "$(ny '+%F %T %Z') data update: exit $status" >> "$log_file"
    return "$status"
}

case "${1:-cron}" in
    now) run_update ;;
    cron)
        [ "$(ny +%u)" -le 5 ] || exit 0
        [ "$(ny +%H)" = 06 ] && run_update
        ;;
    *) echo "usage: $0 [cron|now]" >&2; exit 2 ;;
esac
