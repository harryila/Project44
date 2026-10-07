#!/bin/bash
# auto_train.sh — Wrapper auto-restart du pre-training v2.
# - Boucle sur le training Python (qui auto-resume depuis le dernier checkpoint)
# - Survit aux crashs (OOM, kill, etc.) — relance apres 60s
# - S'arrete tout seul quand "Entrainement termine" apparait dans le log
# - Logs separes : auto_train.log (meta) et train_v2.log (training)
#
# Usage manuel :
#     bash ~/taff/v2/auto_train.sh
# Usage cron (au reboot) :
#     @reboot /bin/bash $HOME/taff/v2/auto_train.sh
# Usage .bashrc (auto-restart au prochain SSH) : voir doc principale.

set -u

V2_DIR=~/taff/v2
SENTINEL=$V2_DIR/training_done.flag
META_LOG=$V2_DIR/auto_train.log
TRAIN_LOG=$V2_DIR/train_v2.log
LOCK_FILE=$V2_DIR/auto_train.lock

cd "$V2_DIR" || { echo "ERR: cd $V2_DIR failed"; exit 1; }

# Verrou pour eviter deux instances concurrentes (un cron + un manuel par exemple)
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "[$(date)] Another instance is already running. Exiting." >> "$META_LOG"
    exit 0
fi

# Si le training est deja fini, on ne fait rien
if [ -f "$SENTINEL" ]; then
    echo "[$(date)] Sentinel found, training already done. Exiting." >> "$META_LOG"
    exit 0
fi

echo "[$(date)] === auto_train.sh START (pid=$$) ===" >> "$META_LOG"

while [ ! -f "$SENTINEL" ]; do
    echo "[$(date)] Launching python train_cwt.py ..." >> "$META_LOG"
    python3 -u train_cwt.py \
        --kern-dir ../kern_all \
        --checkpoint-dir checkpoints \
        --epochs 50 \
        >> "$TRAIN_LOG" 2>&1
    EXIT=$?
    echo "[$(date)] Python exited with code $EXIT" >> "$META_LOG"

    # Critere de fin propre : message "Entrainement termine" dans le log
    if tail -50 "$TRAIN_LOG" | grep -q "Entraînement terminé"; then
        touch "$SENTINEL"
        echo "[$(date)] === Training DONE, sentinel created. ===" >> "$META_LOG"
        break
    fi

    # Sinon crash/OOM/kill : on attend et on retry
    echo "[$(date)] Crash detected. Restart in 60s..." >> "$META_LOG"
    sleep 60
done

echo "[$(date)] === auto_train.sh END ===" >> "$META_LOG"
