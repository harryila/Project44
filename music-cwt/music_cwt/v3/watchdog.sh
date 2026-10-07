#!/bin/bash
# watchdog.sh — relance le training v2 si arrete.
#
# A lancer via cron SUR ALBATROS (login node — cron fiable, machine stable).
# Le watchdog vit sur albatros donc survit aux reboots de la machine GPU.
# Toutes les 15 min : verifie que train_cwt.py tourne sur $TARGET, relance sinon.
#
# Setup cron (sur albatros) :
#     (crontab -l 2>/dev/null; echo "*/15 * * * * /bin/bash $HOME/taff/v2/watchdog.sh") | crontab -
#
# Prerequis : cle SSH sans mot de passe entre albatros et $TARGET
#   (home NFS partage → ssh-keygen + cat id_*.pub >> authorized_keys suffit).
#
# Si tu changes de machine GPU : edite TARGET ci-dessous.

set -u

TARGET=rover                              # machine GPU sur laquelle tourne le training
V2_DIR=~/taff/v2
SENTINEL=$V2_DIR/training_done.flag
LOG=$V2_DIR/watchdog.log

# Training termine ? rien a faire.
if [ -f "$SENTINEL" ]; then
    exit 0
fi

# Le training tourne-t-il deja sur la machine cible ?
if ssh -o ConnectTimeout=8 -o BatchMode=yes "$TARGET" \
       'pgrep -f train_cwt.py >/dev/null 2>&1'; then
    exit 0   # OK, ca tourne — rien a faire
fi

# Sinon : relancer le wrapper sur la machine cible (detache, survit a la session SSH).
echo "[$(date)] training DOWN on $TARGET — relance auto_train.sh" >> "$LOG"
ssh -o ConnectTimeout=8 -o BatchMode=yes "$TARGET" \
    "cd $V2_DIR && nohup bash auto_train.sh >/dev/null 2>&1 & disown" \
    >> "$LOG" 2>&1
echo "[$(date)] relance envoyee." >> "$LOG"
