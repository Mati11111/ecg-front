#!/bin/bash
COMMIT_MSG="$1"

echo "Commit recibido: $COMMIT_MSG"


echo "$COMMIT_MSG" > last_commit.txt


if [ "$COMMIT_MSG" = "BACKEND MODIFIED --> URL CHANGED" ]; then
    echo "BACKEND HAS CHANGED"
    git pull
else
    echo "BACKEND CHANGED RESTARTING FRONT"
fi

pm2 restart 5
echo "FRONT RELOADED"
