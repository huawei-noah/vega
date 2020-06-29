#!/usr/bin/env bash

CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
PARENT_DIR="$(dirname "$CURRENT_DIR")"
THIS_FILE=`basename "$0"`

echo "in $THIS_FILE"


PYTHON=${PYTHON:-"python3"}

echo which python=$PYTHON

CONFIG=$1
GPUS=$2

#--master_addr=127.0.0.1 \
#--master_port=6666 \

$PYTHON -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    $(dirname "$0")/train.py $CONFIG ${@:3}

