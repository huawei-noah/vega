BACKEND=$1
MODEL=$2
WEIGHT=$3
OM_SAVE_PATH=$4
LOG_PATH=$5
INPUT_SHAPE=$6
OUT_NODES=$7

cd /data/tools/hwhiai-ddk-100.500.010.010/tools/tools_omg/

if [ $BACKEND == "tensorflow" ]; then
    ./omg --model=$MODEL --framework=3 --output=$OM_SAVE_PATH/kirin990_npu  --input_shape=$INPUT_SHAPE  --out_nodes=$OUT_NODES  >$LOG_PATH/omg.log  2>&1
elif [ $BACKEND == "caffe" ]; then
    ./omg --model=$MODEL --weight=$WEIGHT --framework=0 --output=$OM_SAVE_PATH/kirin990_npu  >$LOG_PATH/omg.log  2>&1
else
    echo "[ERROR] omg model convert: The backend must be tensorflow or caffe."
fi

