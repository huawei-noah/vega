DAVINCI_ENV_TYPE=$1
BACKEND=$2
MODEL=$3
WEIGHT=$4
OM_SAVE_PATH=$5
LOG_SAVE_PATH=$6
INPUT_SHAPE=$7
PRECISION=$8

if [ $DAVINCI_ENV_TYPE == "ATLAS200DK" ]; then
    if [ $BACKEND == "tensorflow" ]; then
        omg --model=$MODEL  --framework=3   --output=$OM_SAVE_PATH/davinci_model  >$LOG_SAVE_PATH/omg.log  2>&1
    elif [ $BACKEND == "caffe" ]; then
        omg --model=$MODEL --weight=$WEIGHT --framework=0  --output=$OM_SAVE_PATH/davinci_model  >$LOG_SAVE_PATH/omg.log 2>&1
    else
        echo "[ERROR] Davinci model convert: The backend must be tensorflow, caffe."
    fi
else
    if [ $BACKEND == "tensorflow" ]; then
        atc --model=$MODEL  --framework=3  --input_format='NCHW'  --disable_reuse_memory=1  --input_shape=$INPUT_SHAPE  --output=$OM_SAVE_PATH/davinci_model --soc_version=Ascend310 --core_type=AiCore  --output_type=$PRECISION >$LOG_SAVE_PATH/omg.log 2>&1
    elif [ $BACKEND == "caffe" ]; then
        atc --model=$MODEL --weight=$WEIGHT --framework=0  --input_format='NCHW' --disable_reuse_memory=1  --output=$OM_SAVE_PATH/davinci_model --soc_version=Ascend310 --core_type=AiCore  >$LOG_SAVE_PATH/omg.log  2>&1
    elif [ $BACKEND == "mindspore" ]; then
        atc --model=$MODEL  --framework=1  --disable_reuse_memory=1  --output=$OM_SAVE_PATH/davinci_model --soc_version=Ascend310 --core_type=AiCore  --output_type=$PRECISION >$LOG_SAVE_PATH/omg.log  2>&1
    elif [ $BACKEND == "onnx" ]; then
        atc --model=$MODEL  --framework=5  --output=$OM_SAVE_PATH/davinci_model --soc_version=Ascend310 --core_type=AiCore  --output_type=$PRECISION  >$LOG_SAVE_PATH/omg.log  2>&1
    else
        echo "[ERROR] Davinci model convert: The backend must be tensorflow, caffe, mindspore or onnx."
    fi
fi