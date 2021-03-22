BACKEND=$1
MOBILE_WORK_DIR=$2
MODEL=$3
WEIGHT=$4
MODEL_NAME=$5
PRECISON=$6
LOG_PATH=$7


adb shell "mkdir $MOBILE_WORK_DIR"
adb push $MODEL $MOBILE_WORK_DIR

if [ $BACKEND == "tensorflow" ]; then
    adb shell "./data/evaluate_service/tools/tflite2bolt $MOBILE_WORK_DIR $MODEL_NAME $PRECISON"  >$LOG_PATH/omg.log
elif [ $BACKEND == "caffe" ]; then
    adb push $WEIGHT $MOBILE_WORK_DI
    adb shell "./data/evaluate_service/tools/caffe2bolt $MOBILE_WORK_DIR $MODEL_NAME $PRECISON"   >$LOG_PATH/omg.log
elif [ $BACKEND == "onnx" ]; then
    adb shell "/data/local/tmp/X2bolt  -d $MOBILE_WORK_DIR/ -m $MODEL_NAME -i $PRECISON"  >$LOG_PATH/omg.log
else
    echo "[ERROR] Bolt model convert: The backend must be tensorflow, caffe or onnx."
fi

