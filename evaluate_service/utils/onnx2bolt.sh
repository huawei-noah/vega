MOBILE_WORK_DIR=$1
ONNX_MODEL=$2
MODEL_NAME=$3
PRECISON=$4
LOG_PATH=$5

adb shell "mkdir $MOBILE_WORK_DIR"
adb push $ONNX_MODEL $MOBILE_WORK_DIR
adb shell "/data/local/tmp/X2bolt  -d $MOBILE_WORK_DIR/ -m $MODEL_NAME -i $PRECISON"  >$LOG_PATH/omg.log
