MOBILE_WORK_DIR=$1
TFLITE_MODEL=$2
MODEL_NAME=$3
PRECISON=$4
LOG_PATH=$5

adb shell "mkdir $MOBILE_WORK_DIR"
adb push $TFLITE_MODEL $MOBILE_WORK_DIR
adb shell "./data/evaluate_service/tools/tflite2bolt $MOBILE_WORK_DIR $MODEL_NAME $PRECISON"  >$LOG_PATH/omg.log
