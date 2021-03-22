MODEL_PATH=$1
DATA_PATH=$2
MOBILE_DIR=$3
OUTPUT_PATH=$4

adb shell "mkdir $MOBILE_DIR"
adb shell "mkdir $MOBILE_DIR/out_dir"
adb push $DATA_PATH $MOBILE_DIR/data
adb push MODEL_PATH $MOBILE_DIR/data
adb shell "/data/local/tmp/model_run_tool   --model=$MOBILE_DIR/kirin990_npu.om   --input=$MOBILE_DIR/input.bin   --output_dir=$MOBILE_DIR/out_dir/  --enable_item=1" >$OUTPUT_PATH/ome.log
adb shell "/data/local/tmp/data_proc_tool   --result_path=$MOBILE_DIR/out_dir"
cd $OUTPUT_PATH
adb shell "ls $MOBILE_DIR/out_dir/*model.csv" |xargs -I  {} adb pull {}  result.csv



