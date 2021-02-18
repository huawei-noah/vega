MODEL_PATH=$1
DATA_PATH=$2
MOBILE_DIR=$3
OUTPUT_PATH=$4

adb shell "mkdir $MOBILE_DIR/data"
adb push $DATA_PATH $MOBILE_DIR/data
adb shell "/data/local/tmp/benchmark -m $MODEL_PATH -i $MOBILE_DIR/data/input.bin" >$OUTPUT_PATH/ome.log
cd $OUTPUT_PATH
#adb shell "ls $MOBILE_DIR/data/*.txt" |xargs -I  {} adb pull {}  result.txt
#cat ome.log |grep dims |awk -F ":" 'NR==2 {print $NF}' >output_dim.txt
adb pull /sdcard/BoltResult.txt  ./