INPUT_DATA=$1
OM_MODEL=$2
EXECUTE_FILE_PATH=$3
LOG_SAVE_PATH=$4
MUTI_INPUT=$5


if [ $MUTI_INPUT == "True" ]; then
  cp $OM_MODEL  $LOG_SAVE_PATH/
  script_dir=$(cd $(dirname $0);pwd)
  chmod +x  $script_dir/msame
  $script_dir/msame --model $LOG_SAVE_PATH/davinci_model.om --output $LOG_SAVE_PATH --outfmt TXT  >$LOG_SAVE_PATH/ome.log

else
  cp $OM_MODEL  $LOG_SAVE_PATH/
  cp $EXECUTE_FILE_PATH/main  $LOG_SAVE_PATH/
  cp $EXECUTE_FILE_PATH/acl.json  $LOG_SAVE_PATH/
  cd $LOG_SAVE_PATH/

  ./main >$LOG_SAVE_PATH/ome.log
fi