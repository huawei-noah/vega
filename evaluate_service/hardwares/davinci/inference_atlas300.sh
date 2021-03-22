INPUT_DATA=$1
OM_MODEL=$2
EXECUTE_FILE_PATH=$3
LOG_SAVE_PATH=$4

cp $OM_MODEL  $LOG_SAVE_PATH/
# cp $INPUT_DATA  $LOG_SAVE_PATH/
cp $EXECUTE_FILE_PATH/main  $LOG_SAVE_PATH/
cp $EXECUTE_FILE_PATH/acl.json  $LOG_SAVE_PATH/
cd $LOG_SAVE_PATH/

#sudo env "LD_LIBRARY_PATH=/usr/local/Ascend/acllib/lib64:/usr/local/Ascend/add-ons:/usr/local/Ascend/driver/lib64/"   ./main >$WORK_DIR/ome.log
./main >$LOG_SAVE_PATH/ome.log
