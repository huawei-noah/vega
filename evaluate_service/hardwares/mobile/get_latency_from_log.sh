LOG_FILE=$1

cat $LOG_FILE | grep run |awk -F ' '  '{print $2}'
