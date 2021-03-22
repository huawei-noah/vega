LOG_FILE=$1
cat $LOG_FILE |grep costTime | awk -F  ' '  '{print $NF}'
