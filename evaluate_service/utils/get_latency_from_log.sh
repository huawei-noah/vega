LOG_FILE=$1
BACKEND=$2

if [ $BACKEND == 'Davinci' ]; then
    cat $LOG_FILE |grep costTime | awk -F  ' '  'NR==50 {print $NF}'
elif [ $BACKEND == 'Bolt' ]; then
    cat $LOG_FILE | grep run |awk -F ' '  '{print $2}'
else 
    echo "[ERROR]: the backend must be Davinci or Bolt."
fi
