WORK_DIR=$1
DDK_USER_NAME=$2
DDK_HOST_IP=$3
APP_DIR=$4

cd ~
mkdir -p $APP_DIR
cd ~/$APP_DIR
scp -r  $DDK_USER_NAME@$DDK_HOST_IP:$WORK_DIR/build_files/run/out/*   ./
echo "[INFO] copy the fils to Atlas 200 Dk sucess."
./main  >ome.log
echo "[INFO] run exe in Atlas 200 Dk sucess."
scp ome.log $DDK_USER_NAME@$DDK_HOST_IP:$WORK_DIR/
scp ./result_files/result_file  $DDK_USER_NAME@$DDK_HOST_IP:$WORK_DIR/
echo "[INFO] copy the result log to DDK host sucess."
cd ../
rm -rf ./$APP_DIR 
echo "[INFO] delete the temp files in Atlas 200 DK sucess."
