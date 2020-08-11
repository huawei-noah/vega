# inference for Atlas 200 DK
WORK_DIR=$1
EXAMPLE_DIR=$2
DDK_USER_NAME=$3
DDK_HOST_IP=$4
ATLAS_HOST_IP=$5
APP_DIR=$6

CURRENT_DIR=$(pwd)
#source env.sh

# copy the example project to work dir
mkdir $WORK_DIR/build_files/
cp -rf $EXAMPLE_DIR/*   $WORK_DIR/build_files/

mkdir -p $WORK_DIR/build_files/run/out/test_data/model/
mkdir -p $WORK_DIR/build_files/run/out/test_data/data/
cp $WORK_DIR/*.om  $WORK_DIR/build_files/run/out/test_data/model/
cp $WORK_DIR/*.bin  $WORK_DIR/build_files/run/out/test_data/data/


# build the file 
cd  $WORK_DIR/build_files/
mkdir -p build/intermediates/device
mkdir -p build/intermediates/host

cd build/intermediates/device
cmake ../../../src -Dtype=device -Dtarget=RC -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++
make install
echo "[INFO] build the device sucess"
cd ../host
cmake ../../../src -Dtype=host -Dtarget=RC -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++
make install 
echo "[INFO] build the host sucess"

cd $CURRENT_DIR

# execute in Atlas 200 DK
#scp /home/ly/evaluate_test/atlas_execute.sh  HwHiAiUser@$ATLAS_HOST_IP:~/
#echo "[INFO] copy the atlas_execute.sh to Atlas 200 DK."
ssh -o "StrictHostKeyChecking no"   HwHiAiUser@$ATLAS_HOST_IP "bash -s" <  ./utils/atlas200_dk/atlas_execute.sh  $WORK_DIR  $DDK_USER_NAME  $DDK_HOST_IP  $APP_DIR
echo "[INFO] execute in Atlas 200 DK finish."

