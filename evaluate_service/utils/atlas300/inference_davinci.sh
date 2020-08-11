WORK_DIR=$1
EXAMPLE_DIR=$2

cp $WORK_DIR/*.om  $EXAMPLE_DIR/model/
cp $WORK_DIR/*bin  $EXAMPLE_DIR/data/

cd $EXAMPLE_DIR/
mkdir -p build/intermediates/host
cd build/intermediates/host
cmake ../../../src -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE
make

cd ../../../out

sudo env "LD_LIBRARY_PATH=/usr/local/Ascend/acllib/lib64:/usr/local/Ascend/add-ons:/usr/local/Ascend/driver/lib64/"   ./main >$WORK_DIR/ome.log

