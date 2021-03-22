echo "[INFO] start check the enviroment..."
python3 -c "import te" && echo "[INFO] check te sucess"
python3 -c "import topi" && echo "[INFO] check topi sucess"
#cmake --version  && echo "[INFO] check cmake sucess"
atc --version && echo "[INFO] check atc sucess "

echo "[INFO] start compile the example..."

cd ../samples/atlas300/
mkdir -p build/intermediates/host
cd build/intermediates/host
cmake ../../../src -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE
make  && echo "[INFO] check the env sucess!"
