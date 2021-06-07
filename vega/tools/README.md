# command line tools

## fully train

usage:

```text
usage: fully_train.py [-h] [-backend GENERAL.BACKEND]
                      [-devices_per_trainer GENERAL.WORKER.devices_per_trainer]
                      [-master_ip GENERAL.CLUSTER.MASTER_IP]
                      [-slaves [GENERAL.CLUSTER.SLAVES [GENERAL.CLUSTER.SLAVES ...]]]
                      -dataset DATASET.TYPE
                      [-data_path DATASET.COMMON.DATA_PATH]
                      [-batch_size DATASET.COMMON.BATCH_SIZE]
                      [-model_desc MODEL.MODEL_DESC]
                      [-model_file MODEL.PRETRAINED_MODEL_FILE]
                      [-epochs TRAINER.EPOCHS]
                      [-evaluator [EVALUATOR [EVALUATOR ...]]]

Fully train model.

optional arguments:
  -h, --help            show this help message and exit
  -backend GENERAL.BACKEND, --general.backend GENERAL.BACKEND
                        pytorch|tensorflow|mindspore
  -devices_per_trainer GENERAL.WORKER.devices_per_trainer, --general.worker.devices_per_trainer GENERAL.WORKER.devices_per_trainer
  -master_ip GENERAL.CLUSTER.MASTER_IP, --general.cluster.master_ip GENERAL.CLUSTER.MASTER_IP
  -slaves [GENERAL.CLUSTER.SLAVES [GENERAL.CLUSTER.SLAVES ...]], --general.cluster.slaves [GENERAL.CLUSTER.SLAVES [GENERAL.CLUSTER.SLAVES ...]]
                        slave IP list
  -dataset DATASET.TYPE, --dataset.type DATASET.TYPE
                        dataset name.
  -data_path DATASET.COMMON.DATA_PATH, --dataset.common.data_path DATASET.COMMON.DATA_PATH
                        dataset path.
  -batch_size DATASET.COMMON.BATCH_SIZE, --dataset.common.batch_size DATASET.COMMON.BATCH_SIZE
  -model_desc MODEL.MODEL_DESC, --model.model_desc MODEL.MODEL_DESC
  -model_file MODEL.PRETRAINED_MODEL_FILE, --model.pretrained_model_file MODEL.PRETRAINED_MODEL_FILE
  -epochs TRAINER.EPOCHS, --trainer.epochs TRAINER.EPOCHS
  -evaluator [EVALUATOR [EVALUATOR ...]], --evaluator [EVALUATOR [EVALUATOR ...]]
                        evaluator list, eg. -evaluator HostEvaluator DeviceEvaluator
```

example:

```text
python3 -m vega.tools.fully_train -dataset Cifar10 -batch_size 8 -data_path /cache/datasets/cifar10 -model_desc ./tasks/nas/workers/nas1/1/desc_1.json -epochs 1 -evaluator HostEvaluator
```

## benchmark

usage:

```text
usage: benchmark.py [-h] [-backend GENERAL.BACKEND] -dataset DATASET.TYPE
                    [-data_path DATASET.COMMON.DATA_PATH]
                    [-batch_size DATASET.COMMON.BATCH_SIZE]
                    [-model_desc MODEL.MODEL_DESC]
                    [-model_file MODEL.PRETRAINED_MODEL_FILE]
                    [-evaluator [EVALUATOR [EVALUATOR ...]]]

Benchmark.

optional arguments:
  -h, --help            show this help message and exit
  -backend GENERAL.BACKEND, --general.backend GENERAL.BACKEND
                        pytorch|tensorflow|mindspore
  -dataset DATASET.TYPE, --dataset.type DATASET.TYPE
                        dataset name.
  -data_path DATASET.COMMON.DATA_PATH, --dataset.common.data_path DATASET.COMMON.DATA_PATH
                        dataset path.
  -batch_size DATASET.COMMON.BATCH_SIZE, --dataset.common.batch_size DATASET.COMMON.BATCH_SIZE
  -model_desc MODEL.MODEL_DESC, --model.model_desc MODEL.MODEL_DESC
  -model_file MODEL.PRETRAINED_MODEL_FILE, --model.pretrained_model_file MODEL.PRETRAINED_MODEL_FILE
  -evaluator [EVALUATOR [EVALUATOR ...]], --evaluator [EVALUATOR [EVALUATOR ...]]
                        evaluator list, eg. -evaluator HostEvaluator  DeviceEvaluator
```

example:

```bash
python3 -m vega.tools.benchmark -dataset Cifar10 -batch_size 8 -data_path /cache/datasets/cifar10 -model_desc ./tasks/fullytrain/output/fully_train/desc_0.json -model_file=./tasks/fullytrain/output/fully_train/model_0.pth -evaluator HostEvaluator
```
