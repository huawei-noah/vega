# Callbacks

## Callback List

| callbacks | priority | 1<br>init_trainer | 2<br>before_train | 3<br>before_epoch | 4<br>before_train_step | 5<br>make_batch | 6<br>train_step | 7<br>valid_step | 8<br>model_fn | 9<br>train_input_fn | 10<br>valid_input_fn | 11<br>after_train_step | 12<br>after_epoch | 13<br>after_train | 14<br>before_valid | 15<br>before_valid_step | 16<br>after_valid_step | 17<br>after_valid |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| ModelBuilder              | 200   | √ |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
| RuntimeCallback           | 210   |   | √ | √ |   |   |   |   |   |   |   | √ | √ | √ |   |   |   |   |
| ModelStatistics           | 220   |   | √ |   |   |   |   |   |   |   |   |   | √ | √ |   |   |   |   |
| MetricsEvaluator          | 230   |   | √ | √ | √ |   |   |   |   |   |   | √ | √ | √ |   | √ | √ | √ |
| ModelCheckpoint           | 240   |   | √ |   |   |   |   |   |   |   |   |   | √ |   |   |   |   |   |
| PerformanceSaver          | 250   |   | √ |   |   |   |   |   |   |   |   |   | √ | √ |   |   |   |   |
| DdpTorch                  | 260   |   | √ |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
| ProgressLogger            | 270   |   | √ | √ | √ |   |   |   |   |   |   | √ |   | √ |   |   | √ | √ |
| ReportCallback            | 280   |   | √ |   |   |   |   |   |   |   |   |   | √ | √ |   |   |   | √ |
| VisualCallBack            | 290   |   | √ |   |   |   |   |   |   |   |   | √ | √ | √ | √ |   |   |   |
| DetectionMetricsEvaluator |       |   | √ | √ |   |   |   |   |   |   |   | √ |   |   |   |   | √ |   |
| DetectionProgressLogger   |       |   |   |   |   |   |   |   |   |   |   | √ |   |   | √ |   |   |   |
| LearningRateScheduler     |       |   | √ | √ |   |   |   |   |   |   |   | √ |   |   |   |   |   |   |
| TimmTrainerCallback       |       |   | √ | √ |   | √ | √ |   |   |   |   |   | √ |   | √ |   |   |   |
|                           |       | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10| 11| 12| 13| 14| 15| 16| 17|
