# zh
v3:clearml

# step 1：设置数据集
将fer2013.csv文件放置dataset/路径下即可

# checkpoint 1
- 2080Ti+Xeon(R) Gold 6130H
- num_works=4&batch_size=128
- pin_memory=True

|Epoch  |  Time  |  Train Loss    |  Train ACC    |   Val Loss     |   Val ACC||
|--|--|--|--|--|--|--|
|DataRead Time:0.8198294639587402      |  TrainEpoch Time:39.56714701652527|
|1       | 46.0021       |  1.4024     |     24.4202     |    1.8094      |    24.5751     |    saves best|
|DataRead Time:0.4743359088897705    |    TrainEpoch Time:36.5745325088501|
|2       | 42.5028       |  1.3385     |     24.9566      |   1.7989      |    24.4915      |   |
|DataRead Time:0.5064108371734619     |   TrainEpoch Time:34.721256494522095|
|3       | 40.6404       |  1.3371     |     24.9824      |   1.7983      |    24.4636 ||
