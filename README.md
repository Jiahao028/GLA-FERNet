# GLA-FERNet

This is the code implementation for the paper "A gradient-based lightweight network automated design method for facial expression recognition", Expert Systems with Applications, 2026.

## Requirements

- Python 3.6.15
- torch 1.10.1
- torchvision 0.11.2
- numpy 1.19.5

## How to run (Example of RAF-DB)

**Setp 1. Data Preparation**

- Download basic emotions dataset of [RAF-DB](http://www.whdeng.cn/RAF/model1.html#dataset), and make sure it have a structure like following:
 
```
- datapath/raf-basic/
         EmoLabel/
             list_patition_label.txt
             new_10_noise.txt
             new_20_noise.txt
             new_30_noise.txt
         Image/aligned/
             train_00001_aligned.jpg
             test_0001_aligned.jpg
             ...
```

Note: The `new_10_noise.txt` and `new_20_noise.txt` can be download from [Wang Kai's original code](https://github.com/kaiwang960112/Self-Cure-Network).

- Download basic emotions dataset of [AffectNet](http://mohammadmahoor.com/pages/databases/affectnet/), and make sure it have a structure like following:
 
```
- datapath/AffectNet/
         train_class/
             class001/
                image0000002.jpg
                ...
             class002/
             ...
             class008/
         val_class/
             class001/
                image0000002.jpg
                ...
             class002/
             ...
             class008/
```

**Step 2. Change the path where the dataset is loaded to your dataset path and choice the noise level.**

The changes can be found on lines 144, 146 and 148 of the `data_loader.py`.

```
  get_train_and_valid_loader('datapath/raf-basic/',64,1,True,True,0.5)
  get_train_loader('datapath/raf-basic/', 64, 1, True, True)
  get_test_loader('datapath/raf-basic/', 64, 1, False, True)
```

The noise level can be changed on lines 25-28 of the `data_loader.py`.

**Step 3. Search an architecture.**

- Select the sampling method on lines 18-19 of the `train_search.py` and the ratio of channel division M is correspond to the k on line 40 of the file `model_search_***.py` (`model_search_softmax_sogmoid.py` is used for sequential sampling and `model_search_ran_softmax_sogmoid` is used for random sampling).
- Run `train_search.py` to perform the search process (If the server has multiple GPUs, remember to modify the ID of the GPU being used in line 35), and the result will be generated in the folder `search-EXP-{data}-{time}`.
- Find the highest accuracy (best_cc) of the optimal architecture from the last line of the log file `log.txt` in the folder. Then search the log file from scratch to find the architecture `genotype` with the highest accuracy when it first appears (valid_acc).

**Step 4. Validate the performance of the searched architecture.**

- Replace the architecture in the `genotypes.py` file with the searched architecture (line 75).
- Run `train.py` to train the searched architecures (If the server has multiple GPUs, remember to modify the ID of the GPU being used in line 30), and the result will be generated in the folder `eval-EXP-{data}-{time}`.

## Reference
BibTex:
```
@article{fan2025gradient,
  title={A gradient-based lightweight network automated design method for facial expression recognition},
  author={Fan, Jiahao and Deng, Shuchao and Song, Xiaotian and Liu, Jiyuan and Sun, Yanan},
  journal={Expert Systems with Applications},
  pages={129130},
  year={2026},
  publisher={Elsevier}
}
```

GB/T 7714:
```
Fan J, Deng S, Song X, et al. A gradient-based lightweight network automated design method for facial expression recognition[J]. Expert Systems with Applications, 2026, 269: 129130.
```

