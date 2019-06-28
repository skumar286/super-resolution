# RNCNN-keras     

### Dependence
```
tensorflow
keras2
numpy
opencv
```

### Prepare train data
```
$ python data.py
```

Clean patches are extracted from 'data/Train400' and saved in 'data/npy_data'.
### Train
```
$ python main.py
```

Trained models are saved in 'snapshot'.
### Test
```
$ python main.py --only_test True --pretrain 'path of saved model'
```

Noisy and denoised images are saved in 'snapshot'.








