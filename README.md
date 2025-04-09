# MDSBR

## Datasets

We use the Amazon datasets Cellphones, Grocery, Sports and Yelp. Make sure you download the datasets from the Amazon and Yelp.

## Settings

```
python = 3.8
pytorch = 2.1.0
transformers = 4.36.2
cuda = 12.1 
```

## DataProcessing

Enter the data folder for data processing:

```
cd preprocess
```

Then use the following files to process your data：

```
preprocess1.ipynb
preprocess2.ipynb
preprocess3.ipynb
```

Then run this command to get image and text about item:

```
python imagedownload.py
```

Then run this command to get image and text embedding:

```
python processimage.py
python processtext.py
```

## Train

Please make sure all datas are in corresponding folder location, then run this command to Training and Prediction:

```
python main.py
```
