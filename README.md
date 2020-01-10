# Chinese NER using Bert

BERT for Chinese NER. 

## dataset list

1. cner: datasets/cner
2. CLUENER: https://github.com/CLUEbenchmark/CLUENER

## model list

1. BERT+Softmax
2. BERT+CRF
3. BERT+Span
4. BERT+Span+label_smoothing
5. BERT+Span+focal_loss

## requirement

1. pytorch=1.1.0
2. cuda=9.0

## input format

Input format (prefer BIOS tag scheme), with each character its label for one line. Sentences are splited with a null line.

```text
美	B-LOC
国	I-LOC
的	O
华	B-PER
莱	I-PER
士	I-PER

我	O
跟	O
他	O
谈	O
笑	O
风	O
生	O 
```

## run the code

1. Modify the configuration information in `run_ner_xxx.py` or `run_ner_xxx.sh` .
2. `sh run_ner_xxx.sh`

**note**: file structure of the model

```text
├── prev_trained_model
|  └── bert_base
|  |  └── pytorch_model.bin
|  |  └── config.json
|  |  └── vocab.txt
|  |  └── ......
```

## CLUENER result

Tne overall performance of BERT on **dev**:

|              | Accuracy (entity)  | Recall (entity)    | F1 score (entity)  |                                                              |
| ------------ | ------------------ | ------------------ | ------------------ | ------------------------------------------------------------ |
| BERT+Softmax | 0.7916     | 0.7962     | 0.7939    | train_max_length=128 eval_max_length=512 epoch=4 lr=3e-5 batch_size=24 |
| BERT+CRF     | 0.7877     | 0.8008 | 0.7942     | train_max_length=128 eval_max_length=512 epoch=5 lr=3e-5 batch_size=24 |
| BERT+Span    | 0.8132 | **0.8092** | **0.8112** | train_max_length=128 eval_max_length=512 epoch=4 lr=3e-5 batch_size=24 |
| BERT+Span+focal_loss    | 0.8121 | 0.8008 | 0.8064 | train_max_length=128 eval_max_length=512 epoch=4 lr=3e-5 batch_size=24 loss_type=focal |
| BERT+Span+label_smoothing   | **0.8235** | 0.7946 | 0.8088 | train_max_length=128 eval_max_length=512 epoch=4 lr=3e-5 batch_size=24 loss_type=lsr |


## Cner result

Tne overall performance of BERT on **dev(test)**:

|              | Accuracy (entity)  | Recall (entity)    | F1 score (entity)  |                                                              |
| ------------ | ------------------ | ------------------ | ------------------ | ------------------------------------------------------------ |
| BERT+Softmax | 0.9586(0.9566)     | 0.9644(0.9613)     | 0.9615(0.9590)     | train_max_length=128 eval_max_length=512 epoch=4 lr=3e-5 batch_size=24 |
| BERT+CRF     | 0.9562(0.9539)     | 0.9671(**0.9644**) | 0.9616(0.9591)     | train_max_length=128 eval_max_length=512 epoch=10 lr=3e-5 batch_size=24 |
| BERT+Span    | 0.9604(**0.9620**) | 0.9617(0.9632)     | 0.9611(**0.9626**) | train_max_length=128 eval_max_length=512 epoch=4 lr=3e-5 batch_size=24 |
| BERT+Span+focal_loss    | 0.9516(0.9569) | 0.9644(0.9681)     | 0.9580(0.9625) | train_max_length=128 eval_max_length=512 epoch=4 lr=3e-5 batch_size=24 loss_type=focal |
| BERT+Span+label_smoothing   | 0.9566(0.9568) | 0.9624(0.9656)     | 0.9595(0.9612) | train_max_length=128 eval_max_length=512 epoch=4 lr=3e-5 batch_size=24 loss_type=lsr |


The entity performance performance of BERT on **test**:

|                  | CONT   | ORG    | LOC    | EDU    | NAME   | PRO    | RACE   | TITLE  |
| ---------------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| **BERT+Softmax** |        |        |        |        |        |        |        |        |
| Accuracy         | 1.0000 | 0.9446 | 1.0000 | 0.9911 | 1.0000 | 0.8919 | 1.0000 | 0.9545 |
| Recall           | 1.0000 | 0.9566 | 1.0000 | 0.9911 | 1.0000 | 1.0000 | 1.0000 | 0.9508 |
| F1 Score         | 1.0000 | 0.9506 | 1.0000 | 0.9911 | 1.0000 | 0.9429 | 1.0000 | 0.9526 |
| **BERT+CRF**     |        |        |        |        |        |        |        |        |
| Accuracy         | 1.0000 | 0.9446 | 1.0000 | 0.9823 | 1.0000 | 0.9687 | 1.0000 | 0.9591 |
| Recall           | 1.0000 | 0.9566 | 1.0000 | 0.9911 | 1.0000 | 0.9697 | 1.0000 | 0.9534 |
| F1 Score         | 1.0000 | 0.9506 | 1.0000 | 0.9867 | 1.0000 | 0.9697 | 1.0000 | 0.9552 |
| **BERT+Span**    |        |        |        |        |        |        |        |        |
| Accuracy         | 1.0000 | 0.9378 | 1.0000 | 0.9911 | 1.0000 | 0.9429 | 1.0000 | 0.9685 |
| Recall           | 1.0000 | 0.9548 | 1.0000 | 0.9911 | 1.0000 | 1.0000 | 1.0000 | 0.9560 |
| F1 Score         | 1.0000 | 0.9462 | 1.0000 | 0.9911 | 1.0000 | 0.9706 | 1.0000 | 0.9622 |



