# ZDO_2021 - detekce kleštíků
Autor : Adam Klečka

* [coco_parse_dataset.py](coco_parse_dataset.py) -> Pro extrahování pozitivní a negativní množiny dat z COCO formátu.
* [train.py](train.py) -> Pro natrénování klasifikátoru.
* [evaluate.py](evaluate.py) -> Počítá F1 skóre pro testovací dataset. Využívá [prediction.py](prediction.py) a [test.py](test.py). (spustitelné)

* [test_zdo2021.py](test_zdo2021.py) -> Spustitelný test.
* [visual.ipynb](visual.ipynb) -> Také spustitelný test v jupyter notebooku.
* Složka [tests](tests) a [zdo2021](zdo2021) obsahuje spustitelný test ze zadání.

![](img/det_0.png 'před detekcí') ![](img/det_1.jpg 'po detekci')

