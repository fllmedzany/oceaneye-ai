# oceaneye-ai
[![en](https://img.shields.io/badge/lang-en-red.svg)](https://github.com/fllmedzany/oceaneye-ai/blob/main/README.EN.md)
[![sk](https://img.shields.io/badge/lang-sk-blue.svg)](https://github.com/fllmedzany/oceaneye-ai/blob/main/README.md)

Toto je návod ako si natrénovať model umelej inteligenice na rozpoznávanie druhov rýb. Na trénovanie použijeme python a knižnice ultralytics, dataset vytvoríme v programe LabelStudio (opensource)

**Anotovanie a dataset**

- fotky môžete anotovať v programe LabelStudio (opensource)
- najprv si spravíte svoj účet
- potom si vytvoríte projekt, do ktorého importujete svoje fotky a vyberiete si typ anotovania a pridáte si classes, do ktorých budete chcieť triediť
- potom môžte rovno anotovať
- keď to budete mať hotové, stiahnite si dataset vo formáte YOLOv8


**Keď máme dataset, možme ísť trénovať. Vhodný je počítač s NVIDIA grafikou, urýchli to proces**

**1\. Nastavenie prostredia (Windows)**
1. **Aktualizujte ovládače a nainštalujte CUDA (voliteľné)**
    - ak nemáš NVIDIA tak toto vynechaj
    - Na NVIDIA RTX 4060 by malo stačiť, ak máte aktuálny driver z [NVIDIA stránky](https://www.nvidia.com/Download/index.aspx).
    - Oficiálna inštalácia CUDA Toolkitu (z [developer.nvidia.com](https://developer.nvidia.com/cuda-toolkit)) nie je nevyhnutná, ak nainštalujete PyTorch s podporou GPU, ale môže byť užitočná.
    - Na overenie, či CUDA a ovládače bežia, môžete využiť príkaz nvidia-smi v termináli.
3. **Nainštalujte si Python (3.8+, ideálne 3.9 alebo 3.10)**
    - Odporúčame vytvoriť si virtuálne prostredie cez venv.
    ```
    cd c:\
    mkdir moje_projekty
    cd moje_projekty
    c:\Python310\python.exe -m venv c:\moje_projekty\oceaneye-ai
    c:\moje_projekty\oceaneye-ai\Scripts\activate
    ```
4. **Nainštalujte PyTorch s podporou CUDA (len ak máč NVIDIA GPU)**
    - Na stránke [pytorch.org](https://pytorch.org/get-started/locally/) si vygenerujte príkaz pre inštaláciu verzie s podporou CUDA. Napr. (príklad, môže sa meniť podľa verzií):
    - ```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118```
    - Skontrolujte, či vám PyTorch rozpoznal GPU:
    ```
    python
    import torch
    print(torch.cuda.is_available()) # malo by vypísať True
    ```
5. **Nainštalujte ultralytics (YOLOv8)**
    - ```pip install ultralytics```

**2\. Príprava dát (dataset)**

Predpokladajme, že ide o **detekčný** dataset (čiže máte bounding boxy (ohraničujúce boxy) pre ryby).
Vytvorte si dataset napríklad v programe [LabelStudio](https://labelstud.io/) 

**Dataset v YOLO formáte**

YOLOv8 očakáva tieto súbory, label studio vám to vyexportuje, musíte si dorobit potom súbor data.yaml a rozdeliť dataset na train a val v pomere 80:20 alebo 90:10 

- **images** (zložka) – obrázky v JPEG/PNG formáte.
- **labels** (zložka) – textové súbory so značkami (bounding box) k obrázkom.
- **data.yaml** (konfiguračný súbor) – definuje, kde sú dáta a aké triedy (názvy rýb).

Struktúra napríklad takto:
```
fish_dataset/
├── images
│ ├── train
│ │ ├── img001.jpg
│ │ ├── img002.jpg
│ │ └── ...
│ └── val
│ ├── img101.jpg
│ ├── ...
├── labels
│ ├── train
│ │ ├── img001.txt
│ │ ├── ...
│ └── val
│ ├── img101.txt
│ ├── ...
└── data.yaml
```
**Každý .txt súbor** k obrázku vyzerá napr. takto (YOLO formát):
```
0 0.5 0.5 0.2 0.3
1 0.4 0.6 0.1 0.1
```
Kde prvé číslo je **ID triedy** (napr. 0 = Mečúň, 1 = Mrenka, …) a ďalšie štyri čísla sú **x_center, y_center, width, height** v normalizovaných súradniciach (0–1).

**Súbor data.yaml** vyzerá napr. takto:
```
train: path/to/fish_dataset/images/train
val: path/to/fish_dataset/images/val
names:
    0: Mečúň
    1: Mrenka
    2: Ostatné ryby
```
(Prispôsobte podľa toho, koľko máte tried a aké majú názvy.)

Ak už máte dataset v nejakom inom formáte (napr. COCO JSON, Pascal VOC XML a pod.), môžete využiť nejaké konverzné nástroje alebo priamo YOLOv8, ktoré niektoré konverzie zvláda.

**3\. Spustenie tréningu**

**Základné trénovanie s YOLOv8**  
    Dá sa to spustiť úplne “z príkazovej riadky” alebo cez Python.

**Z príkazového riadku**:
```
yolo detect train data=path/to/data.yaml model=yolov8n.pt epochs=50 imgsz=640
```
- - model=yolov8n.pt znamená, že sa použije predtrénovaný základ YOLOv8 nano (najmenší model). My sme použili medium yolov8m.pt
    - epochs=50 napr. 50 epôch.
    - imgsz=640 veľkosť vstupného obrázka.

**Z Python skriptu** (napr. train.py):
```
from ultralytics import YOLO
\# Načítame základný YOLOv8 n model
model = YOLO("yolov8n.pt")
\# Spustíme tréning
model.train(
data="path/to/data.yaml",
epochs=50,
imgsz=640,
project="fish_detect",
name="exp1"
)
```

Po skončení tréningu bude v priečinku fish_detect/exp1/ uložený najlepší model, typicky best.pt.

**Vyhodnotenie modelu**  
    YOLO sám pri tréningu na konci zobrazí mAP (mean Average Precision) a iné štatistiky.  
    Môžete spustiť aj samostatný príkaz na vyhodnotenie (napr. keď máte hotový model):
```yolo detect val model=fish_detect/exp1/best.pt data=path/to/data.yaml```

Alebo v Pythone:

```
model = YOLO("fish_detect/exp1/best.pt")
metrics = model.val()
print(metrics)
```

**Detekcia (inferencia) na novom videu**  
    Napríklad, máte video fish_aquarium.mp4, a chcete vidieť detekcie:
```yolo detect predict model=fish_detect/exp1/best.pt source=fish_aquarium.mp4```

Knižnica automaticky vytvorí priečinok runs/detect/predict/ a tam uloží výsledné video s detekciami.

**V Pythone:**
```
model = YOLO("fish_detect/exp1/best.pt")
results = model.predict(source="fish_aquarium.mp4", show=True, save=True)
```
- show=True zobrazí spracované video v okne (pokiaľ to systém podporuje).
- save=True uloží video s detekciami do runs/detect/predictX/.

**4\. Praktické tipy**

1. **Veľkosť a vyváženosť datasetu**
    - Pre spoľahlivú detekciu treba aspoň pár stoviek obrázkov s rybami (a vyznačenými boundig boxami).
    - Dbajte, aby tam boli rôzne uhly pohľadu, rôzne svetelné podmienky (akvárium s rozličným osvetlením), viacero typov rýb, aby to vedelo generalizovať.
2. **Augmentácie**
    - YOLOv8 má v základe aplikované rozumné augmentácie (random flipy, orezanie, apod.), ktoré pomáhajú zlepšiť robustnosť.
    - Netreba väčšinou nič extra nastavovať, ale dá sa to ľahko doladiť, ak by ste chceli špecifické augmentácie.
3. **Výber modelu**
    - YOLOv8 nano (yolov8n.pt) je veľmi rýchly a malý. Môže sa hodiť, ak chcete bežať detekciu v reálnom čase aj na slabšom HW.
    - Pre lepšiu presnosť môžete skúsiť yolov8s.pt (small), yolov8m.pt (medium), atď.
    - Čím väčší model, tým náročnejší tréning. Ale RTX 4060 by mala hravo zvládnuť aspoň small či medium verziu.

**5\. Zhrnutie**

1. Vytvorte si virtuálne prostredie (env).
2. Nainštalujte PyTorch s podporou CUDA + ultralytics.
3. Dajte svoj dataset do YOLO formátu (images/labels + data.yaml).
4. Spustite tréning:
5. ```yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640```
6. Po skončení tréningu použite model na detekciu na nových obrázkoch či videách:
7. ```yolo detect predict model=best.pt source=video.mp4```
8. (Voliteľne) dolaďte augmentácie, hyperparametre, väčší model, atď.



link na príkladové video našej AI: (https://www.youtube.com/watch?v=LG57YVFjZCo)
