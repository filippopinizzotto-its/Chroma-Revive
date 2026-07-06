# ChromaRevive

Colorizzazione di immagini mediante intelligenza artificiale

---

## Sommario

1. [Introduzione](#introduzione)
2. [Caratteristiche](#caratteristiche)
3. [Architettura del sistema](#architettura-del-sistema)
4. [Utilizzo](#utilizzo)
5. [Struttura del progetto](#struttura-del-progetto)
6. [Modelli disponibili](#modelli-disponibili)
7. [Addestramento dei modelli](#addestramento-dei-modelli)
8. [Aspetti tecnici](#aspetti-tecnici)

---

## Introduzione

ChromaRevive è una piattaforma di colorizzazione intelligente di immagini che utilizza modelli di deep learning. Il progetto consente di trasformare automaticamente fotografie in bianco e nero in immagini a colori, combinando un backend robusto basato su FastAPI con un'interfaccia web moderna e intuitiva.

Lo scopo di questo progetto è dimostrare l'applicazione pratica delle reti neurali convoluzionali per il compito di image-to-image translation nel dominio della colorizzazione automatica.

---

## Caratteristiche

- Supporto per più modelli pre-addestrati (COCO e MirFlickr)
- Interfaccia web interattiva con slider di confronto in tempo reale
- Elaborazione efficiente su CPU e GPU mediante PyTorch
- Elaborazione nello spazio colore LAB per risultati di alta qualità
- Design responsivo con interfaccia glassmorphic
- Cambio del modello senza ricaricare la pagina
- Download delle immagini colorizzate in formato PNG
- API REST per integrazione con applicazioni esterne

---

## Architettura del sistema

### Backend

Il server backend utilizza FastAPI per fornire un'API REST che gestisce la colorizzazione delle immagini:

```
FastAPI Server
│
├─ Endpoint POST /colorize
│  └─ Caricamento e colorizzazione dell'immagine
│
├─ Endpoint GET /models
│  └─ Restituzione della lista dei modelli disponibili
│
└─ Pipeline di elaborazione
   ├─ Preprocessing (conversione LAB, normalizzazione)
   ├─ Inference della rete neurale (ColorizerResNet)
   └─ Postprocessing (canali ab, ridimensionamento, RGB)
```

### Architettura del modello

La rete neurale utilizza un'architettura encoder-decoder basata su ResNet-18:

- Encoder: Backbone ResNet-18 per l'estrazione delle caratteristiche
- Connessioni di skip: Combinazione di caratteristiche a più scale
- Decoder: Upsampling progressivo mediante ConvTranspose2d
- Output: Predizione dei due canali ab dello spazio colore LAB

---

## Utilizzo

### Avviare il server backend

```bash
# Usando Uvicorn
uvicorn main:app --reload --port 8000
```

Il server sarà accessibile all'indirizzo `http://localhost:8000`

### Accedere all'interfaccia web

1. Aprire il file `index.html` nel browser web
2. L'interfaccia consente di caricare immagini in bianco e nero

### Flusso di utilizzo dell'interfaccia

1. Caricare un'immagine: trascinare il file o cliccare per sfogliare
2. Selezionare il modello di colorizzazione desiderato
3. Visualizzare il risultato utilizzando lo slider interattivo per il confronto prima/dopo
4. Scaricare l'immagine colorizzata in formato PNG

### Utilizzo dell'API

Esempio di colorizzazione via API:

```bash
curl -X POST "http://localhost:8000/colorize?model_id=coco15k" \
  -H "accept: image/png" \
  -F "file=@path/to/grayscale_image.jpg"
```

Ottenere la lista dei modelli disponibili:

```bash
curl "http://localhost:8000/models"
```

---

## Struttura del progetto

```
Chroma-Revive/
├── main.py                              # Server FastAPI e modelli AI
├── script.js                            # Logica frontend
├── style.css                            # Stili CSS (design glassmorphic)
├── index.html                           # Interfaccia web
├── requirements.txt                     # Dipendenze Python
├── README.md                            # Questo file
├── ChromaReviveSkip_gan.ipynb           # Notebook Jupyter per l'addestramento dei modelli
└── models/                              # Cartella contenente i pesi dei modelli (.pth)
    ├── colorizer_finale_coco30k.pth     # Pesi del modello COCO 30k
    ├── colorizer_finale_Gan.pth         # Pesi del modello GAN
    ├── colorizer_finaleSkip.pth         # Pesi del modello Skip Connections
    └── colorizer_finale1888.pth         # Pesi del modello Finale 1888
```

---

## Modelli disponibili

### 1. COCO Dataset (30k)
- **ID modello**: `coco30k`
- **Pesi**: `colorizer_finale_coco30k.pth`
- **Caratteristiche**: Addestrato su 30.000 immagini del dataset COCO. Offre la massima precisione e fedeltà dei colori su scene ricche e complesse, grazie all'ampia varietà di scenari inclusi nel dataset di training.
- **Consigliato per**: Uso generale, scene urbane, paesaggi complessi.

### 2. Generative Adversarial Network
- **ID modello**: `gan`
- **Pesi**: `colorizer_finale_Gan.pth`
- **Caratteristiche**: Addestrato utilizzando un framework GAN (Generativa Avversaria) con un Discriminatore custom per valutare la plausibilità del colore ed evitare l'effetto di colori "piatti" o sfocati. Produce tonalità vivide, sature e realistiche.
- **Consigliato per**: Ritratti, foto storiche, risultati ad alto impatto fotorealistico.

### 3. Skip Connections Model
- **ID modello**: `skip`
- **Pesi**: `colorizer_finaleSkip.pth`
- **Caratteristiche**: Architettura U-Net avanzata dotata di skip connections tra l'encoder (ResNet18) e il decoder custom. Aiuta a conservare e trasferire i dettagli geometrici e i bordi ad alta risoluzione direttamente alle fasi finali di colorizzazione.
- **Consigliato per**: Immagini con geometrie complesse, texture definite, illustrazioni.

### 4. Finale 1888
- **ID modello**: `final1888`
- **Pesi**: `colorizer_finale1888.pth`
- **Caratteristiche**: Modello addestrato su un dataset ridotto di 1.888 immagini (indicato nel frontend anche come "30h COCO").
- **Consigliato per**: Test veloci e benchmark leggeri.

Nota: Si consiglia di testare diversi modelli per determinare quale produce la migliore resa cromatica in base alle caratteristiche specifiche della propria immagine.

---

## Addestramento dei modelli

L'addestramento dei modelli è implementato e documentato nel notebook Jupyter [ChromaReviveSkip_gan.ipynb](file:///c:/Users/FilippoPinizzotto/OneDrive%20-%20ITS%20Angelo%20Rizzoli/Desktop/Chroma/Chroma-Revive/ChromaReviveSkip_gan.ipynb). Il processo generale segue una pipeline strutturata:

### 1. Gestione dei Dati e Preprocessing
- **Dataset**: Viene utilizzato principalmente il dataset **COCO 2017** o **Mirflickr**, scaricato ed estratto dinamicamente tramite `kagglehub`.
- **Data Augmentation**: Per evitare l'effetto seppia o tonalità cromatiche monotone, vengono applicati trasformazioni casuali come `ColorJitter` (modifica di luminosità, contrasto, saturazione) e `RandomHorizontalFlip` sulle immagini di addestramento.
- **Spazio Colore LAB**: Le immagini originali RGB vengono convertite nello spazio colore LAB:
  - Il canale **L** (luminosità, normalizzato tra -1 e 1) viene separato e usato come input.
  - I canali **ab** (crominanza/colori, normalizzati tra -1 e 1) rappresentano il target di predizione per la rete neurale.

### 2. Strategia di Training in Due Fasi (Transfer Learning)
Per sfruttare le feature convoluzionali pre-addestrate senza incorrere nel *catastrophic forgetting* (ossia la perdita delle conoscenze generali dell'encoder), l'addestramento viene suddiviso in due passaggi successivi:
- **Fase 1: Encoder Congelato (Frozen Encoder)**
  - L'encoder (backbone ResNet18 pre-addestrato su ImageNet) viene congelato (`requires_grad = False`).
  - Viene addestrato unicamente il decoder custom (composto da blocchi convoluzionali trasposti `DecoderBlock` con `BatchNorm2d` e attivazioni `ReLU`).
  - Questo passaggio iniziale costringe il decoder a imparare a interpretare le feature dell'encoder e a ricostruire i canali colore senza alterare la stabilità del backbone.
- **Fase 2: Fine-Tuning Completo**
  - L'intera rete viene sbloccata per ottimizzare tutti i pesi congiuntamente.
  - Viene impostato un **Learning Rate differenziato**: l'encoder viene aggiornato con un learning rate estremamente basso (es. `1e-5`) per preservare le feature estratte e non distruggere la conoscenza pre-acquisita di ImageNet; il decoder viene invece aggiornato con un learning rate standard (es. `1e-4`).
  - Viene applicato lo scheduler `ReduceLROnPlateau` per monitorare la loss sul validation set e dimezzare il learning rate in caso di stagnazione dell'addestramento (patience=3).

### 3. Addestramento Avversariale (Generative Adversarial Network)
Per il modello `gan`, l'addestramento integra una rete **Discriminatore** custom (un classificatore convoluzionale binario):
- **Generatore (Generator)**: La nostra rete encoder-decoder (U-Net con skip connections) riceve il canale L ed elabora i canali ab stimati, cercando di massimizzare la probabilità che il Discriminatore li classifichi come "reali".
- **Discriminatore (Discriminator)**: Impara a distinguere tra immagini reali `(L, ab_reali)` e immagini colorizzate artificialmente dal generatore `(L, ab_generati)`.
- **Loss del Generatore**: È una loss combinata formata dalla **Loss Adversariale** (Binary Cross Entropy) e dalla **Loss L1** (moltiplicata per un fattore di scala, es. 100). La loss L1 guida la rete a ricostruire accuratamente la struttura cromatica originale, mentre la loss avversariale spinge il modello a produrre colori vibranti, saturi e privi di sfocature grigie tipiche dei soli approcci basati su regressione.

---

## Aspetti tecnici

### Spazio colore LAB

Il modello opera nello spazio colore LAB per risultati ottimali:

- Canale L: Luminanza (brillantezza) - preservato dall'immagine di input
- Canali ab: Crominanza (colore) - predetti dalla rete neurale

### Preprocessing dell'immagine

1. Ridimensionamento dell'immagine originale a 256x256 per l'inference
2. Conversione da RGB a LAB
3. Normalizzazione del canale L: (L / 50) - 1
4. Aggiunta della dimensione batch per l'input del modello

### Inference del modello

1. Forward pass produce predizioni a 2 canali nel range -1 a 1
2. Denormalizzazione a range -128 a 127
3. Upsampling alla risoluzione originale mediante interpolazione bilineare

### Postprocessing dell'output

1. Combinazione dei canali ab predetti con il canale L originale
2. Conversione da LAB a RGB
3. Clipping al range valido e conversione a uint8
4. Salvataggio come immagine PNG

---

## Note finali

Questo progetto rappresenta un'applicazione pratica delle reti neurali convoluzionali per il compito di colorizzazione automatica di immagini. Il codice è strutturato per essere didattico e facilmente estendibile con ulteriori modelli o funzionalità.


