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
7. [Aspetti tecnici](#aspetti-tecnici)

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
├── models/
│   ├── colorizer_finale_coco15k.pth    # Pesi del modello COCO
│   └── colorizer_finale25k.pth         # Pesi del modello MirFlickr
├── inspect_model.py                    # Utility per l'ispezione del modello
└── run_test.py                         # Script di test
```

---

## Modelli disponibili

### Modello COCO Dataset (15000 immagini)

ID: `coco15k`

Caratteristiche:
- Addestrato su 15000 immagini del dataset COCO
- Comprensione di scene diverse
- Copertura ampia di categorie di oggetti
- Colorizzazione equilibrata su diversi domini

Consigliato per: uso generale, soggetti vari

### Modello MirFlickr Dataset (10000 immagini)

ID: `final25k`

Caratteristiche:
- Addestrato su 10000 immagini del dataset MirFlickr
- Precisione elevata nella colorizzazione
- Ottimizzato per contenuti artistici e fotografici
- Risultati esteticamente raffinati

Consigliato per: fotografia professionale, progetti artistici

Nota: Consigliamo di testare entrambi i modelli per determinare quale produce risultati migliori per le vostre immagini.

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


