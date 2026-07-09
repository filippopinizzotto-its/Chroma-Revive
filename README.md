# ChromaRevive

Colorizzazione di immagini mediante intelligenza artificiale

---

## Sommario

1. [Introduzione](#introduzione)
2. [Setup e requisiti](#setup-e-requisiti)
3. [Spiegazione del progetto](#spiegazione-del-progetto)
4. [Caratteristiche](#caratteristiche)
5. [Architettura del sistema](#architettura-del-sistema)
6. [Utilizzo](#utilizzo)
7. [Struttura del progetto](#struttura-del-progetto)
8. [Dati](#dati)
9. [Modelli disponibili](#modelli-disponibili)
10. [Addestramento dei modelli](#addestramento-dei-modelli)
11. [Aspetti tecnici](#aspetti-tecnici)
12. [Ciclo di vita ML](#ciclo-di-vita-ml)
13. [MLOps](#mlops)
14. [Rischi, assunzioni e limiti](#rischi-assunzioni-e-limiti)
15. [Ulteriori informazioni](#ulteriori-informazioni)

---

## Introduzione

ChromaRevive e' una piattaforma di colorizzazione intelligente di immagini che utilizza modelli di deep learning. Il progetto consente di trasformare automaticamente fotografie in bianco e nero in immagini a colori, combinando un backend robusto basato su FastAPI con un'interfaccia web moderna e intuitiva.

Lo scopo di questo progetto e' dimostrare l'applicazione pratica delle reti neurali convoluzionali per il compito di image-to-image translation nel dominio della colorizzazione automatica.

---

## Setup 

### Installazione delle dipendenze

1. Clonare il repository:

```bash
git clone https://github.com/<utente>/Chroma-Revive.git
cd Chroma-Revive
```

2. Creare un ambiente virtuale (consigliato):

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

3. Installare le dipendenze:

```bash
pip install -r requirements.txt
```

Il file `requirements.txt` include le seguenti dipendenze:

| Libreria | Utilizzo |
|---|---|
| torch | Framework di deep learning per l'inference dei modelli |
| torchvision | Modelli pre-addestrati (ResNet18) e trasformazioni |
| fastapi | Framework web per l'API REST |
| uvicorn | Server ASGI per eseguire FastAPI |
| pillow | Caricamento e manipolazione delle immagini |
| numpy | Elaborazione numerica di array e tensori |
| scikit-image | Conversioni nello spazio colore LAB |
| opencv-python-headless | Ridimensionamento immagini ad alta qualita' |
| python-multipart | Gestione upload di file via form multipart |

Nota: il file requirements.txt utilizza `--extra-index-url https://download.pytorch.org/whl/cpu` per installare la versione CPU-only di PyTorch, riducendo significativamente lo spazio occupato. Se si dispone di una GPU CUDA, e' possibile rimuovere questa riga e installare la versione completa di PyTorch dal sito ufficiale.

### Avviare il progetto

```bash
# Avviare il server backend
uvicorn main:app --reload --port 8000
```

Il server sara' accessibile all'indirizzo `http://localhost:8000`. L'interfaccia web e' servita direttamente dal backend tramite il mounting dei file statici: aprire `http://localhost:8000` nel browser per accedere all'interfaccia completa.

In alternativa, e' possibile avviare il server con:

```bash
python main.py
```

---

## Spiegazione del progetto

### Obiettivo

ChromaRevive affronta il problema della colorizzazione automatica di immagini in scala di grigi. L'obiettivo e' fornire uno strumento pratico e accessibile che permetta di ridare colore a fotografie storiche, immagini d'archivio e qualsiasi foto in bianco e nero, senza richiedere competenze di fotoritocco manuale.

### Quale problema risolve

La colorizzazione manuale di fotografie e' un processo lungo, costoso e che richiede competenze artistiche specializzate. ChromaRevive automatizza questo processo utilizzando reti neurali addestrate su decine di migliaia di immagini, producendo risultati plausibili in pochi secondi. Questo e' utile in diversi contesti:

- **Restauro fotografico**: ridare vita a foto di famiglia storiche o archivi fotografici
- **Ambito educativo e museale**: rendere piu' accessibili e coinvolgenti materiali storici in bianco e nero
- **Produzione di contenuti**: arricchire visivamente contenuti editoriali o documentaristici
- **Ricerca e sperimentazione**: esplorare le capacita' delle reti neurali convoluzionali nel dominio dell'image-to-image translation

### Di che si tratta

Il sistema e' composto da un backend Python (FastAPI) che espone un'API REST per la colorizzazione e da un frontend web (HTML/CSS/JS) con interfaccia glassmorphic. L'utente carica un'immagine in bianco e nero, seleziona uno dei modelli disponibili, e riceve in tempo reale l'immagine colorizzata con uno slider interattivo per confrontare il prima e il dopo.

---

## Caratteristiche

- Supporto per piu' modelli pre-addestrati (COCO, GAN, Skip Connections)
- Interfaccia web interattiva con slider di confronto in tempo reale
- Elaborazione efficiente su CPU e GPU mediante PyTorch
- Elaborazione nello spazio colore LAB per risultati di alta qualita'
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
|
|- Endpoint POST /colorize
|  +-- Caricamento e colorizzazione dell'immagine
|
|- Endpoint GET /models
|  +-- Restituzione della lista dei modelli disponibili
|
|- Endpoint GET /status
|  +-- Controllo stato del server e modello caricato
|
+- Pipeline di elaborazione
   |- Preprocessing (conversione LAB, normalizzazione)
   |- Inference della rete neurale (ColorizerResNet)
   +- Postprocessing (canali ab, ridimensionamento, RGB)
```

### Architettura del modello

La rete neurale utilizza un'architettura encoder-decoder basata su ResNet-18:

- Encoder: Backbone ResNet-18 per l'estrazione delle caratteristiche
- Connessioni di skip: Combinazione di caratteristiche a piu' scale
- Decoder: Upsampling progressivo mediante ConvTranspose2d
- Output: Predizione dei due canali ab dello spazio colore LAB

---

## Utilizzo

### Accedere all'interfaccia web

1. Avviare il server come descritto nella sezione [Setup](#setup)
2. Aprire `http://localhost:8000` nel browser

### Flusso di utilizzo dell'interfaccia

1. Caricare un'immagine: trascinare il file o cliccare per sfogliare
2. Selezionare il modello di colorizzazione desiderato
3. Visualizzare il risultato utilizzando lo slider interattivo per il confronto prima/dopo
4. Scaricare l'immagine colorizzata in formato PNG

### Utilizzo dell'API

Esempio di colorizzazione via API:

```bash
curl -X POST "http://localhost:8000/colorize?model_id=coco30k" \
  -H "accept: image/png" \
  -F "file=@path/to/grayscale_image.jpg"
```

Ottenere la lista dei modelli disponibili:

```bash
curl "http://localhost:8000/models"
```

Verificare lo stato del server:

```bash
curl "http://localhost:8000/status"
```

---

## Struttura del progetto

```
Chroma-Revive/
|-- main.py                              # Server FastAPI e modelli AI
|-- script.js                            # Logica frontend
|-- style.css                            # Stili CSS (design glassmorphic)
|-- index.html                           # Interfaccia web
|-- requirements.txt                     # Dipendenze Python
|-- README.md                            # Questo file
|-- ChromaReviveSkip_gan.ipynb           # Notebook Jupyter per l'addestramento dei modelli
+-- models/                              # Cartella contenente i pesi dei modelli (.pth)
    |-- colorizer_finale_coco30k.pth     # Pesi del modello COCO 30k
    |-- colorizer_finale_Gan.pth         # Pesi del modello GAN
    |-- colorizer_finaleSkip.pth         # Pesi del modello Skip Connections
    +-- colorizer_finale1888.pth         # Pesi del modello Finale 1888
```

---

## Dati

### Dataset utilizzati

I modelli sono stati addestrati su due dataset principali:

1. **COCO 2017 (Common Objects in Context)**: dataset di larga scala contenente oltre 200.000 immagini annotate, da cui sono stati estratti sottoinsiemi di 1.888 e 30.000 immagini per le diverse configurazioni di training.
2. **MirFlickr**: dataset di immagini reali raccolte da Flickr, utilizzato per validazione e per testare la generalizzazione dei modelli su fotografie con stili diversi.

### Motivazione della scelta

- **COCO** e' stato scelto perche' offre un'ampia varieta' di scene (interni, esterni, persone, animali, oggetti), garantendo che il modello impari a colorizzare contesti diversi anziche' specializzarsi su un solo tipo di scena. La presenza di annotazioni e categorie ha permesso di verificare qualitativamente la coerenza dei colori predetti.
- **MirFlickr** e' stato scelto come dataset complementare per la sua natura "in the wild": le immagini non sono curate o selezionate, quindi rappresentano bene il tipo di input che un utente reale potrebbe sottoporre al sistema.

### Caratteristiche dei dati

- Le immagini sono in formato RGB a risoluzione variabile
- Per il training, tutte le immagini vengono ridimensionate a 256x256 pixel
- Le immagini vengono convertite dallo spazio colore RGB allo spazio LAB: il canale L (luminanza) diventa l'input del modello, i canali ab (crominanza) diventano il target da predire
- Il dataset include scene diurne, notturne, interni, esterni, ritratti, paesaggi e nature morte

### Gestione dei dati mancanti e formattazione

- **Immagini gia' in scala di grigi**: le immagini che risultano gia' in scala di grigi (canali RGB identici) sono state incluse normalmente nel training, poiche' lo spazio LAB gestisce nativamente questa casistica (i canali ab saranno prossimi a zero)
- **Immagini corrotte o non leggibili**: filtrate in fase di caricamento tramite un try/except nel DataLoader; le immagini non valide vengono semplicemente saltate
- **Immagini con canale alpha (RGBA)**: convertite forzatamente in RGB mediante `Image.convert('RGB')` prima della conversione LAB
- **Data augmentation**: per contrastare il bias cromatico e l'effetto seppia, sono state applicate trasformazioni casuali (`ColorJitter` su luminosita', contrasto e saturazione, `RandomHorizontalFlip`) esclusivamente sul training set

---

## Modelli disponibili

### 1. COCO Dataset (30k)
- **ID modello**: `coco30k`
- **Pesi**: `colorizer_finale_coco30k.pth`
- **Caratteristiche**: Addestrato su 30.000 immagini del dataset COCO. Offre la massima precisione e fedelta' dei colori su scene ricche e complesse, grazie all'ampia varieta' di scenari inclusi nel dataset di training.
- **Consigliato per**: Uso generale, scene urbane, paesaggi complessi.

### 2. Generative Adversarial Network
- **ID modello**: `gan`
- **Pesi**: `colorizer_finale_Gan.pth`
- **Caratteristiche**: Addestrato utilizzando un framework GAN (Generativa Avversaria) con un Discriminatore custom per valutare la plausibilita' del colore ed evitare l'effetto di colori "piatti" o sfocati. Produce tonalita' vivide, sature e realistiche.
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

L'addestramento dei modelli e' implementato e documentato nel notebook Jupyter `ChromaReviveSkip_gan.ipynb`. Mostra l'addestramento di solo uno dei modelli ma il processo generale è simile per tutti ed è così strutturato:

### 1. Gestione dei Dati e Preprocessing
- **Dataset**: Viene utilizzato principalmente il dataset **COCO 2017** o **Mirflickr**, scaricato ed estratto dinamicamente tramite `kagglehub`.
- **Data Augmentation**: Per evitare l'effetto seppia o tonalita' cromatiche monotone, vengono applicati trasformazioni casuali come `ColorJitter` (modifica di luminosita', contrasto, saturazione) e `RandomHorizontalFlip` sulle immagini di addestramento.
- **Spazio Colore LAB**: Le immagini originali RGB vengono convertite nello spazio colore LAB:
  - Il canale **L** (luminosita', normalizzato tra -1 e 1) viene separato e usato come input.
  - I canali **ab** (crominanza/colori, normalizzati tra -1 e 1) rappresentano il target di predizione per la rete neurale.

### 2. Strategia di Training in Due Fasi (Transfer Learning)
Per sfruttare le feature convoluzionali pre-addestrate senza incorrere nel *catastrophic forgetting* (ossia la perdita delle conoscenze generali dell'encoder), l'addestramento viene suddiviso in due passaggi successivi:
- **Fase 1: Encoder Congelato (Frozen Encoder)**
  - L'encoder (backbone ResNet18 pre-addestrato su ImageNet) viene congelato (`requires_grad = False`).
  - Viene addestrato unicamente il decoder custom (composto da blocchi convoluzionali trasposti `DecoderBlock` con `BatchNorm2d` e attivazioni `ReLU`).
  - Questo passaggio iniziale costringe il decoder a imparare a interpretare le feature dell'encoder e a ricostruire i canali colore senza alterare la stabilita' del backbone.
- **Fase 2: Fine-Tuning Completo**
  - L'intera rete viene sbloccata per ottimizzare tutti i pesi congiuntamente.
  - Viene impostato un **Learning Rate differenziato**: l'encoder viene aggiornato con un learning rate estremamente basso (es. `1e-5`) per preservare le feature estratte e non distruggere la conoscenza pre-acquisita di ImageNet; il decoder viene invece aggiornato con un learning rate standard (es. `1e-4`).
  - Viene applicato lo scheduler `ReduceLROnPlateau` per monitorare la loss sul validation set e dimezzare il learning rate in caso di stagnazione dell'addestramento (patience=3).

### 3. Addestramento Avversariale (Generative Adversarial Network)
Per il modello `gan`, l'addestramento integra una rete **Discriminatore** custom (un classificatore convoluzionale binario):
- **Generatore (Generator)**: La nostra rete encoder-decoder (U-Net con skip connections) riceve il canale L ed elabora i canali ab stimati, cercando di massimizzare la probabilita' che il Discriminatore li classifichi come "reali".
- **Discriminatore (Discriminator)**: Impara a distinguere tra immagini reali `(L, ab_reali)` e immagini colorizzate artificialmente dal generatore `(L, ab_generati)`.
- **Loss del Generatore**: E' una loss combinata formata dalla **Loss Adversariale** (Binary Cross Entropy) e dalla **Loss L1** (moltiplicata per un fattore di scala, es. 100). La loss L1 guida la rete a ricostruire accuratamente la struttura cromatica originale, mentre la loss avversariale spinge il modello a produrre colori vibranti, saturi e privi di sfocature grigie tipiche dei soli approcci basati su regressione.

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

## Ciclo di vita ML

### 1. Raccolta dati

I dati sono stati ottenuti dai dataset pubblici COCO 2017 e MirFlickr, scaricati tramite la libreria `kagglehub`. Non e' stata necessaria una raccolta dati proprietaria, poiche' la colorizzazione e' un task self-supervised: qualsiasi immagine a colori puo' essere usata come ground truth convertendola in scala di grigi e chiedendo al modello di ricostruire i canali colore. Questo approccio rende la raccolta dati estremamente scalabile.

### 2. Training

L'addestramento e' avvenuto in locale e/o su piattaforme cloud (Google Colab) utilizzando PyTorch. Come descritto nella sezione [Addestramento dei modelli](#addestramento-dei-modelli), la strategia adottata comprende transfer learning in due fasi e, per il modello GAN, addestramento avversariale. I modelli sono stati addestrati su sottoinsiemi di dimensioni diverse (1.888 e 30.000 immagini) per confrontare l'impatto della quantita' di dati sulla qualita' dei risultati.

### 3. Validazione

La validazione e' stata eseguita tramite:
- **Loss sul validation set**: monitoraggio della loss L1 su un sottoinsieme di immagini non viste durante il training, utilizzata anche dallo scheduler `ReduceLROnPlateau` per regolare il learning rate
- **Valutazione qualitativa**: ispezione visiva dei risultati su immagini di test, confrontando manualmente la plausibilita' cromatica e la coerenza dei colori predetti rispetto alle immagini originali
- **Confronto tra modelli**: i quattro modelli prodotti sono stati confrontati sulle stesse immagini di test per identificare punti di forza e debolezza di ciascuna architettura

### 4. Deploy

Il deploy avviene attualmente in locale tramite un server FastAPI con Uvicorn. L'architettura e' gia' predisposta per un deployment in produzione:
- L'API REST e' stateless e puo' essere containerizzata con Docker
- I pesi dei modelli sono separati dal codice e caricati dinamicamente
- Il frontend e' composto da file statici serviti direttamente dal backend

Per un deploy in produzione si potrebbe valutare il packaging in un container Docker e il deployment su servizi cloud (AWS EC2/ECS, Google Cloud Run, Azure Container Instances).

### 5. Monitoring

Al momento non e' implementato un sistema di monitoring automatico in produzione. In un'ottica di evoluzione del progetto, le metriche da monitorare includerebbero:
- Latenza di inference per richiesta
- Tasso di errore delle API
- Distribuzione delle immagini in input (dimensione, formato, risoluzione)
- Feedback qualitativo degli utenti sulla resa cromatica

Queste informazioni sono approfondite nella sezione [MLOps](#mlops).

---

## MLOps

### Cosa monitorare

In un contesto di produzione, le seguenti metriche andrebbero monitorate sistematicamente:

**Metriche di performance del modello**:
- Distribuzione dei valori dei canali ab predetti: uno shift verso valori prossimi a zero indicherebbe che il modello sta "collassando" verso una colorizzazione grigia/seppia
- Saturazione media delle immagini prodotte: un calo nel tempo potrebbe indicare degrado del modello
- Tempo di inference medio: un aumento potrebbe indicare problemi infrastrutturali o immagini di input anomale

**Metriche di sistema**:
- Utilizzo di CPU/RAM durante l'inference
- Numero di richieste al minuto e distribuzione temporale
- Tasso di errore e tipologie di errore (immagini corrotte, formati non supportati, timeout)

**Metriche di utilizzo**:
- Modello selezionato con piu' frequenza dagli utenti
- Dimensione e formato delle immagini caricate
- Tasso di download delle immagini colorizzate (indicatore di soddisfazione implicita)

### Criteri per il re-training

Il re-training dei modelli andrebbe considerato nei seguenti scenari:

1. **Data drift**: se le immagini sottomesse dagli utenti in produzione hanno caratteristiche sistematicamente diverse dal training set (es. prevalenza di foto satellitari, radiografie mediche, illustrazioni digitali), i modelli potrebbero produrre risultati inadeguati. In tal caso, raccogliere un campione rappresentativo delle nuove tipologie e rifare il training su un dataset arricchito.

2. **Feedback negativo ricorrente**: se gli utenti segnalano ripetutamente risultati insoddisfacenti su una certa categoria di immagini, potrebbe essere necessario un fine-tuning mirato su quella categoria.

3. **Nuove architetture disponibili**: l'evoluzione dello stato dell'arte in ambito image-to-image (es. modelli basati su Transformer o Diffusion) potrebbe rendere opportuno il re-training con architetture piu' performanti.

4. **Aumento del dataset**: la disponibilita' di nuovi dataset pubblici o proprietari di qualita' superiore giustificherebbe un nuovo ciclo di training per migliorare la generalizzazione.

---

## Rischi, assunzioni e limiti

### Limiti noti

- **Ambiguita' cromatica intrinseca**: la colorizzazione e' un problema intrinsecamente ambiguo (un'auto in scala di grigi potrebbe essere rossa, blu, bianca, ecc.). Il modello produce una colorizzazione plausibile, ma non necessariamente quella corretta. Non esiste una soluzione unica al problema.
- **Risoluzione fissa di inference**: l'input viene ridimensionato a 256x256 per l'inference, e i canali colore vengono poi upscalati alla risoluzione originale. Questo puo' causare una leggera perdita di dettaglio cromatico su immagini ad altissima risoluzione, dove il colore appare piu' "sfumato" rispetto ai bordi.
- **Bias del dataset**: i modelli addestrati su COCO tendono a prediligere colori tipici delle scene presenti nel dataset (paesaggi verdi, cieli azzurri, pelle umana). Immagini con soggetti poco rappresentati nel training set (es. cibo esotico, opere d'arte astratta, ambienti industriali) possono ricevere colorizzazioni meno accurate.
- **Effetto seppia/grigio**: nonostante le misure adottate (data augmentation, training GAN), in alcuni casi i modelli piu' semplici (final1888) possono produrre colorizzazioni tendenti al seppia, specialmente su immagini con basso contrasto.
- **Singola risoluzione di training**: l'architettura attuale e' ottimizzata per 256x256. Adattarla a risoluzioni diverse richiederebbe modifiche architetturali.

### Assunzioni

- Si assume che l'immagine in input sia effettivamente in scala di grigi o convertibile in scala di grigi. Il sistema accetta anche immagini a colori, ma in tal caso l'output potrebbe differire significativamente dall'originale poiche' il modello reinterpreta la luminanza ignorando i colori preesistenti.
- Si assume che l'utente disponga di una connessione di rete tra frontend e backend (localhost), anche se in modalita' locale.
- Si assume che i pesi dei modelli siano integri e non corrotti. Non e' implementato un controllo di integrita' (hash checksum) sui file `.pth`.

### Rischi

- **Sicurezza dell'API**: l'endpoint `/colorize` accetta qualsiasi file e attualmente non impone limiti sulla dimensione o sul numero di richieste. In un contesto di deployment pubblico, sarebbe necessario implementare rate limiting, validazione dei file e autenticazione.
- **Dipendenza da librerie esterne**: il progetto dipende da PyTorch, che e' una libreria di grandi dimensioni soggetta a frequenti aggiornamenti. Cambiamenti nelle API di PyTorch o nelle dipendenze transitive potrebbero richiedere manutenzione.

### Stato del progetto

Il progetto e' funzionante dall'inizio alla fine: e' possibile avviare il server, caricare un'immagine, selezionare un modello, ottenere la colorizzazione e scaricare il risultato. L'intero flusso e' operativo sia tramite interfaccia web sia tramite API.

Possibili ampliamenti futuri:
- Supporto per la colorizzazione batch (piu' immagini in una singola richiesta)
- Aggiunta di un sistema di feedback dell'utente per raccogliere valutazioni qualitative
- Implementazione di modelli basati su architetture piu' recenti (Vision Transformer, Diffusion Models)
- Colorizzazione di video frame-by-frame con coerenza temporale
- Containerizzazione con Docker e deployment su cloud
- Aggiunta di controlli manuali per guidare la colorizzazione (es. suggerire il colore di un oggetto specifico)

---

## Ulteriori informazioni

### Tecnologie e framework utilizzati

| Componente | Tecnologia |
|---|---|
| Backend | Python, FastAPI, Uvicorn |
| Deep Learning | PyTorch, torchvision |
| Elaborazione immagini | Pillow, scikit-image, OpenCV |
| Frontend | HTML5, CSS3, JavaScript vanilla |
| Addestramento | Jupyter Notebook, kagglehub |
| Architettura NN | ResNet-18 (encoder), U-Net custom (decoder) |

### Scelte progettuali rilevanti

- **Spazio colore LAB anziche' RGB**: la scelta di operare nello spazio LAB e' motivata dal fatto che permette di separare nettamente la luminanza (L) dal colore (ab). Questo consente al modello di concentrarsi esclusivamente sulla predizione del colore, preservando inalterata la struttura luminosa dell'immagine originale. In uno spazio RGB, il modello dovrebbe predire tutti e tre i canali, rischiando di alterare anche la luminosita'.
- **ResNet-18 come encoder**: si e' preferita ResNet-18 rispetto a backbone piu' profondi (ResNet-50, ResNet-101) per bilanciare qualita' e velocita' di inference su CPU. La compattezza del modello (~13 MB per i pesi) permette un deployment leggero.
- **Frontend statico servito dal backend**: la scelta di servire i file statici direttamente da FastAPI semplifica il deployment eliminando la necessita' di un web server separato (Nginx, Apache).
- **Modelli multipli caricabili a runtime**: il sistema supporta il caricamento dinamico dei pesi, evitando di mantenere piu' istanze del modello in memoria e consentendo di aggiungere nuovi modelli semplicemente inserendo file `.pth` nella cartella `models/`.

### Riferimenti

- He et al., "Deep Residual Learning for Image Recognition" (2015) - Architettura ResNet
- Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015) - Skip connections
- Isola et al., "Image-to-Image Translation with Conditional Adversarial Networks" (pix2pix, 2017) - Framework GAN per image translation
- Zhang et al., "Colorful Image Colorization" (2016) - Riferimento per la colorizzazione automatica nello spazio LAB
- Dataset COCO: https://cocodataset.org/
- Dataset MirFlickr: https://press.liacs.nl/mirflickr/

---

## Note finali

Questo progetto rappresenta un'applicazione pratica delle reti neurali convoluzionali per il compito di colorizzazione automatica di immagini. Il codice e' strutturato per essere didattico e facilmente estendibile con ulteriori modelli o funzionalita'.
