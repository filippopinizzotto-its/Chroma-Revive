# Decision Log

Registro cronologico delle decisioni progettuali prese durante lo sviluppo di ChromaRevive. Per ogni decisione sono riportati il contesto, le alternative considerate e la motivazione della scelta finale.

---

## Indice

1. [Scelta del problema: colorizzazione automatica](#1-scelta-del-problema-colorizzazione-automatica)
2. [Spazio colore LAB anziche' RGB](#2-spazio-colore-lab-anziche-rgb)
3. [ResNet-18 come backbone dell'encoder](#3-resnet-18-come-backbone-dellencoder)
4. [Architettura encoder-decoder con skip connections (U-Net)](#4-architettura-encoder-decoder-con-skip-connections-u-net)
5. [Transfer learning in due fasi](#5-transfer-learning-in-due-fasi)
6. [Addestramento avversariale (GAN) come variante](#6-addestramento-avversariale-gan-come-variante)
7. [Dataset COCO 2017 e MirFlickr](#7-dataset-coco-2017-e-mirflickr)
8. [Risoluzione fissa 256x256 per l'inference](#8-risoluzione-fissa-256x256-per-linference)
9. [FastAPI come framework backend](#9-fastapi-come-framework-backend)
10. [Frontend statico (HTML/CSS/JS vanilla)](#10-frontend-statico-htmlcssjs-vanilla)
11. [Design glassmorphic con tema scuro](#11-design-glassmorphic-con-tema-scuro)
12. [Slider di confronto prima/dopo](#12-slider-di-confronto-primadopo)
13. [Modelli multipli caricabili a runtime](#13-modelli-multipli-caricabili-a-runtime)
14. [CPU come target di inference predefinito](#14-cpu-come-target-di-inference-predefinito)
15. [Attivazione Tanh sull'output](#15-attivazione-tanh-sulloutput)
16. [Dropout nel decoder](#16-dropout-nel-decoder)
17. [Data augmentation con ColorJitter](#17-data-augmentation-con-colorjitter)
18. [Serving dei file statici direttamente da FastAPI](#18-serving-dei-file-statici-direttamente-da-fastapi)
19. [OpenCV per il resize dell'output](#19-opencv-per-il-resize-delloutput)
20. [Formato PNG per l'output](#20-formato-png-per-loutput)
21. [Hosting: da GitHub Pages a Render a solo locale](#21-hosting-da-github-pages-a-render-a-solo-locale)

---

## 1. Scelta del problema: colorizzazione automatica

**Contesto**: in fase di brainstorming iniziale, si cercava un progetto che dimostrasse l'applicazione concreta del deep learning nel dominio della computer vision, con un risultato visivamente immediato e comprensibile anche a un pubblico non tecnico.

**Alternative considerate**:
- Super-risoluzione di immagini (upscaling): risultati meno evidenti a occhio nudo, difficile da apprezzare senza confronto ravvicinato
- Rimozione di rumore (denoising): utile ma meno "spettacolare" come demo
- Trasferimento di stile (style transfer): risultati artistici ma meno applicabili a un caso d'uso pratico
- Segmentazione semantica: output tecnico (maschere), poco coinvolgente visivamente

**Decisione**: la colorizzazione di immagini in bianco e nero e' stata scelta perche' il confronto prima/dopo e' immediato e di forte impatto visivo, il problema e' self-supervised (non serve annotazione manuale), e ha un caso d'uso reale e tangibile (restauro di foto storiche). Inoltre, la colorizzazione permette di esplorare concetti avanzati come lo spazio colore LAB, il transfer learning e le GAN in un unico progetto.

---

## 2. Spazio colore LAB anziche' RGB

**Contesto**: la rete neurale deve predire i colori di un'immagine partendo dalla sua versione in scala di grigi. E' necessario definire in quale spazio colore operare.

**Alternative considerate**:
- **RGB**: il modello predice tutti e 3 i canali (R, G, B). Piu' semplice da implementare, ma il modello deve ricostruire anche la luminosita', rischiando di alterare la struttura dell'immagine originale. Inoltre i tre canali sono correlati tra loro, rendendo la predizione ridondante.
- **HSV**: separa tonalita', saturazione e valore, ma il canale H (hue) e' ciclico (0 e 360 sono lo stesso colore), il che complica la loss function e puo' generare artefatti.
- **YCbCr**: simile a LAB nella separazione luminanza/crominanza, ma meno percettivamente uniforme.

**Decisione**: lo spazio colore **LAB** e' stato scelto perche' separa nettamente la luminanza (canale L) dalla crominanza (canali a, b). Questo permette di preservare inalterato il canale L dell'immagine originale e chiedere alla rete di predire solo i 2 canali di colore. Il risultato e' che la struttura, i bordi e i dettagli dell'immagine non vengono mai alterati dal modello. Inoltre, lo spazio LAB e' percettivamente uniforme: differenze numeriche uguali tra valori corrispondono a differenze percettive uguali, il che rende la loss L1 piu' significativa dal punto di vista della qualita' visiva.

---

## 3. ResNet-18 come backbone dell'encoder

**Contesto**: l'encoder deve estrarre feature visive dall'immagine in scala di grigi. Si e' scelto di usare un backbone pre-addestrato su ImageNet per sfruttare il transfer learning.

**Alternative considerate**:
- **ResNet-34 / ResNet-50**: piu' profondi e con maggiore capacita' rappresentativa, ma significativamente piu' pesanti (circa 85 MB per ResNet-50 contro 44 MB per ResNet-18). Il tempo di inference su CPU sarebbe aumentato in modo rilevante.
- **VGG-16**: architettura classica per l'estrazione di feature, ma molto piu' pesante in memoria (~528 MB) e piu' lenta, senza un vantaggio significativo per questo task.
- **EfficientNet**: ottime performance con parametri ridotti, ma la struttura interna (blocchi MBConv con squeeze-and-excitation) rende piu' complessa l'implementazione delle skip connections rispetto ai blocchi residuali di ResNet.
- **Encoder custom da zero**: massima flessibilita', ma rinuncia al transfer learning e richiede quantita' di dati e tempo di training molto superiori.

**Decisione**: **ResNet-18** rappresenta il miglior compromesso tra capacita' di estrazione delle feature, dimensione del modello (~13 MB per i pesi finali) e velocita' di inference su CPU. Il backbone pre-addestrato su ImageNet fornisce gia' feature convoluzionali di alto livello (bordi, texture, forme) che sono direttamente utili per la colorizzazione, anche se ImageNet e' un dataset di classificazione e non di colorizzazione. La struttura a blocchi residuali di ResNet rende inoltre semplice l'estrazione di feature a scale multiple per le skip connections.

---

## 4. Architettura encoder-decoder con skip connections (U-Net)

**Contesto**: il modello deve trasformare un'immagine (canale L) in un'altra (canali ab), mantenendo la coerenza spaziale. Serve un'architettura adatta all'image-to-image translation.

**Alternative considerate**:
- **Encoder-decoder semplice (senza skip connections)**: l'informazione spaziale si perde attraversando il collo di bottiglia (bottleneck). I risultati tendono ad essere sfocati e con bordi imprecisi.
- **Architettura fully connected**: non praticabile per immagini a causa dell'enorme numero di parametri e della perdita di informazione spaziale.
- **PixelShuffle / Sub-pixel convolution**: efficiente per il super-resolution, ma meno adatto alla colorizzazione dove la coerenza semantica e' piu' importante della ricostruzione ad alta frequenza.

**Decisione**: l'architettura **U-Net** con skip connections e' stata adottata perche' permette di combinare feature semantiche profonde (dal bottleneck dell'encoder) con informazioni spaziali dettagliate (dagli strati iniziali dell'encoder). Le skip connections collegano direttamente gli strati dell'encoder con quelli corrispondenti del decoder tramite concatenazione, permettendo al decoder di accedere ai dettagli ad alta risoluzione senza doverli ricostruire dal bottleneck. Questo produce colorizzazioni piu' nitide e con bordi meglio definiti.

---

## 5. Transfer learning in due fasi

**Contesto**: usare un encoder pre-addestrato su ImageNet comporta il rischio di catastrophic forgetting se tutti i pesi vengono aggiornati fin dall'inizio con un learning rate alto.

**Alternative considerate**:
- **Fine-tuning diretto di tutta la rete**: piu' semplice, ma rischio concreto di distruggere le feature pre-apprese dell'encoder nelle prime epoche, quando il decoder non ha ancora imparato a usarle.
- **Encoder completamente congelato (solo decoder trainabile)**: sicuro contro il forgetting, ma limita la capacita' del sistema di adattare le feature dell'encoder al dominio specifico della colorizzazione.
- **Learning rate scheduling progressivo (warm-up)**: alternativa valida ma meno controllabile rispetto al congelamento esplicito.

**Decisione**: la strategia **in due fasi** (Fase 1: encoder congelato, solo decoder trainabile; Fase 2: fine-tuning completo con learning rate differenziato) e' stata scelta perche' bilancia sicurezza e adattamento. Nella prima fase il decoder impara a interpretare le feature dell'encoder senza alterarle. Nella seconda fase l'intero modello viene ottimizzato congiuntamente, ma l'encoder riceve un learning rate 10 volte piu' basso (`1e-5` vs `1e-4`) per preservare le conoscenze acquisite da ImageNet.

---

## 6. Addestramento avversariale (GAN) come variante

**Contesto**: i modelli addestrati con la sola loss L1 tendono a produrre colorizzazioni "sicure" ma sbiadite, perche' la media delle possibili colorizzazioni di un'area ambigua converge verso il grigio/seppia.

**Alternative considerate**:
- **Solo loss L1**: produce risultati stabili ma desaturati. L'effetto seppia e' frequente, specialmente su immagini con soggetti ambigui.
- **Loss percettiva (perceptual loss)**: confronta le feature estratte da una rete pre-addestrata (es. VGG) anziche' i pixel grezzi. Migliora la qualita' percepita ma non elimina del tutto il problema della desaturazione.
- **Loss L2 (MSE)**: penalizza di piu' gli errori grandi, ma tende a produrre risultati ancora piu' sfocati rispetto a L1.
- **GAN con Discriminatore**: aggiunge un avversario che "punisce" le colorizzazioni poco realistiche, spingendo il generatore verso colori piu' vividi e saturi.

**Decisione**: per il modello GAN si e' adottata una **loss combinata** (L1 + loss avversariale BCELoss, con fattore di scala 100 sulla L1). La componente avversariale spinge il modello a produrre colori piu' vibranti e meno "medi", mentre la L1 ancora la predizione alla verita' a terra per evitare artefatti cromatici. La scelta di mantenere anche i modelli non-GAN (coco30k, skip, final1888) e' stata voluta per offrire alternative piu' stabili quando la saturazione estrema del GAN non e' desiderata.

---

## 7. Dataset COCO 2017 e MirFlickr

**Contesto**: serviva un dataset ampio, vario e pubblicamente disponibile per addestrare i modelli.

**Alternative considerate**:
- **ImageNet**: enorme (oltre 1M di immagini), ma la distribuzione e' sbilanciata verso categorie specifiche di oggetti, e il download richiede autorizzazione accademica.
- **Places365**: focalizzato su scene e luoghi, ottimo per paesaggi ma carente per ritratti e oggetti.
- **Dataset proprietario**: massimo controllo sulla qualita' e composizione, ma richiede un lavoro di raccolta e curazione impraticabile nei tempi del progetto.
- **COCO 2017**: circa 200.000 immagini con ampia varieta' di scene, oggetti e contesti. Facilmente accessibile tramite `kagglehub`.
- **MirFlickr**: immagini "in the wild" da Flickr, non curate, rappresentative di input reali.

**Decisione**: **COCO 2017** e' stato scelto come dataset principale perche' offre la maggiore diversita' di scene in un singolo dataset facilmente accessibile. Sono stati estratti sottoinsiemi di dimensioni diverse (1.888 e 30.000 immagini) per confrontare l'impatto della quantita' di dati. **MirFlickr** e' stato usato come dataset complementare per testare la generalizzazione su immagini non curate, piu' simili a quelle che un utente reale sottometterebbe.

---

## 8. Risoluzione fissa 256x256 per l'inference

**Contesto**: le reti convoluzionali richiedono un input di dimensioni fisse (o comunque predefinite). E' necessario stabilire a quale risoluzione addestrare e fare inference.

**Alternative considerate**:
- **128x128**: piu' veloce nel training e nell'inference, ma la perdita di dettaglio cromatico e' eccessiva. I colori predetti risultano troppo uniformi e sfocati dopo l'upscaling alla risoluzione originale.
- **512x512**: maggiore dettaglio cromatico, ma il tempo di training quadruplica e l'utilizzo di memoria cresce significativamente. Su GPU con poca VRAM diventa impraticabile.
- **Risoluzione variabile (con padding)**: massima flessibilita', ma complica l'architettura e il DataLoader, e le skip connections richiedono dimensioni compatibili tra encoder e decoder.

**Decisione**: **256x256** e' stato scelto come compromesso tra qualita' e prestazioni. A questa risoluzione il modello ha abbastanza dettaglio per predire colori coerenti a livello semantico, il training e' ragionevolmente veloce anche su GPU di fascia media, e l'upscaling dei canali ab alla risoluzione originale tramite interpolazione bilineare produce risultati visivamente accettabili. L'informazione di luminosita' ad alta risoluzione viene preservata dal canale L originale, mitigando la perdita di dettaglio.

---

## 9. FastAPI come framework backend

**Contesto**: il sistema necessita di un server web che riceva immagini, esegua l'inference e restituisca il risultato. Il framework deve essere compatibile con PyTorch e supportare operazioni asincrone.

**Alternative considerate**:
- **Flask**: piu' conosciuto e con piu' risorse disponibili, ma sincrono per natura. Richiede estensioni aggiuntive per la validazione dei dati e la documentazione automatica dell'API.
- **Django**: troppo pesante per un progetto che espone solo 3-4 endpoint. L'ORM e il sistema di template non servono.
- **Gradio / Streamlit**: ottimi per prototipi rapidi con interfaccia automatica, ma limitano fortemente la personalizzazione del frontend e della user experience.
- **TorchServe**: pensato specificamente per il serving di modelli PyTorch, ma aggiunge complessita' infrastrutturale e la curva di apprendimento non giustifica i benefici per un progetto di questa scala.

**Decisione**: **FastAPI** e' stato scelto per la combinazione di velocita' di sviluppo, documentazione automatica (Swagger UI), validazione dei dati integrata (Pydantic), supporto nativo per operazioni async, e compatibilita' diretta con PyTorch. Inoltre, FastAPI puo' servire file statici, eliminando la necessita' di un web server separato.

---

## 10. Frontend statico (HTML/CSS/JS vanilla)

**Contesto**: il progetto necessita di un'interfaccia utente per caricare immagini, selezionare modelli e visualizzare i risultati.

**Alternative considerate**:
- **React / Vue / Angular**: framework moderni e potenti, ma introducono complessita' di build, dipendenze npm e una curva di apprendimento aggiuntiva. Per un'interfaccia con una singola pagina e poche interazioni, il overhead non e' giustificato.
- **Gradio / Streamlit**: generano automaticamente un'interfaccia, ma la personalizzazione estetica e' estremamente limitata. Non sarebbe stato possibile implementare il design glassmorphic, lo slider di confronto o le animazioni custom.
- **Server-side rendering (Jinja2 templates)**: possibile con FastAPI, ma mescola logica di presentazione con il backend e rende piu' difficile la manutenzione.

**Decisione**: **HTML, CSS e JavaScript vanilla** sono stati scelti per massima semplicita' di deployment (nessun passo di build), controllo completo sul design e sulle animazioni, e assenza di dipendenze frontend. Il frontend e' composto da soli 3 file (`index.html`, `style.css`, `script.js`) serviti direttamente dal backend FastAPI. Questo approccio mantiene il progetto leggero e facilmente comprensibile.

---

## 11. Design glassmorphic con tema scuro

**Contesto**: l'interfaccia deve essere visivamente accattivante e comunicare un senso di modernita' e professionalita'.

**Alternative considerate**:
- **Tema chiaro minimale**: semplice e pulito, ma meno d'impatto visivo e meno adatto a un'applicazione che lavora con immagini (lo sfondo chiaro compete visivamente con le foto).
- **Material Design**: riconoscibile e ben documentato, ma rischia di sembrare "generico" e poco originale.
- **Neumorphism**: esteticamente interessante ma con problemi di accessibilita' (contrasto insufficiente) e gia' percepito come un trend passato.

**Decisione**: il **tema scuro con effetti glassmorphic** e' stato scelto perche' lo sfondo scuro fa risaltare le immagini caricate e colorizzate (che sono il focus dell'applicazione), l'effetto vetro smerigliato (backdrop-filter: blur) dona un senso di profondita' e modernita', e l'accento lilla (`#d1b3ff`) con effetti glow crea un'atmosfera premium. I font scelti (Space Grotesk per i titoli, Inter per il corpo) sono moderni, leggibili e professionali.

---

## 12. Slider di confronto prima/dopo

**Contesto**: l'utente deve poter confrontare l'immagine originale in bianco e nero con la versione colorizzata.

**Alternative considerate**:
- **Due immagini affiancate**: semplice da implementare, ma su schermi piccoli le immagini diventano troppo piccole per apprezzare i dettagli. Inoltre non e' possibile confrontare pixel per pixel la stessa area.
- **Toggle on/off**: un pulsante che alterna tra le due versioni. Funzionale ma meno intuitivo e richiede di "ricordare" l'immagine precedente.
- **Overlay con opacita' regolabile**: permette di sovrapporre le due immagini, ma il risultato visivo e' confuso e poco leggibile.
- **Slider interattivo**: una barra trascinabile divide l'immagine in due meta', mostrando simultaneamente il prima e il dopo nella stessa area.

**Decisione**: lo **slider interattivo** e' stato scelto perche' permette un confronto diretto, pixel per pixel, nella stessa posizione. L'implementazione utilizza `clipPath` CSS per ritagliare l'immagine "prima" in base alla posizione del cursore. L'interazione funziona sia con mouse sia con touch (per dispositivi mobili). Questo tipo di comparazione e' lo standard de facto nelle applicazioni di editing fotografico.

---

## 13. Modelli multipli caricabili a runtime

**Contesto**: sono stati addestrati quattro modelli diversi, ciascuno con caratteristiche e punti di forza differenti. E' necessario decidere come renderli disponibili all'utente.

**Alternative considerate**:
- **Singolo modello predefinito**: piu' semplice, ma non permette all'utente di scegliere il risultato migliore per la propria immagine.
- **Istanze multiple del modello in memoria**: caricamento istantaneo nel cambio modello, ma consumo di RAM quadruplicato (~52 MB solo per i pesi). Impraticabile su macchine con risorse limitate.
- **Caricamento lazy con cache singola**: un solo modello alla volta in memoria, caricato dinamicamente quando l'utente lo seleziona.

**Decisione**: il **caricamento lazy con cache singola** e' stato implementato tramite la funzione `load_model_weights`. L'architettura del modello (`ColorizerResNet`) viene istanziata una sola volta; al cambio modello vengono semplicemente sostituiti i pesi (`state_dict`) nello stesso oggetto. Una variabile `current_loaded_model` tiene traccia del modello attualmente caricato per evitare ricaricamenti inutili. Questo approccio minimizza l'uso di RAM pur offrendo flessibilita' all'utente.

---

## 14. CPU come target di inference predefinito

**Contesto**: il progetto deve funzionare sulla maggior parte dei computer, inclusi quelli senza GPU dedicata.

**Alternative considerate**:
- **Solo GPU (CUDA)**: inference piu' veloce, ma esclude chi non ha una GPU NVIDIA compatibile con CUDA.
- **Rilevamento automatico GPU/CPU**: piu' flessibile, ma aggiunge complessita' al codice e alle dipendenze (la versione GPU di PyTorch pesa circa 2 GB in piu').

**Decisione**: il **target predefinito e' la CPU**, con `torch.device('cpu')` nel codice e `--extra-index-url https://download.pytorch.org/whl/cpu` nel `requirements.txt` per installare la versione leggera di PyTorch. Dato che l'architettura ResNet-18 e' relativamente piccola e l'inference avviene su una singola immagine alla volta, il tempo di elaborazione su CPU e' nell'ordine di 1-3 secondi, pienamente accettabile per un'applicazione interattiva. Chi dispone di una GPU puo' modificare il device e installare la versione CUDA di PyTorch.

---

## 15. Attivazione Tanh sull'output

**Contesto**: il layer finale del decoder deve produrre valori nel range appropriato per i canali ab dello spazio LAB.

**Alternative considerate**:
- **Nessuna attivazione**: l'output sarebbe illimitato, richiedendo un clipping successivo e rischiando valori instabili durante il training.
- **Sigmoid**: limita l'output al range [0, 1], ma i canali ab hanno range simmetrico centrato sullo zero (da -128 a 127). Richiederebbe una denormalizzazione asimmetrica.
- **Tanh**: produce output nel range [-1, 1], simmetrico e centrato sullo zero, che mappa naturalmente sui canali ab dopo moltiplicazione per 128.

**Decisione**: **Tanh** e' stata scelta perche' il suo range [-1, 1] corrisponde direttamente alla normalizzazione adottata per i canali ab (divisi per 128 durante il preprocessing). La simmetria dell'attivazione rispecchia la natura dei canali ab, dove valori negativi e positivi sono ugualmente significativi (es. -a = verde, +a = rosso). Questo rende la denormalizzazione una semplice moltiplicazione per 128.

---

## 16. Dropout nel decoder

**Contesto**: il decoder deve ricostruire i canali colore senza overfittare sui pattern specifici del training set.

**Alternative considerate**:
- **Nessun dropout**: rischio maggiore di overfitting, specialmente sui dataset piu' piccoli (1.888 immagini).
- **Dropout standard (nn.Dropout)**: progettato per layer fully connected, meno adatto ai layer convoluzionali dove i pixel vicini sono fortemente correlati.
- **Dropout2d (nn.Dropout2d)**: spegne interi feature map (canali) anziche' singoli neuroni, costringendo la rete a non fare affidamento su un singolo canale per la predizione. Piu' adatto alle reti convoluzionali.

**Decisione**: **Dropout2d con probabilita' 0.1** e' stato inserito in ogni `DecoderBlock`. Il valore basso (10%) e' stato scelto per introdurre regolarizzazione senza degradare significativamente la qualita' delle predizioni. Un valore piu' alto (es. 0.3-0.5) avrebbe introdotto troppo rumore nei canali colore, causando artefatti visivi nella colorizzazione finale.

---

## 17. Data augmentation con ColorJitter

**Contesto**: durante le prime fasi di training si e' osservato che i modelli tendevano a produrre colorizzazioni tendenti al seppia o al grigio-marrone, un fenomeno noto in letteratura.

**Alternative considerate**:
- **Nessuna augmentation**: training piu' veloce, ma il bias cromatico verso il seppia persiste.
- **Solo augmentation geometrica** (flip, rotazione, crop): diversifica le pose e le composizioni ma non affronta il problema della distribuzione cromatica.
- **ColorJitter** (luminosita', contrasto, saturazione): modifica casualmente le proprieta' cromatiche dell'immagine, costringendo il modello a imparare mappature colore piu' varie e robuste.
- **CutMix / MixUp**: tecniche avanzate di augmentation, ma piu' adatte alla classificazione che alla ricostruzione pixel-wise.

**Decisione**: **ColorJitter combinata con RandomHorizontalFlip** e' stata scelta perche' affronta direttamente il problema del bias cromatico. Variando casualmente luminosita', contrasto e saturazione delle immagini di training, il modello e' esposto a una distribuzione cromatica piu' ampia e impara a predire colori piu' diversificati. L'augmentation viene applicata solo al training set per non influenzare la valutazione.

---

## 18. Serving dei file statici direttamente da FastAPI

**Contesto**: il frontend e' composto da file statici (HTML, CSS, JS, immagini) che devono essere serviti all'utente.

**Alternative considerate**:
- **Web server dedicato (Nginx / Apache)**: piu' performante per file statici in produzione, ma aggiunge un componente infrastrutturale da configurare e mantenere.
- **Aprire index.html direttamente nel browser (file://)**: funziona, ma le richieste API dal frontend al backend sono cross-origin, richiedendo configurazione CORS e impedendo l'uso di alcune funzionalita' del browser.
- **Mounting StaticFiles in FastAPI**: FastAPI puo' servire file statici tramite `StaticFiles`. Non e' performante quanto Nginx, ma elimina un componente dell'architettura.

**Decisione**: il **mounting di StaticFiles in FastAPI** (`app.mount("/", StaticFiles(directory="."), name="static")`) e' stato scelto per semplicita' di deployment. Un singolo comando (`uvicorn main:app`) avvia sia il backend sia il frontend. Questo semplifica enormemente il setup per chi deve eseguire il progetto in locale. La configurazione CORS e' comunque presente per supportare anche l'apertura diretta di `index.html` come fallback.

---

## 19. OpenCV per il resize dell'output

**Contesto**: i canali ab predetti a 256x256 devono essere scalati alla risoluzione originale dell'immagine prima di essere combinati con il canale L.

**Alternative considerate**:
- **Pillow (PIL)**: gia' usato per il caricamento delle immagini, ma le opzioni di interpolazione su array numpy richiedono conversioni intermedie.
- **scikit-image (resize)**: offre interpolazione di alta qualita' ma e' significativamente piu' lenta di OpenCV per operazioni di resize.
- **PyTorch (F.interpolate)**: richiederebbe di mantenere i dati in formato tensore, aggiungendo una conversione in piu'.
- **OpenCV (cv2.resize)**: implementazione ottimizzata in C++, veloce e con supporto diretto per array numpy e interpolazione bilineare.

**Decisione**: **OpenCV con interpolazione bilineare** (`cv2.resize` con `INTER_LINEAR`) e' stato scelto per le prestazioni e la compatibilita' diretta con gli array numpy prodotti dal modello. La versione `opencv-python-headless` e' usata per evitare l'installazione delle dipendenze GUI (Qt/GTK) non necessarie per un'applicazione server-side.

---

## 20. Formato PNG per l'output

**Contesto**: l'immagine colorizzata deve essere restituita all'utente in un formato immagine standard.

**Alternative considerate**:
- **JPEG**: file piu' leggeri, ma la compressione lossy introduce artefatti visibili, specialmente nelle aree con transizioni di colore graduali. Per un'applicazione che vuole mostrare la qualita' della colorizzazione, gli artefatti JPEG sarebbero controproducenti.
- **WebP**: ottimo compromesso tra qualita' e dimensione, ma il supporto nei browser piu' datati non e' garantito e il download diretto puo' causare problemi di compatibilita' con i visualizzatori di immagini.
- **PNG**: compressione lossless, nessun artefatto, supporto universale.

**Decisione**: il formato **PNG** e' stato scelto perche' la compressione lossless preserva fedelmente ogni pixel della colorizzazione senza artefatti. In un contesto dove l'utente vuole valutare la qualita' dell'output, qualsiasi degradazione introdotta dal formato comprometterebbe la percezione del risultato. Il peso maggiore del file e' un compromesso accettabile data la natura interattiva dell'applicazione (una singola immagine alla volta).

---

## 21. Hosting: da GitHub Pages a Render a solo locale

**Contesto**: una volta completato il progetto, l'obiettivo era renderlo accessibile online senza costi, in modo che chiunque potesse provarlo senza dover installare nulla in locale.

**Alternative considerate e tentate**:

- **GitHub Pages**: la prima idea e' stata hostare il progetto su GitHub Pages, che offre hosting gratuito per siti statici. Tuttavia, GitHub Pages serve esclusivamente file statici (HTML, CSS, JS) e non supporta l'esecuzione di un backend Python. ChromaRevive richiede un server FastAPI per caricare i modelli PyTorch ed eseguire l'inference, quindi GitHub Pages non e' compatibile con l'architettura del progetto. Inoltre, GitHub Pages impone una struttura di directory specifica (il sito deve risiedere nella root o in una cartella `/docs`) che non si adatta alla struttura del nostro repository, dove backend e frontend coesistono nella stessa directory.

- **Render**: la seconda opzione valutata e' stata Render, una piattaforma cloud che supporta il deployment di applicazioni Python con piano gratuito. Il deployment iniziale e' andato a buon fine, ma il piano gratuito di Render offre risorse di memoria limitate (512 MB di RAM). I pesi dei modelli di ChromaRevive occupano complessivamente circa 130 MB su disco, ma una volta caricati in memoria con PyTorch, l'occupazione di RAM supera ampiamente il limite del piano gratuito, considerando anche il runtime di Python, le dipendenze (PyTorch, NumPy, scikit-image, OpenCV) e l'overhead di FastAPI. Il processo veniva terminato per Out of Memory (OOM) durante il caricamento dei pesi del modello.

- **Piano a pagamento su Render o altri cloud (Heroku, Railway, Fly.io)**: avrebbero risolto il problema di memoria, ma comportano costi ricorrenti non giustificabili per un progetto didattico.

**Decisione**: il progetto rimane **esclusivamente locale**. Dopo aver tentato sia GitHub Pages sia Render, si e' constatato che nessuna piattaforma di hosting gratuita e' in grado di soddisfare i requisiti del progetto: GitHub Pages non supporta backend Python, e i piani gratuiti dei servizi cloud (Render incluso) non offrono memoria sufficiente per caricare modelli PyTorch di queste dimensioni. Il deployment locale tramite `uvicorn main:app` resta il metodo di esecuzione previsto. In futuro, se si volesse rendere il progetto accessibile online, sarebbe necessario un piano cloud a pagamento con almeno 1 GB di RAM oppure una rearchitettura del sistema (ad esempio, un modello piu' leggero o l'uso di ONNX Runtime per ridurre l'occupazione di memoria).

---

## Riepilogo cronologico

| Fase | Decisioni principali |
|---|---|
| Ideazione | Scelta del problema (colorizzazione), definizione del caso d'uso |
| Progettazione modello | Spazio colore LAB, ResNet-18, U-Net, Tanh, Dropout2d |
| Training | Transfer learning in due fasi, GAN come variante, COCO + MirFlickr, ColorJitter, 256x256 |
| Backend | FastAPI, CPU predefinita, modelli caricabili a runtime, OpenCV, PNG |
| Frontend | HTML/CSS/JS vanilla, tema glassmorphic scuro, slider confronto |
| Deployment | File statici da FastAPI, singolo comando di avvio, hosting scartato (GitHub Pages e Render) |
