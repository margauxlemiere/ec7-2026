## Report



## Step 2 — Détection d'activité vocale

**Fonctions nommées.** `highpass_filter` (filtre passe-haut : la moyenne glissante donne la partie lente, on la soustrait ; on garde la voix au-dessus de 300 Hz, car l'atelier gronde sous 200 Hz) et `frame_energy` (puissance moyenne par trame de 10 ms).

**Principe.** Plancher de bruit = 10e percentile de l'énergie en dB. Une trame est de la voix si elle dépasse le plancher de `MARGIN_DB`. Les trous d'une trame à l'intérieur d'un mot sont comblés.

**Balayage de la marge** (`python engine.py --input files --set all --task vad`, parole continue, intonation `order`). Cellules : ordres coupés sur 60 (erreur de début / de fin en ms).

| Marge | clean | snr30 | snr20 | snr10 | snr5 |
|---|---|---|---|---|---|
| 5 dB | 0 | 2 (133/115) | 4 (132/204) | 16 (153/218) | 24 (167/266) |
| 6 dB | 0 | 7 (75/74) | 9 (82/139) | 26 (104/187) | 35 (111/262) |
| 9 dB | 0 | 10 (16/42) | 25 (27/74) | 49 (39/188) | 56 (34/324) |
| 12 dB | 0 | 12 (8/51) | 38 (13/95) | 59 (29/264) | 59 (46/569) |
| 18 dB | 0 | 28 (10/72) | 57 (25/199) | 60 (63/652) | 32 (163/1330) |
| 24 dB | 0 | 47 (17/125) | 60 (37/372) | 20 (200/1292) | 60 sans réponse |

**Marge retenue : 9 dB.** Aucun ordre `clean` n'est coupé, quelle que soit la marge testée, donc `clean` ne départage pas. Une marge plus haute coupe beaucoup plus d'ordres dès snr30 (47/60 à 24 dB) car la voix faible passe sous le seuil. Une marge plus basse coupe moins mais le masque s'allume sur le bruit : l'erreur de début passe de 16 ms à 9 dB à 75 ms à 6 dB et 133 ms à 5 dB à snr30. 9 dB garde les erreurs de début proches de celles de `clean` tout en limitant les ordres coupés.

**Rung où la marge cesse de marcher : snr20** (critère : plus d'un tiers des ordres coupés) : 25/60 coupés, puis 49/60 à snr10. Le plancher de bruit, calculé sur les 10 % de trames les plus calmes, perd son sens quand il n'y a plus de vrai silence ; aucune marge fixe ne règle les rungs en dessous de snr10. Dans l'atelier, où les moteurs du robot tournent sur la moitié des fichiers (`data/manifest.csv`), ce VAD à seuil fixe n'est donc fiable que si le bruit reste faible.

Preuve : `results/vad-continuous-order.csv`.



## Step 3 — La baseline, sans modèle

**Réglages :** `MARGIN_DB = 9.0` (step 2), `SEGMENT_FLOOR_DB = 14`, `UNPARSED_BELOW = 0.45`, `MIN_SILENCE_SAMPLES = 512`. Voix : corpus synthétisé par chatterbox, intonation `order`.

| Run | Commande | `clean` (intent acc / WER) | `snr30` | `snr20` | `snr10` et en dessous |
|---|---|---|---|---|---|
| Parole continue | `python engine.py --input files --set all` | 0,0 / 0,954 | 0,067 / 0,767 | 0,05 / 0,791 | 0,0 |
| Mot par mot | `python engine.py --input files --set all --speech words` | 0,583 / 0,26 | 0,117 / 0,763 | 0,05 / 0,962 | 0,0 |

**Palier d'effondrement.**
- *Parole continue* : il n'y en a pas, l'option échoue déjà sur `clean` (0,0). Les 0,067 à `snr30` et 0,05 à `snr20` sont quelques ordres isolés, pas un fonctionnement.
- *Mot par mot* : la précision tombe de 0,583 (`clean`) à 0,117 à `snr30`, soit une perte de 80 % dès le premier rung bruité, puis 0,05 à `snr20` et 0,0 dès `snr10`. Le palier est entre `clean` et `snr30`.

**Pourquoi la parole continue échoue.** Sans pause entre les mots, `segments_of` ne trouve aucun silence à couper : l'ordre entier forme un seul segment, comparé à un seul mot du vocabulaire. Le robot ne reconnaît qu'un mot (WER proche de 1) et `interpret` renvoie `unparsed`.

**Ce que l'utilisateur doit faire pour que cette option marche.**
1. Parler mot par mot, avec une pause entre chaque mot : c'est exactement l'écart entre les deux runs (0,0 contre 0,583 sur `clean`).
2. Parler dans un endroit calme : l'option est presque nulle dès `snr30`, comme le VAD du step 2 (10/60 ordres coupés à `snr30`, 25/60 à `snr20`). L'atelier, où les moteurs du robot tournent sur la moitié des fichiers, est donc hors de portée.
3. Rester dans le vocabulaire fermé, avec « robot three » devant chaque ordre.

**Remarque sur `stop`.** Le compteur `stops` reste à 0/8 dans les deux runs. Le traitement de `stop` est étudié aux steps 5 et 6.

Preuve : results/baseline-continuous-order.csv la results/baseline-words-order.csv



## Step 4 — Option pré-entraînée (Whisper)

### 1. Choix du modèle Whisper
* **Modèle sélectionné :** `tiny.en`
* **Justification :** 
  * Le modèle doit tourner sur le CPU restreint du robot. `tiny.en` est la version la plus légère (~75 Mo).
  * L'extension `.en` indique un modèle exclusivement entraîné sur la langue anglaise, offrant un meilleur taux d'erreur par mot (WER) qu'un modèle multilingue équivalent tout en restant très rapide.

### 2. Normalisation du texte
Whisper transcrit le texte sous forme littérale avec majuscules, chiffres et ponctuation (ex. *"Robot 3, carry blue pallet."*). Notre grammaire de l'étape 3 attend un format brut (ex. *"robot three carry blue pallet"*).

Sans la fonction `normalise()`, la reconnaissance échoue presque systématiquement :
* Les majuscules empêchent la détection du *wake word* (`"Robot"` ≠ `"robot"`).
* Les chiffres sous forme de digits ne correspondent pas au vocabulaire attendu (`"3"` ≠ `"three"`).
* La ponctuation altère la similarité de chaîne lors de l'interprétation.

Grâce aux règles de conversion (passage en minuscules, conversion des chiffres en lettres via le dictionnaire `DIGITS` et suppression de la ponctuation), le texte normalisé permet de retrouver une précision optimale.

### 3. Analyse des résultats et du Facteur Temps Réel (RTF)

### Résultats de l'exécution (`--asr pretrained`) :
--set all --asr pretrained, parole continue, voix chatterbox, intonation order

| Condition | n | Précision intention | WER | Latence médiane (ms) | Latence P90 (ms) | Stops obéis |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **clean** | 60 | **0.967** | 0.879 | 598.99 | 616.66 | 0/8 |
| **snr30** | 60 | **0.983** | 0.858 | 602.00 | 623.90 | 0/8 |
| **snr20** | 60 | **0.983** | 0.842 | 597.32 | 616.06 | 0/8 |
| **snr10** | 60 | **0.917** | 0.839 | 603.30 | 618.82 | 0/8 |
| **snr5** | 60 | **0.750** | 0.867 | 600.78 | 960.75 | 0/8 |
| **snr0** | 60 | **0.533** | 1.029 | 611.16 | 2377.81 | 0/8 |
| **snrm5** | 60 | **0.100** | 1.184 | 2126.51 | 2492.54 | 0/8 |

* **Facteur Temps Réel (RTF) mesuré sur PC :** `0.32`
* **Facteur Temps Réel (RTF) sur le CPU du robot :** `1.90` (valeur fixe du processeur embarqué)

### Analyse des performances :
1. **Résistance au bruit :** La précision des intentions reste excellente ($\ge 91,7\%$) jusqu'à un niveau de bruit de `snr10`. Elle commence à s'effondrer à partir de `snr5` ($75\%$) et devient inexploitable à `snrm5` ($10\%$).
2. **Gestion des arrêt d'urgence (`stops`) :** La métrique affiche `0/8` sur tous les tests. Cela s'explique par le fait que l'ordre `stop` requiert la vérification spécifique au niveau de `should_act` (Étape 5) pour être validé et exécuté par le moteur.

### Conclusion sur l'option exécutable par le robot :

Pour garantir un traitement en temps réel, le système doit impérativement avoir un **$\text{RTF} < 1.0$** (temps de calcul plus court que la durée du signal audio émis).

* **Sur le PC de test :** Le RTF est de `0.32` ($< 1.0$), ce qui signifie que la transcription tourne rapidement en local.
* **Sur le CPU du robot :** Le RTF est de `1.90` ($> 1.0$). Le processeur met 1,9 seconde pour traiter 1 seconde de parole, ce qui génère un retard cumulatif.

**Décision :** Le robot **ne peut pas exécuter Whisper en local** sur son processeur embarqué. Seule la **baseline de l'étape 3** (reconnaissance par templates) peut être retenue pour un fonctionnement autonome en temps réel sur le robot. Pour conserver l'option Whisper, les calculs devraient être déportés sur une infrastructure distante (serveur/GPU).

Preuve : results/pretrained-continuous-order.csv



## Step 5 — Garde-corps (`should_act`)

### 1. Implémentation du garde-corps
Dans `step5.py`, la fonction `should_act(intent, confidence, state)` filtre les actions selon trois règles fondamentales :
* **Ordres indéterminés (`unparsed`) :** Si la reconnaissance échoue (`intent == "unparsed"`), le robot refuse immédiatement d'agir (`False`).
* **Gestion prioritaire de l'arrêt d'urgence (`stop`) :** On fait le choix suivant : si l'intention détectée est `"stop"`, l'action est exécutée (`True`) quelle que soit la valeur du score de confiance. Il s'agit d'une règle de sécurité !
* **Autres ordres d'action :** L'action est exécutée (`True`) uniquement si le score de confiance dépasse le seuil dynamique `state["threshold"]`, sinon elle est refusée (`False`). 

---

### 2. Résultats du balayage des seuils (`sweep.csv`)

**Mesure** : `python engine.py --input files --set snr5 --asr pretrained --sweep 0.0:1.0:0.05` (parole continue, voix chatterbox, intonation `order`). Preuve : `sweep.csv`.

| Seuil | Correctes | Fausses | Refus | Stops refusés |
|---|---|---|---|---|
| 0,0 à 0,50 | 44 | 1 | 15 | 1 |
| 0,65 | 38 | 1 | 21 | 1 |
| 0,85 | 33 | 1 | 26 | 1 |
| 0,90 | 33 | 0 | 27 | 1 |
| 1,0 | 7 | 0 | 53 | 1 |

**Seuil retenu : 0,50.** Le sweep est plat de 0,0 à 0,50 puis perd des actions correctes dès 0,55 : 0,50 est le plus haut seuil sans perte. Une action fausse (1 sur 60) subsiste jusqu'à 0,85 ; la supprimer (0,90) coûterait 11 actions correctes de plus, soit 27 refus au lieu de 15. Une mauvaise action dans des voies partagées est plus grave qu'un refus, mais l'unique action fausse a une confiance supérieure à 0,85 : aucun seuil raisonnable ne l'arrête.

**Quand le robot refuse**, `answer("refuse", "")` renvoie « not understood » (§3.4) et le robot ne bouge pas ; l'opérateur répète l'ordre.

**Limite.** Un stop est refusé dans 1 cas sur 8 à tous les seuils, y compris 0,0 : il est classé `unparsed` par la reconnaissance, donc `should_act` n'est jamais appelé sur lui. Le seuil ne protège pas le stop ; c'est la raison du canal dédié (step 6).

Preuve : sweep.csv


## Step 6 — Gestion des intonations et de l'arrêt d'urgence (`stop`)

### 1. Résultats des exécutions

### Baseline  :
python engine.py --input files --set clean --set snr5 --intonation all

| Condition | n | Précision intention | WER | Latence médiane (ms) | Stops obéis |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **clean/order** | 60 | 0.000 | 0.954 | 12.88 | **0/8** |
| **clean/cheerful** | 60 | 0.000 | 0.942 | 11.86 | **0/8** |
| **clean/alarmed** | 60 | 0.000 | 0.897 | 12.34 | **0/8** |
| **snr5/order** | 60 | 0.000 | 0.981 | 16.51 | **0/8** |
| **snr5/cheerful** | 60 | 0.000 | 0.984 | 16.26 | **0/8** |
| **snr5/alarmed** | 60 | 0.000 | 1.046 | 16.19 | **0/8** |

### Option pré-entraînée Whisper :
python engine.py --input files --set clean --set snr5 --intonation all --asr pretrained

| Condition | n | Précision intention | WER | Latence médiane (ms) | Stops obéis |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **clean/order** | 60 | 0.967 | 0.879 | 600.91 | **8/8** |
| **clean/cheerful** | 60 | 0.983 | 0.888 | 595.89 | **8/8** |
| **clean/alarmed** | 60 | 0.967 | 0.908 | 608.96 | **8/8** |
| **snr5/order** | 60 | 0.717 | 0.872 | 603.55 | **7/8** |
| **snr5/cheerful** | 60 | 0.867 | 0.860 | 593.39 | **7/8** |
| **snr5/alarmed** | 60 | **0.533** | **1.006** | 618.20 | **3/8** |

Attention : Le WER du moteur est calculé sur le texte brut de Whisper (exemple : 'Robot 3, stop.' contre 'robot three stop' donne 1.0 alors que l'ordre est compris).

---

### 2. Analyse des performances selon l'intonation

1. **Échec de la Baseline sur parole continue :**
   * La baseline (step 3) échoue sur la parole continue (`0%` de précision) car elle repose sur la correspondance directe de gabarits enregistrés sur des mots isolés et calmes.

2. **Impact de l'intonation alarmée avec Whisper :**
   * En condition optimale (`clean`), Whisper détecte parfaitement les ordres d'arrêt d'urgence (**8/8**) quelle que soit l'intonation.
   * En présence de bruit de fond (`snr5`), l'intonation **alarmée** dégrade fortement la reconnaissance : la précision chute à **53,3%** (contre 71,7% en intonation neutre) et le nombre d'arrêts obéis s'effondre à **3/8**.
   * **Explication :** La voix alarmée modifie la hauteur tonale (voix plus aiguë/criée) et la dynamique spectrale, s'éloignant des données d'entraînement standard et augmentant l'erreur (WER > 1.0) sous le bruit.

---

### 3. Recommandations d'architecture pour le système d'arrêt (Scénario §3.5)

Les tests montrent que s'appuyer uniquement sur la chaîne ASR classique pour traiter un ordre `stop` vocal en situation de crise (voix déformée par l'alarme + bruit d'atelier) représente un **risque de sécurité majeur** (seuls 3 arrêts sur 8 exécutés sous `snr5/alarmed`).

Pour garantir un niveau de sécurité industrielle, le robot doit traiter l'ordre `stop` via des canaux complémentaires :

* **Canal physique / matériel dédié (Priorité 1) :** Ne pas faire reposer la sécurité critique uniquement sur la reconnaissance vocale. Un coup de poing d'arrêt d'urgence ou une télécommande sans fil portée par l'opérateur avec relais de sécurité doit rester le canal maître.
* **Détection d'énergie/mot-clé dédié (Keyword Spotting) sur canal séparé :** Implémenter un modèle ultra-léger tournant en tâche de fond sur un canal parallèle dédié uniquement au mot-clé `"stop"`, configuré avec un seuil de confiance très bas et entraîné spécifiquement sur des voix criées/alarmées.
* **Abaissement du seuil et contournement du Wake Word :** Supprimer l'exigence du *wake word* (`"robot three"`) pour le mot `"stop"` et accepter une détection directe à haute priorité sans passer par l'analyse grammaticale complète.

Preuve : results/baseline-continuous-all.csv , results/pretrained-continuous-all.csv



## Step 7 — Réponse du robot et auto-écoute

**Code.** `answer(action, reason)` renvoie `reason` (ou « not understood » s'il est vide) quand l'action est `refuse`, sinon la phrase de `SENTENCES` pour l'intention, ou « ok » par défaut. `speech_mask_while_speaking` garde, pendant que le robot parle, uniquement les trames dont l'énergie dépasse de `BARGE_IN_DB = 3.0` dB le niveau de sa voix (percentile `ROBOT_PERCENTILE = 90` de l'énergie des trames où il parle). Hors parole du robot, le masque de l'étape 2 s'applique.

**Mesures** (`--set clean`, parole continue, voix chatterbox pour l'entrée) :

| Voix | Commande | Accusé (« ok ») | Explication imprévue | Remarque |
|---|---|---|---|---|
| `recorded` | `--voice recorded` | 0,5 ms (mesuré) | 0,1 ms, **aucun son** | phrase absente de `data/phrases/` : le moteur refuse, `no recording for ...` |
| `onboard` | `--voice onboard` | 20,6 ms (mesuré) | 64,1 ms (mesuré) | synthèse locale sur ce CPU, backend `tone` sous Windows |
| `hosted` | `--voice hosted` | 643,1 ms (chiffre déclaré) | 692,4 ms (chiffre déclaré) | synthèse locale (~23 à 72 ms) plus 620 ms de délai réseau fixé (`HOSTED_DELAY_S`) ; aucun service réel |

**Note sur la machine.** Sous Windows, la première version de `tools/voices.py` cherchait la commande macOS `say` et plantait (`FileNotFoundError: [WinError 2]`). Cette erreur venait du code fourni, corrigé ensuite par l'enseignant (`git pull` du dépôt du cours). Les mesures ci-dessus sont faites avec la version corrigée, et `onboard` utilise le backend `tone`. Cette voix simple suffit pour le step 7, mais sa latence peut différer de celle d'un TTS embarqué plus riche.

**Lecture.** `recorded` et `onboard` sont sous les 100 ms de Nielsen (réaction perçue comme instantanée), ce qui tient l'accusé « immédiat » du §3.5. `hosted` reste sous 1 s mais n'est plus instantané, et il s'ajoute à la reconnaissance : avec Whisper (≈ 400 ms médian), l'accusé arriverait vers 1,0 s (addition de mes deux mesures), au-dessus de ce que demande le §3.5.

**Ce que `recorded` ne peut pas dire.** Il ne joue que les phrases fixes de `data/phrases/` (« ok », « no », « not understood », « possible », « impossible », « stop », « go forward », « turn left », « turn right »). La raison d'un refus, comme « conveyor 2 busy with robot 1, waiting 40 seconds », dépend de l'état de l'atelier : le moteur l'a confirmé (`no recording for ...`).

**Ce que je fais à la place.** Le robot dit « no » en voix enregistrée (0,5 ms, fiable), et la raison complète s'affiche sur son écran, qui « carries a full explanation » (§3.1). Les haut-parleurs étant de mauvaise qualité, l'écran sert aussi de canal de secours, et un voyant donne l'état sans latence. Une voix `onboard` ne servirait que pour les phrases dynamiques.

**Auto-écoute (`--voice onboard --loopback`).** Le détecteur se déclenche sur 0 trame de la voix du robot, et sur 146 trames du reste du signal : le robot ne prend plus sa propre voix pour un ordre. Limite : la voix testée est la voix simple `tone`, une voix plus riche pourrait fuiter davantage.

**L'humain peut-il encore interrompre le robot ?** Oui en principe : le micro n'est pas coupé, et une trame compte comme humaine si elle dépasse de plus de 3 dB la voix du robot. Le résultat du loopback ne mesure pas cette interruption : c'est le comportement du code. Un « stop » dit doucement pendant que le robot parle peut passer sous le seuil, d'où le canal d'arrêt dédié (step 6).