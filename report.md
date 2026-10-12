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
| 24 dB | 0 | 47 (17/125) | 60 (37/372) | 20 (200/1292) | 0 fichier détecté |

**Marge retenue : 9 dB.** Aucun ordre `clean` n'est coupé, quelle que soit la marge testée, donc `clean` ne départage pas. Une marge plus haute coupe beaucoup plus d'ordres dès snr30 (47/60 à 24 dB) car la voix faible passe sous le seuil. Une marge plus basse coupe moins mais le masque s'allume sur le bruit : l'erreur de début passe de 16 ms à 9 dB à 75 ms à 6 dB et 133 ms à 5 dB à snr30. 9 dB garde les erreurs de début proches de celles de `clean` tout en limitant les ordres coupés.

**Rung où la marge cesse de marcher : snr20** (critère : plus d'un tiers des ordres coupés) : 25/60 coupés, puis 49/60 à snr10. Le plancher de bruit, calculé sur les 10 % de trames les plus calmes, perd son sens quand il n'y a plus de vrai silence ; aucune marge fixe ne règle les rungs en dessous de snr10. Dans l'atelier, où les moteurs du robot tournent sur la moitié des fichiers (`data/manifest.csv`), ce VAD à seuil fixe n'est donc fiable que si le bruit reste faible.

Preuve : `results/vad-continuous-order.csv`.

## Step 3 — La baseline, sans modèle

**Réglages :** `MARGIN_DB = 9.0` (step 2), `SEGMENT_FLOOR_DB = 14`, `UNPARSED_BELOW = 0.45`, `MIN_SILENCE_SAMPLES = 512`. Voix : corpus synthétisé par chatterbox, intonation `order`.

| Run | Commande | `clean` (intent acc / WER) | `snr30` | `snr20` | `snr10` et en dessous |
|---|---|---|---|---|---|
| Parole continue | `python engine.py --input files --set all` | 0,0 / 0,954 | 0,067 / 0,767 | 0,05 / 0,791 | 0,0 |
| Mot par mot | `python engine.py --input files --set all --speech words` | 0,583 / 0,26 | 0,117 / 0,763 | 0,05 / 0,962 | 0,0 |

Preuves : `results/baseline-continuous-order.csv` et `results/baseline-words-order.csv`.

**Palier d'effondrement.**
- *Parole continue* : il n'y en a pas, l'option échoue déjà sur `clean` (0,0). Les 0,067 à `snr30` et 0,05 à `snr20` sont quelques ordres isolés, pas un fonctionnement.
- *Mot par mot* : la précision tombe de 0,583 (`clean`) à 0,117 à `snr30`, soit une perte de 80 % dès le premier rung bruité, puis 0,05 à `snr20` et 0,0 dès `snr10`. Le palier est entre `clean` et `snr30`.

**Pourquoi la parole continue échoue.** Sans pause entre les mots, `segments_of` ne trouve aucun silence à couper : l'ordre entier forme un seul segment, comparé à un seul mot du vocabulaire. Le robot ne reconnaît qu'un mot (WER proche de 1) et `interpret` renvoie `unparsed`.

**Ce que l'utilisateur doit faire pour que cette option marche.**
1. Parler mot par mot, avec une pause entre chaque mot : c'est exactement l'écart entre les deux runs (0,0 contre 0,583 sur `clean`).
2. Parler dans un endroit calme : l'option est presque nulle dès `snr30`, comme le VAD du step 2 (10/60 ordres coupés à `snr30`, 25/60 à `snr20`). L'atelier, où les moteurs du robot tournent sur la moitié des fichiers, est donc hors de portée.
3. Rester dans le vocabulaire fermé, avec « robot three » devant chaque ordre.

**Remarque sur `stop`.** Le compteur `stops` reste à 0/8 dans les deux runs. Le traitement de `stop` est étudié aux steps 5 et 6.

