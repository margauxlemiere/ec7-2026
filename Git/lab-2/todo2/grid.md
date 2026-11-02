# Grille multicritère — robot d'atelier (TurtleBot3, scénario §3)

Options comparées :
- **A** : baseline sans modèle (step 3), appariement de gabarits + grammaire fermée, exécutée sur le robot
- **B** : Whisper `tiny.en` pré-entraîné (step 4), `faster-whisper` + `normalise` + grammaire fermée

Chiffres issus de `todo2/report.md` : voix chatterbox, intonation `order` sauf mention contraire.

## Scores (−3 à +3, une phrase par score)

| Critère | A — baseline sans modèle | B — Whisper `tiny.en` |
|---|---|---|
| Latence | **+3** — latence médiane de 12 ms en continu et 20 ms en mots, donc le « ok » immédiat du §3.5 est tenu sans effort | **+1** — 600 ms médian sur PC, sous la limite de 1 s de Nielsen (le flux de pensée n'est pas rompu) mais au-dessus de 0,1 s (non instantané) ; avec le trajet réseau et la voix de sortie, l'accusé du §3.5 risque de dépasser la seconde|
| Coût total | **+3** — aucun modèle ni licence, quelques lignes de numpy tournent sur le matériel du robot | **+1** — modèle libre et gratuit (~75 Mo), mais il faut un poste d'atelier ou un CPU plus puissant pour tenir le temps réel |
| Énergie | **+3** — quelques FFT sur des segments de 16 ms, calcul négligeable pour un robot sur batterie | **+2** — le calcul tourne sur un poste d'atelier branché au secteur, donc la batterie du robot n'est pas touchée par la reconnaissance | 
| Mode de traitement | **+1** — chaque mot est traité dès que son segment se ferme, mais seulement si l'opérateur fait des pauses (0,0 en continu contre 0,583 en mots sur `clean`) | **−1** — l'ordre est transcrit en bloc après la fin de la parole, donc rien n'est reconnu avant la fin de la phrase |
| Déploiement / souveraineté | **+3** — tout tourne sur le robot, sans réseau : l'autonomie hors ligne citée au §1 est assurée | **+1** — le robot ne peut pas l'exécuter (RTF 1,90) ; le §1 autorise l'exécution en atelier, mais cela ajoute un lien réseau et une dépendance au poste |
| Robustesse | **−3** — 0,0 d'exactitude d'intention en continu dès `clean`, en mots 0,583 sur `clean` puis 0,117 à `snr30` : la parole « échoue dans le bruit » (§3.1) et ici dès le premier rung | **+2** — 0,967 sur `clean`, 0,917 à `snr10`, 0,750 à `snr5`, mais 0,100 à `snrm5` |
| Explicabilité | **+2** — chaque décision se lit dans un score de cosinus entre gabarits, puis dans la similarité avec la grammaire | **−1** — la transcription sort d'un réseau opaque ; seule la grammaire en aval reste traçable |
| Confidentialité | **+3** — l'audio n'est jamais transmis, il est comparé à des gabarits locaux | **+2** — l'audio reste dans l'atelier mais quitte le robot vers un poste. |
| Maintenabilité | **+1** — code simple, mais chaque nouveau mot demande des enregistrements et un réglage de `MARGIN_DB` (9 dB), `SEGMENT_FLOOR_DB` (14) et `UNPARSED_BELOW` (0,45) | **+2** — ajouter un ordre revient à éditer la grammaire, et changer de taille de modèle se fait avec une seule constante (`MODEL_SIZE`) |
| Contrôle humain (stop) | **−3** — le compteur `stops` reste à 0/8 sur tous les rungs : l'humain ne reprend pas la main par la voix | **0** — 8/8 sur `clean` pour les trois intonations, mais 3/8 à `snr5` en voix alarmée : insuffisant sans autre canal |

## Poids (0 à 5)

| Critère | Poids | Parce que |
|---|---|---|
| Latence | 4 | le dialogue du §3.5 commence par un accusé « ok » immédiat ; sans lui l'opérateur répète l'ordre |
| Coût total | 1 | le scénario ne donne aucun budget |
| Énergie | 2 | le TurtleBot3 est mobile donc sur batterie, mais le scénario ne chiffre aucune contrainte |
| Mode de traitement | 2 | utile pour accuser réception avant la fin de l'ordre, mais secondaire face à la robustesse |
| Déploiement / souveraineté | 4 | le §1 cite l'autonomie hors ligne et le lieu d'exécution (robot, service distant, atelier) comme critères d'ingénierie, et notre RTF de 1,90 impose ce choix |
| Robustesse | 5 | les moteurs du robot tournent dans l'atelier (moitié des fichiers) et le §3.1 dit que la parole échoue dans le bruit |
| Explicabilité | 3 | le §1 dit que les humains attendent des explications, et d'autres équipes subissent les actions du robot |
| Confidentialité | 2 | le micro est dans un atelier partagé avec les autres équipes, sans enjeu de données sensibles dans le scénario |
| Maintenabilité | 2 | l'équipe qui entretient le robot n'est pas spécialiste de la parole |
| Contrôle humain (stop) | 5 | les humains « reprennent le contrôle » (§1) et un stop non obéi dans des voies partagées est une collision |

## Scores pondérés

| Option | Total pondéré |
|---|---|
| A — baseline | +19 |
| **B — Whisper `tiny.en`** | **+26** |

## Knock-out

Critère éliminatoire retenu : **le robot doit s'arrêter quand on dit « stop », y compris avec une voix alarmée et dans le bruit de l'atelier.** C'est le pire échec possible : le §1 dit que les humains « reprennent le contrôle », les voies sont partagées avec les robots des autres équipes, et un stop non obéi est une collision. Aucun score élevé ailleurs ne le compense.

Seuil d'acceptation : 8/8 stops obéis sur `clean` et sur `snr5`, pour les trois intonations (`order`, `cheerful`, `alarmed`). Mesures issues de `results/baseline-continuous-all.csv` et `results/pretrained-continuous-all.csv` :

| Option | clean (order / cheerful / alarmed) | snr5 (order / cheerful / alarmed) | Verdict |
|---|---|---|---|
| A — baseline | 0/8, 0/8, 0/8 | 0/8, 0/8, 0/8 | **éliminée** : aucun stop obéi, même sur fichier propre (parole continue) |
| B — Whisper `tiny.en` | 8/8, 8/8, 8/8 | 7/8, 7/8, **3/8** | **éliminée aussi** pour la voix seule : un stop alarmé sur deux est perdu à `snr5` |

**Lecture.** Le critère élimine les deux options *si le stop dépend de la reconnaissance vocale seule*. B est la plus proche (8/8 sur `clean`), mais 3/8 en voix alarmée à `snr5` reste inacceptable : c'est justement la façon dont une personne crie « stop » dans un atelier bruyant. Le knock-out ne départage donc pas A et B sur la voix : il impose que **le stop ne passe pas par la chaîne de reconnaissance générale** (réponse au point à régler du §3.5 : non, il ne passe pas par la même chaîne que `make <product>`).

**Conséquence sur le choix.** Une fois le stop sorti de la chaîne, le choix entre A et B se fait sur le reste des critères. Là, B l'emporte au total pondéré (+26 contre +19) et surtout sur la robustesse (0,917 contre 0,0 à `snr10`), qui est le critère qui rend les autres ordres utilisables.

> **« Je retiens Whisper `tiny.en`, exécuté sur un poste d'atelier plutôt que sur le robot, et j'accepte une dépendance réseau et un délai supplémentaire afin d'obtenir une reconnaissance utilisable dans le bruit pour les ordres courants. Je n'accepte pas que le stop dépende de cette chaîne : il passe par un canal dédié (arrêt physique ou télécommande, plus un mot-clé « stop » sans mot d'activation et à seuil bas), car un stop non obéi est le seul échec que le système ne peut pas se permettre. »**

