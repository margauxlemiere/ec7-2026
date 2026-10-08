# Grille multicritère — robot d'atelier (TurtleBot3, scénario §3)

Options comparées :
- **A** : baseline sans modèle (step 3), appariement de gabarits + grammaire fermée, exécutée sur le robot
- **B** : Whisper `tiny.en` pré-entraîné (step 4), `faster-whisper` + `normalise` + grammaire fermée

Chiffres issus de `todo2/report.md` : voix chatterbox, intonation `order` sauf mention contraire.

## Scores (−3 à +3, une phrase par score)

| Critère | A — baseline sans modèle | B — Whisper `tiny.en` |
|---|---|---|
| Latence | **+3** — latence médiane de 12 ms en continu et 20 ms en mots, donc le « ok » immédiat du §3.5 est tenu sans effort | **−2** — 600 ms médian sur PC, mais RTF de 1,90 sur le CPU du robot : 1,9 s de calcul par seconde de parole, l'accusé n'est plus immédiat |
| Coût total | **+3** — aucun modèle ni licence, quelques lignes de numpy tournent sur le matériel du robot | **+1** — modèle libre et gratuit (~75 Mo), mais il faut un poste d'atelier ou un CPU plus puissant pour tenir le temps réel |
| Énergie | **+3** — quelques FFT sur des segments de 16 ms, calcul négligeable pour un robot sur batterie | **−1** — réseau de neurones dont le RTF dépasse 1 sur le robot : CPU saturé, donc batterie sollicitée en continu |
| Mode de traitement | **+1** — chaque mot est traité dès que son segment se ferme, mais seulement si l'opérateur fait des pauses (0,0 en continu contre 0,583 en mots sur `clean`) | **−1** — l'ordre est transcrit en bloc après la fin de la parole, donc rien n'est reconnu avant la fin de la phrase |
| Déploiement / souveraineté | **+3** — tout tourne sur le robot, sans réseau : l'autonomie hors ligne citée au §1 est assurée | **−1** — le robot ne peut pas l'exécuter (RTF 1,90) ; le §1 autorise l'exécution en atelier, mais cela ajoute un lien réseau et une dépendance au poste |
| Robustesse | **−3** — 0,0 d'exactitude d'intention en continu dès `clean`, en mots 0,583 sur `clean` puis 0,117 à `snr30` : la parole « échoue dans le bruit » (§3.1) et ici dès le premier rung | **+2** — 0,967 sur `clean`, 0,917 à `snr10`, 0,750 à `snr5`, mais 0,100 à `snrm5` |
| Explicabilité | **+2** — chaque décision se lit dans un score de cosinus entre gabarits, puis dans la similarité avec la grammaire | **−1** — la transcription sort d'un réseau opaque ; seule la grammaire en aval reste traçable |
| Confidentialité | **+3** — l'audio n'est jamais transmis, il est comparé à des gabarits locaux | **+1** — l'audio reste dans l'atelier mais quitte le robot vers un poste, et le micro capte aussi les personnes des autres équipes |
| Maintenabilité | **+1** — code simple, mais chaque nouveau mot demande des enregistrements et un réglage de `MARGIN_DB` (9 dB), `SEGMENT_FLOOR_DB` (14) et `UNPARSED_BELOW` (0,45) | **+2** — ajouter un ordre revient à éditer la grammaire, et changer de taille de modèle se fait avec une seule constante (`MODEL_SIZE`) |
| Contrôle humain (stop) | **−2** — le compteur `stops` reste à 0/8 sur tous les rungs : l'humain ne reprend pas la main par la voix | **0** — 8/8 sur `clean` pour les trois intonations, mais 3/8 à `snr5` en voix alarmée : insuffisant sans autre canal |

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
| **A — baseline** | **+24** |
| B — Whisper `tiny.en` | −2 |

## Knock-out

Le total favorise A, mais il masque que A n'atteint presque jamais le but. Deux critères éliminatoires donnent des résultats opposés :

- **« Reconnaître au moins 90 % des ordres à `snr10` »** (bruit d'atelier) élimine A (0,0 en continu, 0,0 en mots) et garde B (0,917).
- **« Tourner en temps réel sur le CPU du robot (RTF < 1) »** élimine B (RTF 1,90) et garde A.

Aucune option ne passe les deux. La somme pondérée récompense A parce qu'elle est légère et locale, alors qu'elle ne reconnaît presque rien dans le bruit. Je ne retiens donc pas A sur ce seul total.

## Recommandation

> **« Je retiens Whisper `tiny.en`, exécuté sur un poste d'atelier plutôt que sur le robot (RTF 1,90 sur le CPU du robot), et j'accepte une dépendance réseau et un délai supplémentaire afin d'obtenir une reconnaissance utilisable dans le bruit de l'atelier. Le stop ne dépend pas de la chaîne de reconnaissance seule : il a son propre canal (mot-clé dédié sans mot d'activation, plus un arrêt physique). »**

Le stop pèse 5 et aucune option ne le tient sous bruit : 0/8 pour A, 3/8 à `snr5` en voix alarmée pour B. C'est la réponse au point à régler du §3.5 (« whether stop goes through the same chain as make <product> ») : non, il ne passe pas par la même chaîne.