# Grille multicritère — robot d'accueil

Options comparées :
- **A** — grammaire fermée de vingt formulations, boîtier local (baseline, pas d'apprentissage)
- **B** — reconnaisseur pré-entraîné exécuté localement + règles sur le texte
- **C** — API vocale + langage hébergée (cloud)

## Scores (−3 à +3, une phrase de justification par score)

| Critère (poids) | A — grammaire fermée | B — ASR local + règles | C — API cloud |
|---|---|---|---|
| Latence | **+3** — l'appariement à une grammaire fixe sur un boîtier local est quasi instantané, bien sous la seconde | **+2** — la reconnaissance locale ajoute des dizaines à quelques centaines de ms, mais reste sous la seconde sur du matériel courant | **−2** — l'aller-retour réseau plus le traitement distant risquent de dépasser le seuil d'une seconde pour l'accusé de réception |
| Coût total | **+3** — un boîtier avec une grammaire fixe n'a quasi aucun coût de licence ni d'infrastructure | **+1** — le matériel local coûte plus cher à l'achat que A, mais pas de frais récurrents d'API | **−1** — frais récurrents par requête (~30 requêtes/heure aux heures de pointe) |
| Énergie | **+3** — un petit boîtier qui compare à une liste de vingt phrases consomme très peu | **+1** — la reconnaissance locale demande plus de calcul que A, mais rien n'est déporté vers un datacentre | **−1** — le calcul est déporté vers un service distant, avec l'infrastructure réseau et serveur que cela suppose |
| Mode de traitement | **+1** — une grammaire fermée peut matcher un préfixe de phrase dès qu'il arrive pour déclencher l'accusé, mais ne couvre que des formulations fixes | **+2** — un reconnaisseur local peut émettre des hypothèses partielles pour l'accusé, puis raisonner sur l'énoncé complet | **−2** — les API vocales hébergées renvoient généralement une réponse seulement une fois le segment audio complet, ce qui retarde l'accusé |
| Déploiement/souveraineté | **+3** — tout tourne sur le boîtier dans le hall, rien ne sort du bâtiment, aucune approbation nécessaire | **+2** — tout tourne localement, les données restent sur site, mais les mises à jour logicielles viennent de l'extérieur | **−3** — envoie l'audio vers un service externe non approuvé par l'IT, et l'approbation prend un mois, plus long que le délai d'installation |
| Robustesse | **−2** — une grammaire de vingt formulations échoue dès qu'un visiteur formule autrement, et la réverbération/le bruit de la machine à café l'éloignent encore des modèles fixes | **+1** — un reconnaisseur pré-entraîné tolère mieux le bruit et la variation de formulation que A, même si le plafond vitré et les pics de bruit le dégradent encore | **+2** — un modèle hébergé plus grand est généralement entraîné sur des données plus variées et plus bruitées, donc il tient mieux face à la réverbération et aux pics de bruit |
| Explicabilité | **+3** — chaque formulation correspond à une règle traçable, le superviseur peut pointer exactement la règle qui s'est déclenchée mardi à 17h | **+2** — les règles appliquées sur le texte reconnu gardent un chemin de décision traçable, même si la reconnaissance elle-même l'est moins | **−2** — le raisonnement du modèle hébergé est opaque, le superviseur ne peut pas montrer pourquoi le robot a dit ce qu'il a dit sans les journaux du fournisseur, inaccessibles à l'IT |
| Confidentialité| **+3** — les conversations non adressées dans le hall ne sont jamais transmises, elles ne matchent simplement aucune formulation | **+3** — l'audio reste sur l'appareil local | **−3** — chaque énoncé capté par le micro ouvert, y compris les conversations non adressées, est envoyé à un tiers externe |
| Maintenabilité | **+2** — une équipe IT de deux personnes peut ajouter une formulation à une liste sans compétence spécialisée | **+1** — régler un reconnaisseur local et ses règles demande plus de compétence et de temps que d'éditer une liste, mais sans dépendance externe à suivre | **−2** — une équipe de deux personnes doit suivre les évolutions d'une API externe, en plus du mois d'approbation à chaque changement |
| Contrôle humain | **+3** — les deux personnes de l'IT peuvent débrancher le boitier qui tourne localement, corriger une formulation ou une réponse directement dans la liste, sans passer par un tiers — rien ne s'exécute qui n'ait été explicitement pré-autorisé | **+1** — 	le boîtier reste local donc l'IT peut l'arrêter ou modifier la couche de règles elle-même, mais corriger une erreur venant du modèle de reconnaissance demande une expertise que deux personnes n'ont pas forcément, donc une erreur de reconnaissance peut persister entre deux interventions | **−2** — le comportement du modèle ne peut être corrigé que par le fournisseur, et toute modification de ce qui est approuvé repasse par le mois d'approbation de l'IT ; le seul levier rapide dont dispose l'équipe est de couper le trafic sortant pour tout arrêter d'un coup — un pouvoir d'arrêt existe, mais pas de reprise en main ni de correction en place |

## Poids (0 à 5)

| Critère | Poids | Parce que |
|---|---|---|
| Latence | 4 | l'accusé de réception doit arriver en moins d'une seconde, sinon la personne répète et deux voix se superposent |
| Coût total | 1 | aucun chiffre n'étant donné dans le cas, on considère donc que ce n'est pas un critère prioritaire |
| Énergie | 1 | le cas ne mentionne aucune contrainte énergétique ; le poids est arbitrairement choisi faible faute de fait qui le justifie |
| Mode de traitement (flux/complet) | 4 | l'exigence « montrer qu'il a entendu en moins d'une seconde » suppose un traitement incrémental, pas une attente de fin d'énoncé |
| Déploiement et souveraineté | 5 | le service informatique bloque tout trafic sortant vers un service non approuvé, et l'approbation prend un mois |
| Robustesse | 4 | hall bruyant et fréquenté (200 personnes/heure), plafond vitré qui fait résonner, machine à café qui tourne 30 s |
| Explicabilité | 4 | le superviseur de l'accueil doit pouvoir dire à un visiteur pourquoi le robot a dit, mardi dernier, que le bureau fermait à 17h |
| Confidentialité | 3 | le micro est ouvert dans un hall public : il capte des conversations non adressées au robot |
| Maintenabilité | 3 | le service informatique qui maintiendra le système n'a que deux personnes |
| Contrôle humain | 3 | ne pas réussir à appeler un humain pour quelqu'un qui le demande est ce qui fait éteindre le robot |

## Scores pondérés

| Option | Total pondéré |
|---|---|
| **A — grammaire fermée** | **+62** |
| B — ASR local + règles | +55 |
| C — API cloud | −51 |




## Recommandation

L'option A domine le score pondéré (+62 contre +55 pour B et −51 pour C), principalement parce qu'elle gagne sur les deux critères les plus lourds du cas : la latence et le déploiement/souveraineté. Elle tient aussi la promesse d'explicabilité exigée par le superviseur et ne transmet jamais les conversations non adressées captées par le micro ouvert. Sa faiblesse réelle est la robustesse : une grammaire de vingt formulations ne couvrira pas toutes les façons de demander la même chose. Le risque se compense en élargissant progressivement la liste de formulations à partir des échecs observés, sans changer d'architecture. Je retiens donc A comme choix par défaut, avec B comme option de repli si la couverture de vingt formulations s'avère trop étroite à l'usage.

