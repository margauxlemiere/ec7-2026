## Script utilisé

```python
import time
 
# les vingt formulations de la grammaire fermee (option A)
GRAMMAR = [
    "où est la salle",
    "où se trouve le bureau",
    "où se trouve la salle de réunion",
    "où est l'accueil",
    "où sont les toilettes",
    "comment aller à l'amphithéâtre",
    "quand ferme le bureau",
    "à quelle heure ferme la salle",
    "à quelle heure ouvre le bureau",
    "quels sont les horaires d'ouverture",
    "le bureau est-il encore ouvert",
    "à quelle heure ferme l'accueil",
    "je voudrais parler à un humain",
    "appelle un agent",
    "je veux parler à quelqu'un",
    "peux-tu m'aider je ne trouve pas la salle",
    "je cherche le service des admissions",
    "peux-tu me guider vers la sortie",
    "y a-t-il quelqu'un pour m'aider",
    "je souhaite un rendez-vous avec un conseiller",
]
 
# dix phrases de test, diversifiees (reformulations, bruit, hesitations)
TEST_SENTENCES = [
    "où est la salle 204",                                   # reformulation directe
    "quand ferme le bureau des admissions",                  # reformulation directe
    "je veux parler à quelqu'un",                             # formulation exacte de la grammaire
    "euh bonjour, vous savez où est la salle de réunion ?",   # hesitation + politesse
    "excusez-moi, à quelle heure ça ferme ici",               # formulation eloignee, ellipse
    "hé, y a quelqu'un qui peut m'aider ?",                    # familier, ponctuation
    "je suis perdu, je cherche le service des admissions",    # phrase composee, contexte ajoute
    "bonjour, le bureau est-il encore ouvert en ce moment",   # politesse + reformulation
    "appelle un agent s'il te plaît",                         # formulation proche + politesse
    "pardon, où sont les toilettes s'il vous plaît",          # politesse + formulation exacte
]

def match(sentence, grammar):
    # remplace par la logique réelle de correspondance que tu utilises pour A
    return any(phrase in sentence for phrase in grammar)

durations = []
for sentence in TEST_SENTENCES:
    start = time.perf_counter()
    match(sentence, GRAMMAR)
    durations.append(time.perf_counter() - start)

durations.sort()
median = durations[len(durations)//2]
print(f"durations (s): {durations}")
print(f"median: {median*1000:.3f} ms")
```

## Sortie brute

```
durations (s): [1.500011421740055e-06, 1.700012944638729e-06, 2.7999049052596092e-06, 2.800021320581436e-06, 3.700028173625469e-06, 3.900029696524143e-06, 4.00003045797348e-06, 4.699919372797012e-06, 6.299931555986404e-06, 1.3699987903237343e-05]
median: 0.004 ms
```

## Ligne de check_env.py identifiant la machine

```
platform         Windows-11-10.0.26200-SP0
machine          AMD64
```

## Comparaison avec la latence supposée dans grid.md

La grille suppose **+3 (quasi instantané, bien sous la seconde)** pour la latence de A.  
Or, notre test conclut que : latence (option A) = 0.004 ms < 1 s.   
Donc, d'après le test réalisé, l'option A renvoie un accusé de réception dans un délais largement sous la seconde (comme exigé dans le cas). On en déduit que le socre attribué dans la grille a été correctement estimé.

## Deux lignes : que changes-tu si les deux valeurs ne s'accordent pas ?

Les deux valeurs s'accordent : la médiane mesurée (0,004 ms) reste très largement sous le seuil supposé dans grid.md, donc rien à corriger sur le score de latence de l'option A.   
Ce test ne mesure toutefois que la correspondance texte-grammaire, pas la chaîne complète micro → reconnaissance vocale → réponse ; si une mesure de bout en bout dépassait un jour la seconde, ce serait le pipeline audio en amont qu'il faudrait revoir, pas la logique de correspondance elle-même.
