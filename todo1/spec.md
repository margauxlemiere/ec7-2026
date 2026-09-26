## The specification

## 1
Qui lui parle : tout visiteur qui traverse le hall d'entrée et s'adresse directement au robot — pas de connexion, pas de relation préalable.  
Qui ne lui parle pas : les non-visiteurs (le personnel), car ils connaissent déjà le bâtiment, et les conversations non adressées (bruit de fond, discussions entre visiteurs, etc.) qui ne doivent déclencher aucune action de la part du robot.

## 2  
Positionné près de l'entrée d'un bâtiment public :   

Conversations non adressées alentour (discussions entre visiteurs, …)  
Plafond vitré → écho, réverbération et forte lumière naturelle  
Bruit de fond (machine à café à proximité) → bruit haute fréquence  

→ Entrée audio dégradée + problèmes potentiels de visibilité de l'écran (affichage)  

## 3
Entrée = voix   
Sortie = accusé de réception visuel montrant que le robot écoute (< 1 s) +  
             Questions sur la localisation d'une salle / les horaires du bureau = réponse audio courte + une carte/flèche  
             Appel à un agent humain = message séparé envoyé à un agent humain (non affiché au visiteur) + confirmation audio au visiteur  

## 4
- L'accusé de réception doit arriver en moins d'une seconde, sinon la personne se répète et deux voix se superposent  
- Le service informatique bloque tout trafic sortant vers un service non approuvé, et l'approbation prend un mois — ce qui élimine toute option dépendant d'un service externe non pré-approuvé au moment de l'installation  
- Ne pas réussir à appeler un humain quand cela est demandé est ce qui fait éteindre le robot — ce qui élimine toute option dont la reconnaissance de cette intention n'est pas fiable.  
