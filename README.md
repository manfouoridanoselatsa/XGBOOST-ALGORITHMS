# XGBOOST-FROM-SCRATCH-CPP
ici nous avons un code XGBOOST fait en c++

Pour Executer le code faire un **make** pour compiler et un **make run** en console pour executer
README

Pour lancer l'application il faudrait faire: 
   - make: pour compiler
   - make run: pour lancer le programme
   - make clean: pour supprimer le programme compile


Pour les hyper parametres de notre code en utilisant le make file nous avons:
	- <<nom fichier>>: ce sera le nom de l'executable
	- <<nombre-thread>>: le nombre de thread avec lequel on vas executer notre code
	- <<fichier-dataset>>: le nom du dataset ne contenant que les caracteristiques a utiliser pour la prediction
	- <<fichier-labels>>: le nom du fichier des labels ne contenant que les labels de nos observations 
	- <<train-test-percent>>: le pourcentage du jeux d'entrainnement pour le train test split(entre [0..1])
	- <<subsample_cols>>: le pourcentage de colonne(caracteristiques) a utiliser pour creer nos arbres XGBOOST(entre [0..1])
	- <<min_child_weight>>: la valeur minimale que doit posseder un noeud en terme de hessienne(en general a 1 car optionel)
	- <<depth>>: la profondeur maximale
	- <<min_leaf>>: le nombre minimum de noeud fils que doit posseder un noeud parent(en general a 1 car optionel) 
	- <<learning_rate>>: le taux d'apprentissage par arbre(en generale [0..1])
	- <<trees>>: le nombre d'arbres de notre modele
	- <<lambda>>: parametre de regularisation L2 (en generale [0..1])
	- <<gamma>>:parametre de regularisation L1(en generale [0..1])
	- <<epsilon>>:ici c'est le facteur de creation de blocks utilise pour l'algorithme approximate greedy algorithm(entre [0..1]) 
	- <<NanOuPas>>: ici c'est pour savoir si nous avons ou pas des donnees manquantes dans notre dataset(la valeur est 'oui' ou 'non')
	- <<choixAlgo>>: ici l'algorithme a utiliser
	- <<num_bins>>: ici on definis le nombre d'intervalle pour notre algorithme histogramme.
	

Pour les fichiers resultats nous avons:
	- outputPerLossPerTree.csv: Ici l'on a l'evolution de la perte de notre modele XGBOOST par arbre.
	- outputCsvFileLossPar.csv: Ici l'on a l'evolution de la perte de notre modele XGBOOST par arbre en fonction des differentes version parallel.
	- outputTreePerTime+i.csv: Ici l'on a le temps d'execution de notre modele XGBOOST au fur et a mesure de la creation de notre modele.
	- resultLog.txt: Ici l'on a les resultats de notre modele en terme de matrice de confusion, accuracy, precision, rappel.
	
	
	
Pour les datasets nous avons:
	- Nous avons le dataset PIMA dennome diabetes.csv, que nous avons subdivise en 2 fichiers:
	   	    * labels.csv : pour les labels des individus
	   	    * diabetes.csv: pour les caracterisques des individus
	   	    
	- Nous avons aussi WiDS2021 sous le nom cleanDiabetes_WiDS2021.csv, que nous subdivisons en 2 fichiers:
			* cleanDiabetes_data_WiDS2021.csv: pour les caracteristiques des individus
			* cleanDiabetes_labels_WiDS2021.csv: pour les labels des individus
