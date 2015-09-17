#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division				# Avant toute instruction

from math import sqrt
from collections import defaultdict

from pprint import pprint

print
print "Name of running program: __name__     = "	,__name__
print "Running file:           __file__      = "	,__file__
print "Does it resides in a:   __package__   = "	,__package__


__doc__ = """
Reprise by MM
	Today is 9/17/15. It is now 4:13 PM"""


def sources():
	"""	
def sources():
=================

# Ref:

1/
https://github.com/kilimandjaro2/PCI/blob/c2eefe23794be646baac3200027b2aca766df128/chapter2/recommendations.py

2/
rodelrod/collective-intelligence-GitHub

	
3/
https://github.com/PyBeaner/ProgrammingCollectiveIntelligence/tree/master/Chapter2
"""

	t = """C'est la Liste des  3 sources consultees"""
	return t
	
	
def the_aim():
	"""
def the_aim():
==============
	
	Playing with recommendation systems
	
	Code supporting Chapter 2 of 
	"Programming Collective Intelligence",
	Author: Toby SEGARAN
	                    
	                    First Edition
	"""
	pass


class ArgumentError(Exception):
	pass

# A dictionary of movie critics and their ratings 
# of a small set of movies:

critics = {
	'Lisa Rose': {
		'Lady in the Water': 2.5, 'Snakes on a Plane': 3.5, 
		'Just My Luck': 3.0, 'Superman Returns': 3.5, 
		'You, Me and Dupree': 2.5, 'The Night Listener': 3.0},
	'Gene Seymour': {
		'Lady in the Water': 3.0, 'Snakes on a Plane': 3.5, 
		'Just My Luck': 1.5, 'Superman Returns': 5.0, 
		'The Night Listener': 3.0, 'You, Me and Dupree': 3.5},
	'Michael Phillips': {
		'Lady in the Water': 2.5, 'Snakes on a Plane': 3.0, 
		'Superman Returns': 3.5, 'The Night Listener': 4.0}, 
	'Claudia Puig': {
		'Snakes on a Plane': 3.5, 'Just My Luck': 3.0, 
		'The Night Listener': 4.5, 'Superman Returns': 4.0, 
		'You, Me and Dupree': 2.5},
	'Mick LaSalle': {
		'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
		'Just My Luck': 2.0, 'Superman Returns': 3.0, 
		'The Night Listener': 3.0, 'You, Me and Dupree': 2.0},
	'Jack Matthews': {
		'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0, 
		'The Night Listener': 3.0, 'Superman Returns': 5.0, 
		'You, Me and Dupree': 3.5},
	'Toby': {
		'Snakes on a Plane':4.5, 'You, Me and Dupree':1.0,
		'Superman Returns':4.0}
}




def sim_distance(prefs, p1, p2):
	'''
	sim_distance(prefs, p1, p2):
	
	Returns a distance-based similarity score for person1 and person2.
	'''
# 	print p1,p2
# 	pprint(prefs)
	# Get the list of si (shared_items)
	si = {}
	for item in prefs[p1]:
		if item in prefs[p2]:
			si[item] = 1
			
# 	print "From sim_Distance	si :", si
	
	# If they have no ratings in common, return 0.
# 	if len(si) == 0: return 0
	if len(si) == 0: return 0.
	
	# Add up the squares of all the differences
	sum_of_squares = sum([pow(prefs[p1][item] - prefs[p2][item], 2) for item in
						 prefs[p1] if item in prefs[p2]])
						 
# 	return 1 / (1 + sum_of_squares)							# BUG
	return 1 / (1 + sqrt(sum_of_squares))



#
#			Autre version  Debut
#

def sim_distance0(prefs, p1, p2):
	"""sim_distance0(prefs, p1, p2):"""

	shared_movies = get_shared_movies(prefs, p1, p2)
	if len(shared_movies) == 0: return 0.
	else:
		sum_squares = sum([ pow(prefs[p1][film]-prefs[p2][film], 2) 
							for film in prefs[p1].keys() 
							if film in prefs[p2].keys() 
						  ])
		return 1./(1. + sqrt(sum_squares))	# Enfin un calcul correct dans GitHub


def get_shared_movies(prefs, p1, p2):
	return [m for m in prefs[p1].keys() if m in prefs[p2].keys()] 


def get_rating_across_movie_list(prefs, person, movie_list):
	return [prefs[person][m] for m in movie_list] 


def average(seq):
	return sum(seq)/len(seq)


def stdev(seq):
	"""Ecart-type (Standard Deviation)"""
	N = len(seq)
	avg = average(seq)
	return sqrt(sum([(s-avg)**2 for s in seq])/N)


def stdev2(seq):
	"""Valable seulement si avg = 0."""
	N = len(seq)
	return sqrt(sum([s**2 for s in seq])/N - (sum(seq)/N)**2)


def cov(seq1, seq2):
	if len(seq1) != len(seq2):
		raise ArgumentError("Both argument sequences must have the same length.")
	N = len(seq1)
	eXY = sum([seq1[i]*seq2[i] for i in range(0,N)])/N
	eX = sum([s for s in seq1])/N
	eY = sum([s for s in seq2])/N
	return eXY - (eX * eY)


def pearson_corr(seq1, seq2):
	if len(seq1) != len(seq2):
		return 0	# this should actually raise an exception
	return cov(seq1, seq2)/(stdev2(seq1)*stdev2(seq2))


def sim_pearson0(prefs, p1, p2):
	shared_movies = get_shared_movies(prefs, p1, p2)
	if len(shared_movies) == 0:
		return 0
	else:
		ratings = [get_rating_across_movie_list(prefs, person, shared_movies)
				   for person in [p1, p2]]
		return pearson_corr(ratings[0], ratings[1])
#
#			Autre version  Fin
#


def sim_pearson(prefs, p1, p2):
	'''
	Returns the Pearson correlation coefficient for 
	p1 (person1) and p2 (person2).
	'''

	# Get the list of mutually rated items
	si = {}
	for item in prefs[p1]:
		if item in prefs[p2]:
			si[item] = 1


# 	print "From sim_Pearson	si :", si
	
	# If they are no ratings in common, return 0
	n = len(si)
# 	if len(si) == 0: return 0
	if n	   == 0: return 0


	# Sum calculations
# 	n = len(si)				# BUG sur n (cas de la divion Entiere)
	n = float(n)	
	
	# Sums of all the preferences
	sum1 = sum([prefs[p1][it] for it in si])
	sum2 = sum([prefs[p2][it] for it in si])
	
	# Sums of the squares
	sum1Sq = sum([pow(prefs[p1][it], 2) for it in si])
	sum2Sq = sum([pow(prefs[p2][it], 2) for it in si])
	
	# Sum of the products
	pSum = sum([prefs[p1][it] * prefs[p2][it] for it in si])
	
	# Calculate r (Pearson score)
	num = pSum - sum1 * sum2 / n
	den = sqrt((sum1Sq - pow(sum1, 2) / n) * (sum2Sq - pow(sum2, 2) / n))
	
	if den == 0: return 0
	r = num / den
	return r


def sim_tanimoto(prefs,p1,p2):
	"""
	# Calculate Tanimoto coefficient.
	#
	# Ref:
	# Kountzr/programming-collective-intelligence-code 
	# 	forked from cataska/programming-collective-intelligence-code
	"""
	ci = set([])
	
	for item in prefs[p1]:
		if item in prefs[p2]:
			ci.add(item)


# 	print "From sim_tanimoto	ci :", ci
	
	if len(ci) == 0: return 0

	# use only items in both sets for a and b
	a = sum([pow(prefs[p1][k], 2) for k in ci])
	b = sum([pow(prefs[p2][k], 2) for k in ci])

	# use the full sets for a and b
	#a = sum([s*s for s in prefs[person1].values()])
	#b = sum([s*s for s in prefs[person2].values()])
	
	c = sum([prefs[p1][k] * prefs[p2][k] for k in ci])
	
	den = a + b - c
	
	if den < 1.0e-7 : return 1.0e-7	# Attention aux comparaisons avec zero

# 	return c/(a + b - c)
	return c/den


def topMatches(prefs, person, n=5, similarity=sim_pearson):
	"""
# 	Returns the best matches for person from the prefs dictionary.
# 
# 	n ----------- Number of results 
# 	similarity -- similarity function (pearson, euclidean, etc.)
	"""
	all_matches = [(similarity(prefs, person, other), other) 
				   for other in prefs.keys()
				   if person != other]
	all_matches.sort()
	all_matches.reverse()
	return all_matches[0:n]


def getRecommendations(prefs,person,similarity=sim_pearson):
	"""
# 	Gets recommendations for a person.
# 	
# 	Uses a weighted average of every other user's rankings	
	"""
	weighted_similarities = dict((
			(other, similarity(prefs, person, other)) 
			for other in prefs.keys() if other != person))
			
	# Eliminate critics with negative correlation (I'm not sure why
	# this is a good idea)
	for critic, sim in weighted_similarities.items():
		if sim <= 0: del weighted_similarities[critic]
			
	sum_ratings = defaultdict(int)	  # int() initializes to 0
	sum_weights = defaultdict(int)
	
	for other, weight in weighted_similarities.items():
		for movie, rating in prefs[other].items():
			sum_ratings[movie] += rating * weight
			sum_weights[movie] += weight
			
	recommendations = [(sum_ratings[movie]/sum_weights[movie], movie)
					   for movie in sum_ratings.keys()
					   if movie not in prefs[person].keys()]
					   
	recommendations.sort()
	recommendations.reverse()
	return recommendations


def getRecommendedItemsBasedOnUser(prefs,peopleMatch,user):
	similarUsers = peopleMatch[user]
	scores = {}

	totalSim = {}
	for similarity,other in similarUsers:
		for item,rating in prefs[other].items():
			if item in prefs[user]:continue

			scores.setdefault(item,0)
			scores[item]+=similarity*rating
			totalSim.setdefault(item,0)
			totalSim[item] += similarity

	ranking = [(score/totalSim[item],item) for item,score in scores.items()]
	ranking.sort()
	ranking.reverse()
	return ranking
	

def transformPrefs(prefs):
	new_prefs = defaultdict(dict)
	for person, movies in prefs.items():
		for movie, rating in movies.items():
			new_prefs[movie][person] = rating
	return new_prefs


def printMovies ():							# ADDED
	print """
# 	Scores (from 1 to 5) given to Movies by each Critic:
# 	(critics Dictionnary)
# 	========================================================="""
	pprint (critics)


def calculateSimilarItems(prefs, n=10):
	'''
# 	Create a dictionary of items showing 
#	which other items they are most similar to.
	'''

	result = {}
	# Invert the preference matrix to be item-centric
	itemPrefs = transformPrefs(prefs)
	
	c = 0
	for item in itemPrefs:
	
		# Status updates for large datasets
		c += 1
		if c % 100 == 0:
			print '%d / %d' % (c, len(itemPrefs))
			
		# Find the most similar items to this one
		scores = topMatches(itemPrefs, item, n=n, similarity=sim_distance)
		
		result[item] = scores
	return result


def printSimilarItems ():					# ADDED
	itemsim = calculateSimilarItems(critics)
	print """
	Similarities betwen Movies:
	==========================="""
	pprint (itemsim)
	

def getRecommendedItems(prefs, itemMatch, user):
	"""
	"""
	userRatings = prefs[user]
	scores = {}
	totalSim = {}
	
	# Loop over items rated by this user
	for (item, rating) in userRatings.items():
	
		# Loop over items similar to this one
		for (similarity, item2) in itemMatch[item]:
		
			# Ignore if this user has already rated this item
			if item2 in userRatings: continue
			
			# Weighted sum of rating times similarity
			scores.setdefault(item2, 0)
			scores[item2] += similarity * rating
			
			# Sum of all the similarities
			totalSim.setdefault(item2, 0)
			totalSim[item2] += similarity
			
	# Divide each total score by total weighting to get an average
	rankings = [(score / totalSim[item], item) for (item, score) in
				scores.items()]
				
	# Return the rankings from highest to lowest
	rankings.sort()
	rankings.reverse()
	return rankings

	
	
def test_tanimoto():		# 2015-05-06 Function a revoir
	prefs = {'p1':
				{'a' : 1, 'b' : 1},
			 'p2':
				{'a' : 1, 'b' : 0},
			 'p3':
				{'a' : 0, 'b' : 1},
			 'p4':
				{'a' : 0, 'b' : 0}}
	
	p1p1 = sim_tanimoto(prefs, 'p1', 'p1')
	p4p4 = sim_tanimoto(prefs, 'p4', 'p4')
	p2p3 = sim_tanimoto(prefs, 'p2', 'p3')
	p1p3 = sim_tanimoto(prefs, 'p1', 'p3')
	
	# 	assert_equal(p1p1, 1.0)
	# 	assert_equal(p4p4, 1.0)		#BUG 0.0   A VOIR
	# 	assert_equal(p2p3, 0.0)
	# 	assert_equal(p1p3, 0.5)
	

	print "p1p1 :", p1p1 ,"Coeff attendu = 1"
	print "p4p4 :", p4p4 ,"Coeff attendu = 1"
	print "p2p3 :", p2p3 ,"Coeff attendu = 0"
	print "p1p3 :", p1p3 ,"Coeff attendu = 0.5"


def calculateSimilarUsers(prefs,n=10):
	result = {}
	# user based key
	c = 0
	for person in prefs:
		c+=1
		if c%100==0:print("%d / %d"%(c,len(prefs)))
		scores = topMatches(prefs,person,n=n,similarity=sim_distance)
		result[person] = scores
	return result




def loadMovieLens(path='data/data/movielens'):
	"""
	load_MovieLens(path='/data/movielens')
	"""
	# Get movie titles
	movies = {}
	for line in open(path + '/u.item'):
		(id, title) = line.split('|')[0:2]
		movies[id] = title
		
	# Load data
	prefs = {}
	for line in open(path + '/u.data'):
		(user, movieid, rating, ts) = line.split('\t')
		prefs.setdefault(user, {})
		prefs[user][movies[movieid]] = float(rating)
		
	return prefs


def Recalage_sur_les_Exemples_du_Livre ():
	"""
	# But: Utiliser les Exemples de PCI de Toby
	# 	Recherche - dans Git Hub - s'il existe un developeur avance qui
	# 	a le meme But
	# 	Et cela passe par:
	# 	
	# 		- Reprise des Exemples du Livre
	# 		- Recalage des Resultats
	# 


	#######################################
	# Programming Collective Intelligence #
	#######################################
	
	Examples from [Programming Collective Intelligence](http://oreilly.com/catalog/9780596529321/)
	
 
	Code was "tidied" up using [PythonTidy](http://pypi.python.org/pypi/PythonTidy/1.11) with the following settings:
	"""

# 	print; printMovies0 ()
# 	
# 	print; printSimilarItems0 ()


	print; print "test_tanimoto"; test_tanimoto()
	
	print; printMovies ()
	
	p1 = "Lisa Rose" ; p2= "Gene Seymour"
	print
	print "Calcul du Coefficient de Similitude pour diverses Mesures"
	print "Entre: ",p1,"et: ", p2
	
	print "sim_distance : " , sim_distance(critics, p1, p2)
	print "sim_pearson : " ,  sim_pearson(critics, p1, p2)
	print "sim_tanimoto : " , sim_tanimoto(critics, p1, p2)
	
	#Resultats:
	# 	Lisa Rose Gene Seymour
	# 	sim_distance :  From sim_distance	si : {'Lady in the Water': 1, 'Snakes on a Plane': 1, 'Just My Luck': 1, 'Superman Returns': 1, 'You, Me and Dupree': 1, 'The Night Listener': 1}
	# 	0.294298055086
	
	# 	sim_pearson :  From sim_pearson	si : {'Lady in the Water': 1, 'Snakes on a Plane': 1, 'Just My Luck': 1, 'Superman Returns': 1, 'You, Me and Dupree': 1, 'The Night Listener': 1}
	# 	0.396059017191
	
	# 	sim_tanimoto :  From sim_tanimoto	ci : set(['Lady in the Water', 'Snakes on a Plane', 'Just My Luck', 'Superman Returns', 'You, Me and Dupree', 'The Night Listener'])
	# 	0.911877394636

	print 
	print "Ranking the critics"
	print "==================="
	
	person = "Toby"
	print; print "Critics dont les gouts se rapprochent le plus de ceux de : ", person
	print; print topMatches(critics,person, n=3)

	
	print 
	print "Recommending Items"
	print "=================="

	person = "Toby"
	print; print "Les Movies que doit absolument voir : ", person,"- Calcul selon 3 Mesures differentes -par preference decroissante"
	
	print
	print """getRecommendations(critics,person,similarity=sim_distance)"""
	print; print getRecommendations(critics,person,similarity=sim_distance)
	
	print
	print """getRecommendations(critics,person,similarity=sim_pearson)"""
	print; print getRecommendations(critics,person,similarity=sim_pearson)
	
	print
	print """getRecommendations(critics,person,similarity=sim_tanimoto)"""
	print; print getRecommendations(critics,person,similarity=sim_tanimoto)


	print; print """
	Similarities betwen Movies:
	==========================="""
	
	itemsim = calculateSimilarItems(critics)
	
	pprint (itemsim)



	person = "Toby"

	print; print """
	Nouveau jeu de recommendations pour:""", person
	print """
	============================================="""
	

	gRec = getRecommendedItems(critics, itemsim, person)
	print; pprint(gRec)
	
#  Attention , il y a une legere difference dans le resultat p25
# Toby
# 
# [(3.1667425234070894, 'The Night Listener'),
#  (2.936629402844435, 'Just My Luck'),
#  (2.868767392626467, 'Lady in the Water')]

	print; print " Remarque, itemsim n'a pas beoin d etre recalculee a chaque interrogation"


def test_02():
	"""
# 	Ref:
# 	https://github.com/PyBeaner/ProgrammingCollectiveIntelligence/tree/master/Chapter2
# 	7 heures apres
	"""
	print; print """critics"""
	
	r = sim_distance(critics, "Lisa Rose", "Gene Seymour")
	print; print """sim_distance(critics,"Lisa Rose", "Gene Seymour") = """, r

	r = sim_pearson(critics,"Lisa Rose", "Gene Seymour")
	print; print """sim_pearson(critics,"Lisa Rose", "Gene Seymour") = """, r

	r = topMatches(critics,person="Lisa Rose")
	print; print """calculating  topMatches(critics,person="Lisa Rose" = )""",r
	
	r = topMatches(critics,"Lisa Rose",5,sim_distance)
	print; print """topMatches(critics,"Lisa Rose",5,sim_distance) = """, r

	print; print("Getting Recommendations")
	r = getRecommendations(critics,"Toby",sim_pearson)
	print; print """getRecommendations(critics,"Toby",sim_pearson = """, r


	print; print("movies related")
	movies = transformPrefs(critics)
	r = topMatches(movies,"Superman Returns")
	print; print """topMatches(movies,"Superman Returns") = """, r

	r = getRecommendations(movies,"Just My Luck")
	print; print getRecommendations(movies,"Just My Luck") , r


	print; print("item relationship")
	itemsSim = calculateSimilarItems(critics)
	print; pprint(itemsSim)

	print; print("rankings for Toby")
	r6 = getRecommendedItems(critics,itemsSim,'Toby')
	print; print(r6)


	print; print """movielens : test en reel d'une file"""
	print; print("movielens:recommendation for user(87)")
	prefs = loadMovieLens()
	items = getRecommendations(prefs,'87')[0:30]
	print; pprint(items)

	# print("it will take some time to calculate the similarity between items")
	# itemsSim = calculateSimilarItems(prefs,n=50)
	# print("after that,getting recommendations would be quite quick")
	# items = getRecommendedItems(prefs,itemsSim,"87")[0:30]
	# print(items)

	print; print("calculating similarity between people")
	peopleSim = calculateSimilarUsers(prefs,n=50)
	items = getRecommendedItemsBasedOnUser(prefs,peopleSim,"87")[0:30]
	print; print("recommended items are:")
	print; pprint(items)




if __name__ == '__main__':

	print; print __doc__
		
	print; print the_aim.__doc__, the_aim()
	
	print; print sources.__doc__, sources()
	

# 	print Recalage_sur_les_Exemples_du_Livre.__doc__
# 	Recalage_sur_les_Exemples_du_Livre ()


	print test_02.__doc__
	test_02()
	
	
	
	
	
output = """
Last login: Thu Sep 17 17:58:14 on ttys001
(Canopy 32bit) iMac-10DDB1BCC9D0:~ mm$ /var/folders/q1/pgnyc6955fb6fy1wc7yplc780000gn/T/Cleanup\ At\ Startup/recommendations-464223536.558.py.command ; exit;

Name of running program: __name__     =  __main__
Running file:           __file__      =  /projects/projects/workspace/pci/ch_02/recommendations.py
Does it resides in a:   __package__   =  None


Reprise by MM
	Today is 9/17/15. It is now 4:13 PM


def the_aim():
==============
	
	Playing with recommendation systems
	
	Code supporting Chapter 2 of 
	"Programming Collective Intelligence",
	Author: Toby SEGARAN
	                    
	                    First Edition
	None

	
def sources():
=================

# Ref:

1/
https://github.com/kilimandjaro2/PCI/blob/c2eefe23794be646baac3200027b2aca766df128/chapter2/recommendations.py

2/
rodelrod/collective-intelligence-GitHub

	
3/
https://github.com/PyBeaner/ProgrammingCollectiveIntelligence/tree/master/Chapter2
C'est la Liste des  3 sources consultees

# 	Ref:
# 	https://github.com/PyBeaner/ProgrammingCollectiveIntelligence/tree/master/Chapter2
# 	7 heures apres
	

critics

sim_distance(critics,"Lisa Rose", "Gene Seymour") =  0.294298055086

sim_pearson(critics,"Lisa Rose", "Gene Seymour") =  0.396059017191

calculating  topMatches(critics,person="Lisa Rose" = ) [(0.9912407071619299, 'Toby'), (0.7470178808339965, 'Jack Matthews'), (0.5940885257860044, 'Mick LaSalle'), (0.5669467095138396, 'Claudia Puig'), (0.40451991747794525, 'Michael Phillips')]

topMatches(critics,"Lisa Rose",5,sim_distance) =  [(0.4721359549995794, 'Michael Phillips'), (0.4142135623730951, 'Mick LaSalle'), (0.38742588672279304, 'Claudia Puig'), (0.3483314773547883, 'Toby'), (0.3405424265831667, 'Jack Matthews')]

Getting Recommendations

getRecommendations(critics,"Toby",sim_pearson =  [(3.3477895267131013, 'The Night Listener'), (2.8325499182641614, 'Lady in the Water'), (2.5309807037655645, 'Just My Luck')]

movies related

topMatches(movies,"Superman Returns") =  [(0.6579516949597695, 'You, Me and Dupree'), (0.4879500364742689, 'Lady in the Water'), (0.11180339887498941, 'Snakes on a Plane'), (-0.1798471947990544, 'The Night Listener'), (-0.42289003161103106, 'Just My Luck')]

[(4.0, 'Michael Phillips'), (3.0, 'Jack Matthews')] [(4.0, 'Michael Phillips'), (3.0, 'Jack Matthews')]

item relationship

{'Just My Luck': [(0.3483314773547883, 'Lady in the Water'),
                  (0.32037724101704074, 'You, Me and Dupree'),
                  (0.2989350844248255, 'The Night Listener'),
                  (0.2553967929896867, 'Snakes on a Plane'),
                  (0.20799159651347807, 'Superman Returns')],
 'Lady in the Water': [(0.4494897427831781, 'You, Me and Dupree'),
                       (0.38742588672279304, 'The Night Listener'),
                       (0.3483314773547883, 'Snakes on a Plane'),
                       (0.3483314773547883, 'Just My Luck'),
                       (0.2402530733520421, 'Superman Returns')],
 'Snakes on a Plane': [(0.3483314773547883, 'Lady in the Water'),
                       (0.32037724101704074, 'The Night Listener'),
                       (0.3090169943749474, 'Superman Returns'),
                       (0.2553967929896867, 'Just My Luck'),
                       (0.1886378647726465, 'You, Me and Dupree')],
 'Superman Returns': [(0.3090169943749474, 'Snakes on a Plane'),
                      (0.252650308587072, 'The Night Listener'),
                      (0.2402530733520421, 'Lady in the Water'),
                      (0.20799159651347807, 'Just My Luck'),
                      (0.1918253663634734, 'You, Me and Dupree')],
 'The Night Listener': [(0.38742588672279304, 'Lady in the Water'),
                        (0.32037724101704074, 'Snakes on a Plane'),
                        (0.2989350844248255, 'Just My Luck'),
                        (0.29429805508554946, 'You, Me and Dupree'),
                        (0.252650308587072, 'Superman Returns')],
 'You, Me and Dupree': [(0.4494897427831781, 'Lady in the Water'),
                        (0.32037724101704074, 'Just My Luck'),
                        (0.29429805508554946, 'The Night Listener'),
                        (0.1918253663634734, 'Superman Returns'),
                        (0.1886378647726465, 'Snakes on a Plane')]}

rankings for Toby

[(3.1667425234070894, 'The Night Listener'), (2.936629402844435, 'Just My Luck'), (2.868767392626467, 'Lady in the Water')]

movielens : test en reel d'une file

movielens:recommendation for user(87)

[(5.0, 'They Made Me a Criminal (1939)'),
 (5.0, 'Star Kid (1997)'),
 (5.0, 'Santa with Muscles (1996)'),
 (5.0, 'Saint of Fort Washington, The (1993)'),
 (5.0, 'Marlene Dietrich: Shadow and Light (1996) '),
 (5.0, 'Great Day in Harlem, A (1994)'),
 (5.0, 'Entertaining Angels: The Dorothy Day Story (1996)'),
 (5.0, 'Boys, Les (1997)'),
 (4.89884443128923, 'Legal Deceit (1997)'),
 (4.815019082242709, 'Letter From Death Row, A (1998)'),
 (4.7321082983941425, 'Hearts and Minds (1996)'),
 (4.696244466490867, 'Pather Panchali (1955)'),
 (4.652397061026758, 'Lamerica (1994)'),
 (4.538723693474813, 'Leading Man, The (1996)'),
 (4.535081339106103, 'Mrs. Dalloway (1997)'),
 (4.532337612572981, 'Innocents, The (1961)'),
 (4.527998574747078, 'Casablanca (1942)'),
 (4.510270149719864, 'Everest (1998)'),
 (4.493967755428439, 'Dangerous Beauty (1998)'),
 (4.485151301801343, 'Wallace & Gromit: The Best of Aardman Animation (1996)'),
 (4.463287461290221, 'Wrong Trousers, The (1993)'),
 (4.450979436941034, 'Kaspar Hauser (1993)'),
 (4.431079071179519, 'Usual Suspects, The (1995)'),
 (4.427520682864959, 'Maya Lin: A Strong Clear Vision (1994)'),
 (4.414870784592075, 'Wedding Gift, The (1994)'),
 (4.377445252656463, 'Affair to Remember, An (1957)'),
 (4.376071110447772, 'Good Will Hunting (1997)'),
 (4.376011099001396, 'As Good As It Gets (1997)'),
 (4.374146179500976, 'Anna (1996)'),
 (4.367437266504598, 'Close Shave, A (1995)')]

calculating similarity between people
100 / 943
200 / 943
300 / 943
400 / 943
500 / 943
600 / 943
700 / 943
800 / 943
900 / 943

recommended items are:

[(5.0, 'World of Apu, The (Apur Sansar) (1959)'),
 (5.0, 'Wallace & Gromit: The Best of Aardman Animation (1996)'),
 (5.0, 'Unbearable Lightness of Being, The (1988)'),
 (5.0, 'Tango Lesson, The (1997)'),
 (5.0, 'Stealing Beauty (1996)'),
 (5.0, 'Star Kid (1997)'),
 (5.0, "She's the One (1996)"),
 (5.0, 'Secret of Roan Inish, The (1994)'),
 (5.0, 'Room with a View, A (1986)'),
 (5.0, 'Rendezvous in Paris (Rendez-vous de Paris, Les) (1995)'),
 (5.0, 'Remains of the Day, The (1993)'),
 (5.0, 'Raise the Red Lantern (1991)'),
 (5.0, "Preacher's Wife, The (1996)"),
 (5.0, 'Pillow Book, The (1995)'),
 (5.0, 'People vs. Larry Flynt, The (1996)'),
 (5.0, 'Pather Panchali (1955)'),
 (5.0, 'Nikita (La Femme Nikita) (1990)'),
 (5.0, 'Night Flier (1997)'),
 (5.0, 'Letter From Death Row, A (1998)'),
 (5.0, 'Infinity (1996)'),
 (5.0, 'In Love and War (1996)'),
 (5.0, 'Garden of Finzi-Contini, The (Giardino dei Finzi-Contini, Il) (1970)'),
 (5.0, 'For the Moment (1994)'),
 (5.0, 'Enchanted April (1991)'),
 (5.0, 'Eat Drink Man Woman (1994)'),
 (5.0, 'Cyrano de Bergerac (1990)'),
 (5.0, 'Cinema Paradiso (1988)'),
 (5.0, 'Brassed Off (1996)'),
 (5.0, 'Bent (1997)'),
 (5.0, 'Before the Rain (Pred dozhdot) (1994)')]
logout

[Process completed]

'''
