#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# multirobot.py
# Contact (ce fichier uniquement): nicolas.bredeche(at)upmc.fr
# 
# Description:
#   Template pour simulation mono- et multi-robots type khepera/e-puck/thymio
#   Ce code utilise pySpriteWorld, développé par Yann Chevaleyre (U. Paris 13)
# 
# Dépendances:
#   Python 2.x
#   Matplotlib
#   Pygame
# 
# Historique: 
#   2016-03-28__23h23 - template pour 3i025 (IA&RO, UPMC, licence info)
#   2018-03-27__20h55 - réécriture de la fonction step(.), utilisation de valeurs normalisées (senseurs et effecteurs). Interface senseurs, protection du nombre de commandes de translation et de rotation (1 commande par appel de step(.) pour chacune)
# 	2018-03-28__10:00 - renommage de la fonction step(.) en stepController(.), refactoring
#
# Aide: code utile
#   - Partie "variables globales": en particulier pour définir le nombre d'agents. La liste SensorBelt donne aussi les orientations des différentes capteurs de proximité.
#   - La méthode "stepController" de la classe Agent
#   - La fonction setupAgents (permet de placer les robots au début de la simulation)
#   - La fonction setupArena (permet de placer des obstacles au début de la simulation)
#   - il n'est pas conseillé de modifier les autres parties du code.
# 

from robosim import *
from random import random, shuffle, randint
import time
import sys
import atexit
from itertools import count
import math
import numpy as np
import scipy.special as sp


'''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''
'''  Aide                 '''
'''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''

#game.setMaxTranslationSpeed(3) # entre -3 et 3
# size of arena: 
#   screenw,screenh = taille_terrain()
#   OU: screen_width,screen_height

'''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''
'''  variables globales   '''
'''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''

game = Game()

agents = []
screen_width=512 #512,768,... -- multiples de 32  
screen_height=512 #512,768,... -- multiples de 32
nbAgents = 5

maxSensorDistance = 30              # utilisé localement.
maxRotationSpeed = 5
maxTranslationSpeed = 1

SensorBelt = [-170,-80,-40,-20,+20,40,80,+170]  # angles en degres des senseurs (ordre clockwise)

maxIterations = -1 # infinite: -1

showSensors = True
frameskip = 0   # 0: no-skip. >1: skip n-1 frames
verbose = True

'''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''
'''  Classe Agent/Robot   '''
'''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''

class Agent(object):
    
    agentIdCounter = 0 # use as static
    id = -1
    robot = -1
    name = "Equipe chekirou_kaci" # A modifier avec le nom de votre équipe

    translationValue = 0 # ne pas modifier directement
    rotationValue = 0 # ne pas modifier directement

    def __init__(self,robot):
        self.id = Agent.agentIdCounter
        Agent.agentIdCounter = Agent.agentIdCounter + 1
        #print "robot #", self.id, " -- init"
        self.robot = robot

    def getRobot(self):
        return self.robot

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    def stepController(self):
		
        params = [0, 1, 1, 1, 1, -0.18119810467087671, 0, 1, 1, 0.8141703138943657, -1, 0.902349482457901, -1, 1, 1, 0.11312830543725605, -1, 0.9896338430795174, 1, -1, 0, -1, 1, -1, -1, 1, 1, -1, 0, 0, -1, -0.026734838362200003, 1, 1, -1, 0, 1, 0, 1, -1, 0, 0, -1, -1, -1, -1, 1, -1, -1, 0, 1, 1, -0.8790231813317837, 1, 0, -1, 1, -1, 1, 0, -1, 0, 0, 0, 0, 0, -1, -1, 0, 1, 1, 1, 0.19785814842551097, -1, 1, -1, -1, 0, 1, 1, -1, 0, 0, -1, -1, 1, 1, 1, -1, 1, 1, 0, 0.11230066980987213, 1, 1, -1, 0, 1, 0, 1, 0.8920718889502143, -1, 0, 0.8232783146561736, -1, 1, -1, -1, 1, 0, -1, -0.8525001892824183, 0, 0, 1, 1, -0.9032353031174674, 1, 1, -1, -1, -1, -1, 0, 0, 0, 1, 1, -1, 1, 0, -1, 0, 1, 1, 0.8118896193559196, 0, 0, 1, 1, 0, -1, -1, 0, -1, -1, 1, 0, 1, 0, 0, 1, -1, 1, 0, 0, 0, 0, 0.09974391079058242, 0, 0, 0, 0, 1, -1, 1, 1, -1, 0, -1, 1, 1, 1, -0.06636307636984104, 0, 0, -1, 0, -1, -1, 0, 0, 0, 0, 1, -0.043022793929845926, 1, -1, 0, 1, 1, 0, -0.053594039582556346, 0, 1, 1, -1, -1, -1, 0, 1, -1, 0, 1, -1, 1, -1, 1, 0, 1, 1, -1, 1, 1, 0, -1, -1, -1, 0, 1]
        #print "robot #", self.id, " -- step"

        translation = 0
        rotation = 0
        nbhidden = 20
        
        sensorMinus80 = self.getDistanceAtSensor(1)
        sensorMinus40 = self.getDistanceAtSensor(2)
        sensorMinus20 = self.getDistanceAtSensor(3)
        sensorPlus20 = self.getDistanceAtSensor(4)
        sensorPlus40 = self.getDistanceAtSensor(5)
        sensorPlus80 = self.getDistanceAtSensor(6)

        if len(params) != 220: # vérifie que le nombre de paramètres donné est correct
            print ("[ERROR] number of parameters is incorrect. Exiting.")
            exit()

        # Perceptron: a linear combination of sensory inputs with weights (=parameters). Use an additional parameters as a bias, and apply hyperbolic tangeant to ensure result is in [-1,+1]
        ##translation =  math.tanh( sensorMinus40 * self.params[0] + sensorMinus20 * self.params[1] + sensorPlus20 * self.params[2] + sensorPlus40 * self.params[3] + self.params[4]) 
        ##rotation =  math.tanh( sensorMinus40 * self.params[5] + sensorMinus20 * self.params[6] + sensorPlus20 * self.params[7] + sensorPlus40 * self.params[8] + self.params[9])

        #print ("robot #", self.id, "[r =",rotation," - t =",translation,"]")
        inputs_list = np.array([self.getDistanceAtSensor(i) for i in range(0,8)] + [1])
        inputs = np.array(inputs_list, ndmin=2)
        #print(inputs)
        # calculate signals into hidden layer
        m = np.array(params[0:nbhidden * inputs_list.size ]).reshape(inputs_list.size, nbhidden)
        #print(m)

        hidden_inputs = np.dot( inputs,m)
        # calculate the signals emerging from hidden layer
        
        hidden_outputs = sp.expit(hidden_inputs)
        m = np.array(params[nbhidden * inputs_list.size:]).reshape(nbhidden, 2)
        # calculate signals into final output layer
        final_inputs = np.dot( hidden_outputs,m)
        # calculate the signals emerging from final output layer
        final_outputs = sp.expit(final_inputs)
        #print(final_outputs)
        self.setRotationValue( final_outputs[0][0])

        self.setTranslationValue( final_outputs[0][1] )
        # monitoring - affiche diverses informations sur l'agent et ce qu'il voit.
        # pour ne pas surcharger l'affichage, je ne fais ca que pour le player 1
        if verbose == True and self.id == 0:

            efface()    # j'efface le cercle bleu de l'image d'avant
            color( (0,0,255) )
            circle( *game.player.get_centroid() , r = 22) # je dessine un rond bleu autour de ce robot
	
			# monitoring (optionnel - changer la valeur de verbose)
            print("Robot #"+str(self.id)+" :")
            for i in range(len(SensorBelt)):
                print("\tSenseur #"+str(i)+" (angle: "+ str(SensorBelt[i])+"°)")
                print ("\t\tDistance  :",self.getDistanceAtSensor(i))
                print ("\t\tType      :",self.getObjectTypeAtSensor(i)) # 0: rien, 1: mur ou bord, 2: robot
                print("\t\tRobot info:",self.getRobotInfoAtSensor(i)) # dict("id","centroid(x,y)","orientation") (si pas de robot: renvoi "None" et affiche un avertissement dans la console

        return

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


    def step(self):
        self.stepController()
        self.move()

    def move(self):
        self.robot.forward(self.translationValue)
        self.robot.rotate(self.rotationValue)

    def getDistanceAtSensor(self,id):
        sensor_infos = sensors[self.robot] # sensor_infos est une liste de namedtuple (un par capteur).
        return min(sensor_infos[id].dist_from_border,maxSensorDistance) / maxSensorDistance

    def getObjectTypeAtSensor(self,id):
        if sensors[self.robot][id].dist_from_border > maxSensorDistance:
            return 0 # nothing
        elif sensors[self.robot][id].layer == 'joueur':
            return 2 # robot
        else:
            return 1 # wall/border

    def getRobotInfoAtSensor(self,id):
        if sensors[self.robot][id].dist_from_border < maxSensorDistance and sensors[self.robot][id].layer == 'joueur':
            otherRobot = sensors[self.robot][id].sprite
            info = {'id': otherRobot.numero, 'centroid': otherRobot.get_centroid(), 'orientation': otherRobot.orientation()}
            return info
        else:
            #print "[WARNING] getPlayerInfoAtSensor(.): not a robot!"
            return None

    def setTranslationValue(self,value):
        if value > 1:
            print( "[WARNING] translation value not in [-1,+1]. Normalizing.")
            value = maxTranslationSpeed
        elif value < -1:
            print ("[WARNING] translation value not in [-1,+1]. Normalizing.")
            value = -maxTranslationSpeed
        else:
            value = value * maxTranslationSpeed
        self.translationValue = value

    def setRotationValue(self,value):
        if value > 1:
            print ("[WARNING] translation value not in [-1,+1]. Normalizing.")
            value = maxRotationSpeed
        elif value < -1:
            print ("[WARNING] translation value not in [-1,+1]. Normalizing.")
            value = -maxRotationSpeed
        else:
            value = value * maxRotationSpeed
        self.rotationValue = value


'''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''
'''  Fonctions init/step  '''
'''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''

def setupAgents():
    global screen_width, screen_height, nbAgents, agents, game

    # Make agents
    nbAgentsCreated = 0
    for i in range(nbAgents):
        while True:
            p = -1
            while p == -1: # p renvoi -1 s'il n'est pas possible de placer le robot ici (obstacle)
                p = game.add_players( (random()*screen_width , random()*screen_height) , None , tiled=False)
            if p:
                p.oriente( random()*360 )
                p.numero = nbAgentsCreated
                nbAgentsCreated = nbAgentsCreated + 1
                agents.append(Agent(p))
                break
    game.mainiteration()


def setupArena():
    addObstacle(row=4,col=3)
    addObstacle(row=5,col=12)
    addObstacle(row=8,col=12)
    addObstacle(row=11,col=3)
    addObstacle(row=10,col=3)
    addObstacle(row=9,col=3)
    addObstacle(row=12,col=5)
    addObstacle(row=10,col=8)
    addObstacle(row=6,col=9)
    addObstacle(row=1,col=2) 

def updateSensors():
    global sensors
    # throw_rays...(...) : appel couteux (une fois par itération du simulateur). permet de mettre à jour le masque de collision pour tous les robots.
    sensors = throw_rays_for_many_players(game,game.layers['joueur'],SensorBelt,max_radius = maxSensorDistance+game.player.diametre_robot() , show_rays=showSensors)

def stepWorld():
    global sensors
    
    updateSensors()

    # chaque agent se met à jour. L'ordre de mise à jour change à chaque fois (permet d'éviter des effets d'ordre).
    shuffledIndexes = [i for i in range(len(agents))]
    shuffle(shuffledIndexes)
    for i in range(len(agents)):
        agents[shuffledIndexes[i]].step()
    return


'''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''
'''  Fonctions internes   '''
'''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''

def addObstacle(row,col):
    # le sprite situe colone 13, ligne 0 sur le spritesheet
    game.add_new_sprite('obstacle',tileid=(0,13),xy=(col,row),tiled=True)

class MyTurtle(Turtle): # also: limit robot speed through this derived class
    maxRotationSpeed = maxRotationSpeed # 10, 10000, etc.
    def rotate(self,a):
        mx = MyTurtle.maxRotationSpeed
        Turtle.rotate(self, max(-mx,min(a,mx)))

def onExit():
    print ("\n[Terminated]")

'''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''
'''  Main loop            '''
'''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''

init('empty',MyTurtle,screen_width,screen_height) # display is re-dimensioned, turtle acts as a template to create new players/robots
game.auto_refresh = False # display will be updated only if game.mainiteration() is called
game.frameskip = frameskip
atexit.register(onExit)

setupArena()
setupAgents()
game.mainiteration()

iteration = 0
while iteration != maxIterations:
    stepWorld()
    game.mainiteration()
    iteration = iteration + 1
