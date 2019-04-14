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
frameskip = 3   # 0: no-skip. >1: skip n-1 frames
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
		
        params =[-1.4048749268174487, 3.337245973955776, -2.713266548829219, 1.9861026628589011, 0.7980404463131668, 1.3647650389863386, -2.400655332331208, 1.729963539266255, 0.47562016441351435, -0.48057899418837724, 1.6034268063986103, 2.1500900873230857, 1.163370357707722, -1.7888964319304943, 1.5006569537888599, -1.130231674956136, 0.4215742227007578, 0.45772670300041507, 3.1441326746329166, -0.5697228540885226, -0.28519967482151165, 0.05960342847902533, -1.2774977313697955, -1.7810186738433516, 1.429076986357299, 0.49871051667280686, -0.36883046633806593, -2.741680401230393, 2.100486940712768, 2.5209715017246923, -1.107306747488219, -4.114831431278385, 1.5729445792457248, 0.8708616296913966, 0.782944809753938, 3.1201598963952466, -0.8171546152351681, 2.2204410824322043, -0.8064869792585352, 0.6663734012122439, 2.321020856120491, -2.6540184381260317, 0.2506593558202532, -0.4309422788770461, 0.8538775696302291, 3.483227869937267, -0.40588951573123183, 2.481291912125245, -2.721054480610473, 0.9356807074760193, 1.639342251630141, 2.8122692507007314, 0.4689574363076966, -0.49447298013724666, 1.070589795845843, 1.5463569036932676, -0.7628977760628951, 1.5962901279486017, 0.647183562099712, -1.159918478340115, 2.0660420615872375, -0.25741288474055546, 1.5912222351689171, -2.6292213228493786, -0.9208380749784806, 0.33894805791273297, 0.06136759772207001, 2.050361684703867, -0.7879465303815094, -1.6695984500607122, -4.818050196014815, -1.0485333869936626, 1.1913359807427883, -1.4099640521982866, 0.22264641311248465, -0.0700757317535978, -2.78989294518952, -2.693393995808787, -2.016149653343856, -0.30274810031222626, 1.503569136924936, -0.0033172622808860797, 0.5818550654500476, -0.08018756149103752, -0.32976160948569583, 1.3924352568224247, 2.821213886830532, -1.9262781249190268, 1.566038874419812, 4.3911641436788775, 1.5272969295344194, 2.087677545505941, 0.8830035096724843, 1.7779935694975741, 0.5011823765342661, 0.3791246491139391, 1.9758107825542481, 1.4579548674432123, 0.8476344055083607, 0.14203610081400536, 6.013536329960462, -2.588463652912219, 1.0587468662896709, 2.1725007326304855, 1.950435056612018, -0.6547180675493663, 2.041354046521038, 2.4186429850415787, -0.17889405778580525, -0.4548994378131149, 0.018422413529993364, 0.7992851691362528, -1.6035250596980053, 2.219641757414725, -0.7950881261593515, 1.2537630542890095, 0.9478192702603699, 0.02161483984407016, 0.25557408446959995, 0.16719658533896112, -1.2944652057411137, 0.5771435548749676, -0.045262946064920595, 0.5198767846276866, -0.38721778342948054, -0.8966091187299136, -1.0572365151567977, -5.227739051591392, -3.2297337567993694, -1.0478939128134266, 0.05860208249738327, 0.9006923398752607, -1.6241290222501847, 2.5340229866527033, -1.1985561933029936, 3.581731972177177, 2.1845940966628987, -1.2789211004594794, 0.8171022687868031, -1.400006814951472, 1.3084419990456044, -0.008772644394814746, -0.5124256263073121, -1.1940482571381617, 0.7518974959255539, 3.1258043766204935, 0.7230925460175885, 1.6098999305818307, 0.5462786778725323, 0.6546067277303537, -1.8048821311315477, -0.6244672625990499, 1.3839228822146723, 1.3909498550413502, 2.022495059406169, 4.671768992223021, -1.1868376146168582, -1.8873095846259609, 2.45093312845352, -2.3672147685791973, -2.514570817471546, 1.077032969320966, -2.923564947646798, -0.8586754468180804, -1.5052810667662953, -2.0357186853329323, -1.7865152817516115, -2.5219075652099088, -2.2407218564814477, 0.05381104997734365, 1.9425755716999191, 0.8703562848554122, -3.0624549433792376, -1.0654778298831997, -0.04517654858637926, -0.6804152690119185, 1.0257288657717536, -2.0204418580960914, 0.28418966890027986, -1.3547612404453413, -0.18039792849792682, 0.40141525829769276, 1.847406973780897, -0.33914033931330073, -1.21388686741756, 1.164101439122105, -0.08332007955273502, -3.045264711065223, 1.4820238977741649, 2.026990825592867, 1.1955683350800321, -1.8894276542467834, -3.920267817900465, -0.39330248942568674, 3.1805914063497194, 2.879597318780972, 1.3758606230914163, -0.25704466511703844, 1.673686688271299, 1.6154258617416912, -3.1217773803209035, -3.071087884975112, 2.8517549654529204, 1.667500516153579, -0.7243099839162711, 2.2240321691477876, -4.109157633993524, -0.16282628613782096, -1.980611008449568, 1.148616092855812, 0.15475164931705243, 0.8115667916855125, 3.3778953924202453, -1.1795955384304186, 0.3020405247173672, 1.1520621345188895, -0.34742591563585024, -1.8550781247076775, 3.2601474262385004, -3.831144401214184]
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
