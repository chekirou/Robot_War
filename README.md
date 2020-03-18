# Sorbonne University
# Robot war 
Artficial intelligence and operational research project on behavioral multi-agent robots.
 - this project takes the form of a tournament, each team has 5 robots who can follow any behavior wanted.
 - each robot has eight sensors and detects the distance to the object and its type (wall, own robot or enemy robot)
 - the goal of the game is to own the biggest part of the arena.
 - a position is held by a team if the last agent to go over this position belongs to the team.

# behavior: 
- if nothing is detected => explore 
- if a wall is near  => the avoiding behavior is followed (trained)
- if an enemy robot is following us => stop
- if a robot is detected => avoiding behavior 
- if stuck (rare) => move randomly in any direction

# tested strategies for trainning : 
- genetical algorithms with different muting methodes
- neural network 
- multiples fitnesses 
- multiple arenas
	
