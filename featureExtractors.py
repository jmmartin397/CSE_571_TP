# featureExtractors.py
# --------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"Feature extractors for Pacman game states"

from game import Directions, Actions
import util

class FeatureExtractor:
    def getFeatures(self, state, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()

class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[(state,action)] = 1.0
        return feats

class CoordinateExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[state] = 1.0
        feats['x=%d' % state[0]] = 1.0
        feats['y=%d' % state[0]] = 1.0
        feats['action=%s' % action] = 1.0
        return feats

def closestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None

def closestObject(pos, object_positions, walls):
    """
    closestObject -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if (pos_x, pos_y) in object_positions:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    return None

class SimpleExtractor(FeatureExtractor):
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    """

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)
        features.divideAll(10.0)
        return features

class BetterExtractor(FeatureExtractor):
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a capsule will be eaten
    - how far away the next capsule is
    - whether a ghost collision is imminent (with a scared ghost)
    - whether a ghost collision is imminent (with a non scared ghost)
    - whether a scared ghost is one step away
    - whether a non scared ghost is one step away
    """

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        capsules = state.getCapsules()
        scared_ghosts = []
        non_scared_ghosts = []
        ghost_states = state.getGhostStates()
        for ghost_state in ghost_states:
            if ghost_state.scaredTimer > 1: #will be scared for at least the next move
                scared_ghosts.append(ghost_state.getPosition())
            else:
                non_scared_ghosts.append(ghost_state.getPosition())

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of non-scared ghosts 1-step away
        features["#-of-non-scared-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(
            g, walls) for g in non_scared_ghosts)

        # count the number of scared ghosts 1-step away
        features["#-of-scared-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(
            g, walls) for g in scared_ghosts)
        # **********Commenting the ghost centroid extraction for now since I need to generalize it for more than 2 ghosts************
        # if non_scared_ghosts:
        #     centroid = lambda ghosts: (sum(gh[0] for gh in ghosts)/len(ghosts), sum(gh[1] for gh in ghosts)/len(ghosts))
        #     ghostCentroid = non_scared_ghosts[0] if len(non_scared_ghosts) == 1 else centroid(non_scared_ghosts)
        #     distance = lambda pac, centroid: math.sqrt((ghostCentroid[0]-pac[0])**2 + (ghostCentroid[1]-pac[1])**2)
        #     features["non-scared-centroid-distance"] = distance((x,y), centroid(non_scared_ghosts))/ (walls.width * walls.height)
        
        # if scared_ghosts:
        #     centroid = lambda ghosts: (sum(gh[0] for gh in ghosts)/len(ghosts), sum(gh[1] for gh in ghosts)/len(ghosts))
        #     ghostCentroid = scared_ghosts[0] if len(scared_ghosts) == 1 else centroid(scared_ghosts)
        #     distance = lambda pac, centroid: math.sqrt((ghostCentroid[0]-pac[0])**2 + (ghostCentroid[1]-pac[1])**2)
        #     features["scared-centroid-distance"] = distance((x,y), centroid(scared_ghosts))/ (walls.width * walls.height)

        if food[next_x][next_y]:
            features["can-eat-food"] = 1.0
        if (next_x, next_y) in capsules:
            features["can-eat-capsule"] = 1.0

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)

        dist = closestObject((next_x, next_y), capsules, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-capsule"] = float(dist) / (walls.width * walls.height)

        dist = closestObject((next_x, next_y), scared_ghosts, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-scared_ghost"] = float(dist) / (walls.width * walls.height)

        features.divideAll(10.0)
        return features

import itertools

class HunterExtractor(FeatureExtractor):
    """
    - whether a capsule will be eaten
    - how far away the next capsule is
    - sensory field showing locations of ghosts (scared and non-scared) 1 step away
    - distance to nearest scared ghost
    """
    def getFeatures(self, state, action):
        walls = state.getWalls()
        capsules = state.getCapsules()
        scared_ghosts = []
        non_scared_ghosts = []
        ghost_states = state.getGhostStates()
        for ghost_state in ghost_states:
            if ghost_state.scaredTimer > 2: #will be scared for at least the next move
                scared_ghosts.append(ghost_state.getPosition())
            else:
                non_scared_ghosts.append(ghost_state.getPosition())

        features = util.Counter()
        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # whether a capsule will be eaten
        if (next_x, next_y) in capsules:
            features["can-eat-capsule"] = 1.0

        # distance to closest capsule
        dist = closestObject((next_x, next_y), capsules, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update will diverge wildly
            features["closest-capsule"] = float(dist) / (walls.width * walls.height)

        # by adding each of these offsets to a position (x,y) you get the eight
        # surrounding positions and the position itself
        sensory_offsets = list(itertools.product([-1, 0, 1], [-1, 0, 1]))

        # sensory field for both types of ghosts
        for offset in sensory_offsets:
            pos = (next_x + offset[0], next_y + offset[1])
            if pos in non_scared_ghosts:
                features['sense-ghost-{}'.format(offset)] = 1
            if pos in scared_ghosts:
                features['sense-scared-ghost-{}'.format(offset)] = 1

        dist = closestObject((next_x, next_y), scared_ghosts, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update will diverge wildly
            features["closest-scared-ghost"] = float(dist) / (walls.width * walls.height)
        features.divideAll(10.0)
        return features