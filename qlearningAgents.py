# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        self.q_values = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        return self.q_values[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        actions = self.getLegalActions(state)
        if not actions: #None or empty list
            return 0
        return max(self.getQValue(state, action) for action in actions)

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        actions = self.getLegalActions(state)
        if not actions: #None or empty list
            return None
        best_action = None
        best_q_value = float('-inf')
        for action in actions:
            q_value = self.getQValue(state, action)
            if q_value > best_q_value:
                best_q_value = q_value
                best_action = action
        return best_action

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        if not legalActions: #None or empty list
            return None
        if flipCoin(self.epsilon):
            action = random.choice(legalActions)
        else:
            action = self.computeActionFromQValues(state)
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        next_state_actions = self.getLegalActions(nextState)
        if not next_state_actions: #None of empty list
            max_next_q = 0
        else:
            max_next_q = max(self.getQValue(nextState, next_action) for next_action in next_state_actions)
        sample = reward + self.discount * max_next_q
        new_q = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * sample
        self.q_values[(state, action)] = new_q

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        features = self.featExtractor.getFeatures(state, action)
        q_value = 0
        weights = self.getWeights()
        for feature, value in features.items():
            q_value += weights[feature] * value
        return q_value

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        next_state_actions = self.getLegalActions(nextState)
        if not next_state_actions: #None of empty list
            max_next_q = 0
        else:
            max_next_q = max(self.getQValue(nextState, next_action) for next_action in next_state_actions)
        difference = reward + self.discount * max_next_q - self.getQValue(state, action)
        features = self.featExtractor.getFeatures(state, action)
        new_weights = self.getWeights().copy()
        current_weights = self.getWeights()
        for feature, value in features.items():
            new_weights[feature] = current_weights[feature] + self.alpha * difference * value
        self.weights = new_weights
        

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"

class TrueOnlineLambdaSarsa(ApproximateQAgent):
    """
       ApproximateTrueOnlineLambdaSarasaAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """  

    def __init__(self, lamb=0.9, extractor='IdentityExtractor', **args):
        ApproximateQAgent.__init__(self, **args)
        self.featExtractor = util.lookup(extractor, globals())()
        self.lamb = lamb
        self.z = util.Counter()
        self.weights = util.Counter()
        self.qOld = 0

    def getWeights(self):
        return self.weights

    def getEligiblityTraces(self):
        return self.z
    
    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        feats = self.featExtractor.getFeatures(state, action)
        weights = self.getWeights()
        return sum(feats[k]*weights[k] for k in feats.keys())

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        feats = self.featExtractor.getFeatures(state, action)
        nextAction = self.getAction(nextState)
        Q = self.getQValue(state, action)
        nextQ = self.getQValue(nextState, nextAction) if nextAction else 0
        TDE = reward + self.discount * nextQ - Q
        for k in feats.keys():
          self.z[k] = self.lamb*self.discount*self.z[k] + (1 - self.alpha*self.discount*self.lamb*sum(feats[k]*self.z[k] for k in feats.keys()))*feats[k]
          self.weights[k] += self.alpha*(TDE + Q - self.qOld)*self.z[k] - self.alpha*(Q - self.qOld)*feats[k]
        self.qOld = nextQ

    def clearTraces(self):
        self.qOld = 0
        self.z = util.Counter()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        self.clearTraces()
        ApproximateQAgent.final(self, state)
