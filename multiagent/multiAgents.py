# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC
# Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import random

import util
from game import Agent
from util import manhattanDistance


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation
        function.

        Just like in the previous project, getAction takes a GameState and
        returns some Directions.X for some X in the set {North, South, West,
        East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[
            index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are
        better.

        The code below extracts some useful information from the state, like
        the remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # Compute the minimum manhattanDistance to the nearest ghost
        ghostPositions = successorGameState.getGhostPositions()
        ghostDists = [
            manhattanDistance(ghostPosition, newPos) for ghostPosition in
            ghostPositions]
        minGhostDist = min(ghostDists)
        minGhostDist = minGhostDist if minGhostDist < 4.0 else 4.0
        minIndices = [index for index in range(len(ghostDists)) if ghostDists[
            index] == minGhostDist]
        for ind in minIndices:
            # If we get closer to a scared ghost we're happy :)
            if newScaredTimes[ind] > 0:
                minGhostDist = 15.0 / minGhostDist
                break

        # Scared bonus
        scaredBonus = 0
        for scaredtime in newScaredTimes:
            if scaredtime > 0:
                scaredBonus = 100
                break

        # Compute minimum distance to next capsule
        minCapsuleDist = 1
        capsulePositions = successorGameState.getCapsules()
        if len(capsulePositions) > 0:
            capsuleDists = [
                manhattanDistance(capsulePosition, newPos) for capsulePosition
                in capsulePositions]
            minCapsuleDist = min(capsuleDists)

        foodPositions = newFood.asList()
        foodDists = [
            manhattanDistance(foodPosition, newPos) for foodPosition in
            foodPositions]
        minFoodDist = 1
        if len(foodDists) > 0:
            minFoodDist = min(foodDists)

        score = successorGameState.getScore() + minGhostDist + \
                5.0 / minCapsuleDist + 10.0 / minFoodDist + scaredBonus

        # # For debugging only
        # print "==================================="
        # print "minGhostDist {}".format(minGhostDist)
        # print "minCapsuleDist {}".format(1.0/minCapsuleDist)
        # print "minFoodDist {}".format(1.0/minFoodDist)
        # print "scaredBonus {}".format(scaredBonus)
        # print "score {}".format(score)

        return score


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using
          self.depth and self.evaluationFunction.

          Here are some method calls that might be useful when implementing
          minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions(0)
        v = float("-inf")
        bestAction = legalMoves[0]
        for action in legalMoves:
            successorState = gameState.generateSuccessor(0, action)
            stateVal = self.min(successorState, 0, 1)
            if stateVal > v:
                v = stateVal
                bestAction = action
        return bestAction

    def max(self, gameState, curDepth, agentIndex):
        if curDepth == self.depth:
            return scoreEvaluationFunction(gameState)
        else:
            legalMoves = gameState.getLegalActions(agentIndex)
            v = float("-inf")
            if len(legalMoves) == 0:
                return scoreEvaluationFunction(gameState)
            else:
                for action in legalMoves:
                    successState = \
                        gameState.generateSuccessor(agentIndex, action)
                    v = max(v, self.min(successState, curDepth, agentIndex + 1))
                return v

    def min(self, gameState, curDepth, agentIndex):
        if curDepth == self.depth:
            return scoreEvaluationFunction(gameState)
        else:
            legalMoves = gameState.getLegalActions(agentIndex)
            v = float("inf")
            if len(legalMoves) == 0:
                return scoreEvaluationFunction(gameState)
            else:
                for action in legalMoves:
                    successState = \
                        gameState.generateSuccessor(agentIndex, action)
                    if agentIndex == gameState.getNumAgents() - 1:
                        v = min(v, self.max(successState, curDepth + 1, 0))
                    else:
                        v = min(v, self.min(successState, curDepth,
                                            agentIndex + 1))
                return v


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # Collect legal moves and successor states
        alpha = float("-inf")
        beta = float("inf")
        legalMoves = gameState.getLegalActions(0)
        v = float("-inf")
        bestAction = legalMoves[0]
        for action in legalMoves:
            successorState = gameState.generateSuccessor(0, action)
            stateVal = self.min(successorState, 0, 1, alpha, beta)
            if stateVal > v:
                v = stateVal
                bestAction = action
            alpha = max(v, alpha)
        return bestAction

    def max(self, gameState, curDepth, agentIndex, alpha, beta):
        if curDepth == self.depth:
            return scoreEvaluationFunction(gameState)
        else:
            legalMoves = gameState.getLegalActions(agentIndex)
            v = float("-inf")
            if len(legalMoves) == 0:
                return scoreEvaluationFunction(gameState)
            else:
                for action in legalMoves:
                    successState = \
                        gameState.generateSuccessor(agentIndex, action)
                    v = max(v, self.min(successState, curDepth, agentIndex + 1, alpha, beta))
                    if v > beta:
                        return v
                    alpha = max(v, alpha)
                return v

    def min(self, gameState, curDepth, agentIndex, alpha, beta):
        if curDepth == self.depth:
            return scoreEvaluationFunction(gameState)
        else:
            legalMoves = gameState.getLegalActions(agentIndex)
            v = float("inf")
            if len(legalMoves) == 0:
                return scoreEvaluationFunction(gameState)
            else:
                for action in legalMoves:
                    successState = \
                        gameState.generateSuccessor(agentIndex, action)
                    if agentIndex == gameState.getNumAgents() - 1:
                        v = min(v, self.max(successState, curDepth + 1, 0, alpha, beta))
                    else:
                        v = min(v, self.min(successState, curDepth,
                                            agentIndex + 1, alpha, beta))
                    if v < alpha:
                        return v
                    beta = min(v, beta)
                return v



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and
          self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from
          their legal moves.
        """
        "*** YOUR CODE HERE ***"
        legalMoves = gameState.getLegalActions(0)
        v = float("-inf")
        bestAction = legalMoves[0]
        for action in legalMoves:
            successorState = gameState.generateSuccessor(0, action)
            stateVal = self.expecti(successorState, 0, 1)
            if stateVal > v:
                v = stateVal
                bestAction = action
        return bestAction

    def max(self, gameState, curDepth, agentIndex):
        if curDepth == self.depth:
            return scoreEvaluationFunction(gameState)
        else:
            legalMoves = gameState.getLegalActions(agentIndex)
            v = float("-inf")
            if len(legalMoves) == 0:
                return scoreEvaluationFunction(gameState)
            else:
                for action in legalMoves:
                    successState = \
                        gameState.generateSuccessor(agentIndex, action)
                    v = max(v, self.expecti(successState, curDepth, agentIndex + 1))
                return v

    def expecti(self, gameState, curDepth, agentIndex):
        if curDepth == self.depth:
            return scoreEvaluationFunction(gameState)
        else:
            legalMoves = gameState.getLegalActions(agentIndex)
            numMoves = len(legalMoves)
            v = 0.0
            if numMoves == 0:
                return scoreEvaluationFunction(gameState)
            else:
                for action in legalMoves:
                    successState = \
                        gameState.generateSuccessor(agentIndex, action)
                    if agentIndex == gameState.getNumAgents() - 1:
                        v += self.max(successState, curDepth + 1, 0)
                    else:
                        v += self.expecti(successState, curDepth, agentIndex + 1)
                return v/numMoves


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
