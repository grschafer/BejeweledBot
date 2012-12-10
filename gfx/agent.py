from pybrain.rl.agents.logging import LoggingAgent
from pybrain.rl.agents.learning import LearningAgent
from scipy import where
from random import choice

class BejeweledAgent(LearningAgent):

    def getAction(self):
        # get best action for every state observation
        # overlay all action values for every state observation, pick best
        LoggingAgent.getAction(self)

        # for each color, get best action, then pick highest-value action
        # among those actions
        actions = []
        values = []
        num_colors = len(self.lastobs[0])
        # TODO: why are same values printed many times in a row here? episodes
        #print '========== in agent =========='
        #print 'states:', self.lastobs
        for board_loc in self.lastobs:
            for color_state in board_loc:
                #print 'state:', color_state
                actions.append(self.module.activate(color_state))
                values.append(self.module.lastMaxActionValue)
                #self.module.printState(state)
                #print ' best:', actions[-1], 'value:', values[-1]

                # add a chance to pick a random other action
                if self.learning:
                    actions[-1] = self.learner.explore(color_state, actions[-1])

        actionIdx = where(values == max(values))[0]
        ch = choice(actionIdx)
        self.lastaction = [actions[ch], ch]
        loc, color = divmod(ch, num_colors)
        self.bestState = self.lastobs[loc][color]

        #print 'assigning reward to state', self.bestState
        #print 'chosen action:', self.lastaction, 'value:', max(values)

        #print '============= end ============'
        return self.lastaction

    def giveReward(self, r):
        """Step 3: store observation, action and reward in the history dataset. """
        # step 3: assume that state and action have been set
        assert self.lastobs != None
        assert self.lastaction != None
        assert self.lastreward == None

        self.lastreward = r

        # store state, action and reward in dataset if logging is enabled
        if self.logging:
            # TODO: assigning reward to only best estimate for now
            #for state in self.lastobs:
                # TODO: assign reward to state correctly? NO because we're in
                #  the learner -- learning will be slower though, because of
                #  false positives for every obs
                self.history.addSample(self.bestState, self.lastaction[0], self.lastreward)
