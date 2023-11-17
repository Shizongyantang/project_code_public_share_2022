import random
import pickle
import os
import numpy as np


class Agent(object):
    def __init__(self, agent_number, params={}):
        self.this_agent_number = agent_number  # index for this agent
        self.opponent_number = 1 - agent_number  # index for opponent
        self.project_part = params['project_part'] #useful to be able to use same competition code for each project part
        self.n_items = params["n_items"]

        # Potentially useful for Part 2 -- 
        # Unpickle the trained model
        # Complications: pickle should work with any machine learning models
        # However, this does not work with custom defined classes, due to the way pickle operates
        # TODO you can replace this with your own model
        self.filename = 'agents/yourteamname/trained_model'
        self.trained_model = pickle.load(open(self.filename, 'rb'))
        self.pricing_history = []
        self.customer_decision_history = []
        self.competitor_pricing_history = []

    def _process_last_sale(self, last_sale, profit_each_team):
        # print("last_sale: ", last_sale)
        # print("profit_each_team: ", profit_each_team)
        my_current_profit = profit_each_team[self.this_agent_number]
        opponent_current_profit = profit_each_team[self.opponent_number]

        my_last_prices = last_sale[2][self.this_agent_number]
        opponent_last_prices = last_sale[2][self.opponent_number]

        did_customer_buy_from_me = last_sale[1] == self.this_agent_number
        did_customer_buy_from_opponent = last_sale[1] == self.opponent_number

        which_item_customer_bought = last_sale[0]

        # print("My current profit: ", my_current_profit)
        # print("Opponent current profit: ", opponent_current_profit)
        # print("My last prices: ", my_last_prices)
        # print("Opponent last prices: ", opponent_last_prices)
        # print("Did customer buy from me: ", did_customer_buy_from_me)
        # print("Did customer buy from opponent: ",
        #       did_customer_buy_from_opponent)
        # print("Which item customer bought: ", which_item_customer_bought)

        # TODO - add your code here to potentially update your pricing strategy based on what happened in the last round
        self.pricing_history.append(my_last_prices)
        self.competitor_pricing_history.append(opponent_last_prices)
        self.customer_decision_history.append(did_customer_buy_from_me)
        pass

    # Given an observation which is #info for new buyer, information for last iteration, and current profit from each time
    # Covariates of the current buyer, and potentially embedding. Embedding may be None
    # Data from last iteration (which item customer purchased, who purchased from, prices for each agent for each item (2x2, where rows are agents and columns are items)))
    # Returns an action: a list of length n_items, indicating prices this agent is posting for each item.
    def action(self, obs):
        new_buyer_covariates, last_sale, profit_each_team = obs
        self._process_last_sale(last_sale, profit_each_team)

        # Example strategy for Part 1:
        if self.project_part == 1:
            customer_valuation = new_buyer_covariates[0]
            suggested_price = self.determine_price(customer_valuation)
            return [suggested_price]
        # For Part 1, new_buyer_covariates will simply be a vector of length 1, containing a single numeric float indicating the valuation the user has for the (single) item
        # For Part 2, new_buyer_covariates will be a vector of length 3 that can be used to estimate demand from that user for each of the two items
        new_buyer_covariates, last_sale, profit_each_team = obs
        self._process_last_sale(last_sale, profit_each_team)

        # Potentially useful for Part 1 --
        # Currently output is just a deterministic price for the item, but students are expected to use the valuation (inside new_buyer_covariates) and history of prices from each team to set a better price for the item
        if self.project_part == 1:
            return [3]

        # Potentially useful for Part 2 -- 
        # TODO Currently this output is just a deterministic 2-d array, but the students are expected to use the buyer covariates to make a better prediction
        # and to use the history of prices from each team in order to set prices for each item.
        if self.project_part == 2:
            return self.trained_model.predict(np.array([1, 2, 3]).reshape(1, -1))[0] + random.random()
    def determine_price(self, customer_valuation):
        """
        Determines the price to set based on customer valuation and historical data.
        """

        if len(self.customer_decision_history) > 0 and not self.customer_decision_history[-1]:
            last_competitor_price = self.competitor_pricing_history[-1]
            last_my_price = self.pricing_history[-1]

            if isinstance(last_competitor_price, (list, tuple)):
                last_competitor_price = sum(last_competitor_price) / len(last_competitor_price)  

            if isinstance(last_my_price, (list, tuple)):
                last_my_price = sum(last_my_price) / len(last_my_price) 
            if last_my_price > last_competitor_price:
                return max(customer_valuation * 0.9, last_competitor_price * 1.01)
        return customer_valuation * 0.95
