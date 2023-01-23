import random


IDS = ["213336753, 212362024"]


class Agent:
    def __init__(self, initial_state, player_number):
        self.ids = IDS

    def act(self, state):
        raise NotImplementedError


class UCTAgent:
    def __init__(self, initial_state, player_number):
        self.ids = IDS

    def selection(self, UCT_tree):
        raise NotImplementedError

    def expansion(self, UCT_tree, parent_node):
        raise NotImplementedError

    def simulation(self):
        raise NotImplementedError

    def backpropagation(self, simulation_result):
        raise NotImplementedError

    def act(self, state):
        raise NotImplementedError


class TreeSearchNode:
    def __init__(self, depth, j_state, action=None, parent=None):
        """
        :param depth: the depth of the node in the tree
        :param j_state: the json representation of the state
        :param action: action that led from parent to this node
        :param parent: node of the parent state
        """
        self.action = action
        self.parent = parent
        self.depth = depth
        self.j_state = j_state
        self.n_visits = 0
        self.q_value = 0
        # dictionary with actions as keys and sets of TreeSearchNodes as values
        self.children = defaultdict(list)

    def set_children(self, children):
        """
        :param children: list of nodes to add to the children's dictionary
        """
        for child in children:
            self.children[child.action].append(child)

    def get_uct_value(self, explore=0.9):
        """
        :return: the uct value of the node w.r.t its parent
        """
        if self.n_visits == 0:
            return 0 if explore == 0 else np.inf
        else:
            return self.q_value + explore * np.sqrt(2 * np.log(self.parent.n_visits) / self.n_visits)


class TaxiAgent2(Agent):
    """
    Agent which implements a MCTS algorithm. Uses the TreeSearchNode class
    """

    def __init__(self, initial):
        super().__init__(initial)

        self.start_time = time.time()
        self.root = TreeSearchNode(0, self.j_reduced_init)
        self.current_node = self.root
        self.previous_action = None
        self.node_count = 0
        self.num_rollouts = 0

        self.get_policy(timeout=20)

    def act(self, state):
        """
        given a state, return the best action according to the tree
        """
        if self.previous_action is None:
            to_return = self.best_action(self.current_node)
            return to_return

        else:
            state = {'terminated': False, 'taxis': state['taxis'], 'passengers': state['passengers']}
            j_state = json.dumps(state)
            for node in self.current_node.children[self.previous_action]:
                if node.j_state == j_state:
                    self.current_node = node
                    self.previous_action = self.best_action(node)
                    return self.previous_action

    def get_policy(self, timeout=300):
        """
        find policy using UCT algorithm
        :param timeout: time we have to run the procedure
        """
        n_simulations = 0

        # within the given time frame (take one second spare)
        while time.time() - self.start_time < timeout - 1:
            node = self.selection()
            reward = self.simulation(node)
            self.backpropagation(node, reward)
            n_simulations += 1

    def weighted_average_value(self, nodes_set, explore=0.9):
        """
        return the average value of states in c, weighted by the probability to reach each of them
        :param nodes_set: set of children nodes
        :return: float
        """
        v = 0.0
        for node in nodes_set:
            v += node.get_uct_value(explore=explore) * self.p(node.parent.j_state, node.action, node.j_state)
        return v

    def selection(self):
        """
        selection part of the algorithm
        :return:
        """
        node = self.root

        # until we reach a leaf in the tree
        while len(node.children) > 0:
            # descend using the action that maximizes the weighted average value of children nodes corresponding to that
            # action.
            value_action_pairs = [(self.weighted_average_value(c), a) for a, c in node.children.items()]
            max_value, max_action = max(random.sample(value_action_pairs, len(value_action_pairs)), key=lambda x: x[0])
            # select the next node from the possible children corresponding to max_action according to the probability
            node = random.choices(list(node.children[max_action]),
                                  weights=[self.p(node.j_state, max_action, next_node.j_state)
                                           for next_node in node.children[max_action]], k=1)[0]
            # if the note has not been explored, explore it before moving on
            if node.n_visits == 0:
                return node

        # we either reached a terminal node or a leaf in the current tree. If it is not terminal and we can expand it,
        # return one of its children. otherwise, return itself.
        if self.expansion(node):
            a = random.choice(list(node.children.keys()))
            node = random.choices(list(node.children[a]),
                                  weights=[self.p(node.j_state, a, next_node.j_state)
                                           for next_node in node.children[a]], k=1)[0]

        return node

    def expansion(self, parent):
        """
        expand the given parent node and add children to the tree.
        :return: True if not reached maxed depth, otherwise return False.
        """
        if parent.depth == self.max_time:
            return False

        for action, j_next_states in self.get_next_states(parent.j_state).items():
            for j_next_state in j_next_states:
                parent.children[action].append(TreeSearchNode(depth=parent.depth + 1, j_state=j_next_state,
                                                              action=action, parent=parent))
        return True

    def simulation(self, node):
        """
        simulate an entirely random run from the given state and return the maximal reward along the way
        :param node: the node from which we start the run
        :return: float
        """
        # max_value_found = 0
        curr_value = 0
        j_state = node.j_state
        for t in range(node.depth, self.max_time + 1):
            a = random.choice(list(self.get_next_states(j_state).keys()))
            curr_value += self.get_reward(a)
            # max_value_found = max(curr_value, max_value_found)
            if a == 'terminate':
                break
            j_state = random.choice(self.get_next_states(j_state)[a])

        return curr_value

    def backpropagation(self, node, reward):
        """
        update the node statistics to reflect the outcome of a simulation
        """
        while node is not None:
            node.n_visits += 1
            node.q_value += reward
            node = node.parent

    def best_action(self, node):
        """
        :return: the best action according to the current tree
        """
        # choose the action which leads to the set of states that were simulated most
        if node.depth == self.max_time:
            return 'terminate'

        else:
            value_action_pairs = [(self.weighted_average_value(c, explore=0.0), a) for a, c in node.children.items()]
            max_value, max_action = max(random.sample(value_action_pairs, len(value_action_pairs)), key=lambda x: x[0])
            return max_action