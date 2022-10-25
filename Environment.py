import numpy as np
from ItemAndAgent import Item, Agent
from ValuesNorms import Values, Norms
from window import Window

"""

 Implementation of the Public Civility Game Environment as defined in Rodriguez-Soto et al. 'A Structural Solution to
 Sequential Moral Dilemmas' (2020). The code in this file is based on the one provided by Rodriguez-Soto et al.

"""
class Cell:
    def __init__(self, tile):
        self.tile = tile
        self.items = list()

    def get_tile(self):
        return self.tile

    def get_item(self):

        x = -1
        for item in self.items:
            if not isinstance(item, Agent):
                x = item
        return x

    def is_for_garbage(self):
        return self.tile == Environment.GC or self.tile == Environment.AC

    def is_accessible(self):
        return self.tile == Environment.AC

    def is_free(self):
        return len(self.items) == 0

    def there_is_an_agent(self):
        for item in self.items:
            if isinstance(item, Agent):
                return True
        return False

    def get_agent(self):
        for item in self.items:
            if isinstance(item, Agent):
                return item
        return 'Sorry'

    def appendate(self, item):
        self.items.append(item)

    def remove(self, item):
        if item not in self.items:
            pass
        else:
            self.items.remove(item)


class Environment:

    AC = 0  # ACCESSIBLE CELL
    IC = 1  # INACCESSIBLE CELL
    GC = 2 # GARBAGE CELL

    all_actions = range(3 * 2)

    initial_agent_left_position = [5, 1]
    initial_agent_right_position = [5, 2]

    agent_left_goal = [1, 1]
    agent_right_goal = [2, 2]



    NB_AGENTS = 1

    def __init__(self, garbage_pos=-1, is_deterministic=True, seed=-1, who_is_the_learning_agent=0):
        self.seed = seed
        self.name = 'Envious'
        self.initial_garbage_position = garbage_pos

        self.waste_basket = [2, 0]
        self.waste_basket2 = [2, 3]
        self.last_cell = [5, 2]

        self.map_tileset = create_base_map(self.waste_basket, self.waste_basket2)
        self.map = self.create_cells()

        self.nb_cells = self.translate(self.last_cell) + 2 + 1

        self.original_garbage_position = [0, 0]
        self.in_which_wastebasket = [0, 0]
        self.where_garbage = 0

        self.terminal_state_agent_left = [self.translate(Environment.agent_left_goal)]
        self.terminal_state_agent_right = [self.translate(Environment.agent_right_goal)]

        self.states_agent_left = [i for i in range(4, 12)]
        self.states_agent_right = [i for i in range(6, 12)]

        if who_is_the_learning_agent == 1:
            self.states_agent_left = self.terminal_state_agent_left + self.states_agent_left
        elif who_is_the_learning_agent == 0:
            self.states_agent_right = self.terminal_state_agent_right + self.states_agent_right

        self.states_garbage = [i for i in range(10)] + [12, 13]

        self.garbage_in_basket = False

        self.agents = self.generate_agents()

        self.items = self.generate_items(first=True)

        self.agent_succeeds = False
        self.happened_tragedy = False

        self.norm_activated = False

        self.is_deterministic = is_deterministic

        self.window = Window(self.give_window_info())

    def generate_item(self, kind, name, position, goal=None):
        """
        Creates an  item in the game, also modifying the map in order to represent it.
        :param kind: if it is an item or an agent
        :param name: the item/agent's name
        :param position: and its location in the map
        :return:
        """

        if kind == 'Item':
            item = Item(name, position)
        else:
            item = Agent(name, position, goal, self.map_clone())

        self.map[position[0], position[1]].appendate(item)

        return item

    def generate_agents(self, where_left=initial_agent_left_position, where_right=initial_agent_right_position, where_goal_left=agent_left_goal, where_goal_right=agent_right_goal):
        agents = list()
        agents.append(self.generate_item('Agent', 8005, where_left, where_goal_left))
        agents.append(self.generate_item('Agent', 2128, where_right, where_goal_right))

        return agents

    def generate_items(self, mode='hard', where_garbage=-1, first=False):
        """
        Generates all the items/agents in the game.
        :return:
        """
        items = list()

        ### It is only equal to -1 when we do not select an initial state (i.e., when learning)
        if where_garbage == -1:
            if self.initial_garbage_position != -1:
                where_garbage = self.initial_garbage_position
            else:
                where_garbage = generate_garbage(self.seed, where_garbage)

        garbage_position = generate_garbage(self.seed, where_garbage)

        if mode == 'hard':
            items.append(self.generate_item('Item', 5, garbage_position))
        if mode == 'soft':
            if len(self.items) > 0:
                items.append(self.generate_item('Item', 5, self.items[0].get_position()[:]))
            else:
                items.append(self.generate_item('Item', 5, garbage_position))

        if not first:
            garbage_position = items[0].get_position()
            if garbage_position[0] > 0:
                if garbage_position[1] == 1:
                    self.original_garbage_position[0] += 1
                    self.where_garbage = 0
                elif garbage_position[1] == 2:
                    self.original_garbage_position[1] += 1
                    self.where_garbage = 1

        return items

    def reset(self, mode='soft', where_left=initial_agent_left_position, where_right = initial_agent_right_position, where_garbage=-1):
        """
        Returns the game to its original state, before the player or another agent have changed anything.
        :return:
        """
        if where_garbage == -1:
            where_garbage = self.initial_garbage_position


        self.map = self.create_cells()
        self.items = self.generate_items(mode, where_garbage)
        self.agents = self.generate_agents(where_left, where_right)

        if mode == 'hard':
            self.garbage_in_basket = False

    def hard_reset(self, where_left=initial_agent_left_position, where_right = initial_agent_right_position, where_garbage=-1):
        """
        Returns the game to its original state, before the player or another agent have changed anything.
        :return:
        """
        self.reset('hard', where_left, where_right, where_garbage)

    def approve_move(self, move):
        """
        Checks if the move goes to an accessible cell.
        To avoid bugs it also checks if the origin of the move makes sense.
        :param move: the move to be checked
        :return: true or false.
        """

        is_damaged = False
        origin = move.get_origin()
        destiny = move.get_destination()

        ori = self.map[origin[0], origin[1]]
        dest = self.map[destiny[0], destiny[1]]

        if move.get_moved() == -1:
            return False, is_damaged

        if not ori.is_free():

            if dest.is_for_garbage():

                moved = move.get_moved()

                if isinstance(moved, Agent):

                    if dest.is_accessible():
                        "Si en su mapa pone que hay un obstaculo, que no vaya"
                        if not moved.get_map()[destiny[0], destiny[1]].is_free():
                            return False, is_damaged
                        elif dest.there_is_an_agent():
                                return False, is_damaged
                        else:
                            if not dest.is_free():
                                moved.get_damaged()
                                moved.current_damage = 1
                                is_damaged = True
                        "y si en el global pone que hay un agent, que no vaya"
                    else:
                        return False, is_damaged
                else:

                    if destiny == self.waste_basket or destiny == self.waste_basket2:
                        move.get_mover().being_civic = True
                    if dest.there_is_an_agent():  # the destiny of the object objectively
                        damaged_agent = self.map[destiny[0], destiny[1]].get_agent()
                        damaged_agent.get_damaged()
                        damaged_agent.current_damage = 1
                        is_damaged = True

                return True, is_damaged
        return False, is_damaged

    def render(self, mode='Training'):
        """
        Until this method is applied the window will not appear.
        :param mode: training or evaluating
        :return:
        """

        self.window.create(mode)

    def give_window_info(self):
        """
        Gives to the window rendering the game the initial information and the information
        that will never change.

        :return:
        """
        return self.map_tileset.copy(), self.agents[0].origin[:], self.waste_basket[:], self.waste_basket2[:], self.agents[0].goal[:], self.agents[1].goal[:]

    def update_window(self):
        """
        Gives all the needed info to the window rendering the game in order to update it.
        :return:
        """

        info = list()
        info.append(self.agents[0].get_position())

        item_pos = self.agents[1].get_position()[:]
        item_pos.append(self.agents[1].get_name())
        info.append(item_pos)

        for item in self.items:
            item_pos = item.get_position()[:]
            item_pos.append(item.get_name())
            info.append(item_pos)

        self.window.update(info)

    def drawing_paused(self):
        """
        Wrapper to know if the window rendering has been paused or not.
        :return:
        """
        return self.window.paused

    def do_move_or_not(self, move):
        """
        The method that decides if the move is ultimately approved or not.
        :param move: the move to be checked.
        :return:
        """


        we_approve, is_damaged = self.approve_move(move)

        if we_approve:



            moved = move.get_moved()
            self.remove_from_cell(move.get_origin(), moved)
            self.put_in_cell(move.get_destination(), moved)
            moved.move(move.get_destination())

            mover = move.get_mover()
            mover.tire()

        return is_damaged


    def act(self, actions):
        """
        A turn in the environment's game. See step().
        :param actions: the player's action
        :return:
        """
        for agent in self.agents:
            agent.set_map(self.map_clone())

        #shuffled = list(range(len(self.agents)))
        #np.random.shuffle(shuffled)

        move_requests = list()

        for i in range(len(self.agents)):
            while len(actions) < len(self.agents):
                actions.append(8000)

            if actions[i] >= 0:
                if actions[i] == 8000:
                    move_request = self.agents[i].act_clever(Values.TO_TRASHCAN, Norms.NO_THROWING, Norms.NO_UNCIVILITY)
                else:
                    move_request = self.agents[i].move_request(actions[i])

                move_requests.append(move_request)



        ### Here we check if the right agent is going directly upwards whereas the left agent tries to change lane,
        ### It would be unfair to give priority to the left agent's move, so instead we will change the order
        reverse_ordering = False

        if len(move_requests) == 2:
            if move_requests[0].get_destination() == move_requests[1].get_destination():
                if isinstance(move_requests[1].get_moved(), Agent):
                    if move_requests[1].get_destination()[1] == move_requests[1].get_origin()[1]:
                        reverse_ordering = True

                    ## And here we also check if there was a garbage-agent collision
                    if isinstance(move_requests[0].get_moved(), Item):
                        move_requests[1].get_moved().get_damaged()
                        move_requests[1].get_moved().current_damage = 1
                        self.happened_tragedy = True

                ## here we also check if there was a garbage-agent collision
                elif isinstance(move_requests[0].get_moved(), Agent):
                    if isinstance(move_requests[1].get_moved(), Item):
                        move_requests[0].get_moved().get_damaged()
                        move_requests[0].get_moved().current_damage = 1
                        self.happened_tragedy = True


        if reverse_ordering:
            move_requests = list(reversed(move_requests))
        ##### Order checking finished here

        for i in range(len(move_requests)):
            move_request = move_requests[i]
            self.do_move_or_not(move_request)

            #if not self.agents[i].succeeds:
            #    self.agents[i].time_increase()



    def remove_from_cell(self, pos, item):
        self.map[pos[0], pos[1]].remove(item)

    def put_in_cell(self, pos, item):
        self.map[pos[0]][pos[1]].appendate(item)



    def to_reward(self, agent, action):
        """
        Translates the current applied action into its corresponding reward.
        :return: an integer representing the reward.
        """

        reward = [0.0, 0.0]
        if not agent.succeeds:
            if self.happened_tragedy:
                if agent.damage > 0:
                    reward[0] += 0.0 # penalising for getting hurt, not in the current version
            if agent.get_position() == agent.goal:
                reward[0] += 20.0
                agent.succeeds = True
            else:
                reward[0] += -1.0

        # Norm 1

        if self.happened_tragedy:
            if agent.current_damage == 0:
                reward[1] += Norms.NO_THROWING
                self.norm_activated = True
                self.happened_tragedy = False
            else:
                agent.current_damage = 0

        # Norm 2
        if self.items[0].get_position()[0] == agent.get_position()[0] - 1:
            if self.items[0].get_position()[1] != agent.get_position()[1]:
                if action == Agent.RIGHT + Agent.NB_ACTIONS/2 or (action == Agent.LEFT and agent.get_position() != agent.previous_position):
                    if agent.get_position()[1] == 1:
                        reward[1] += Norms.NO_UNCIVILITY
                elif action == Agent.LEFT + Agent.NB_ACTIONS/2 or (action == Agent.RIGHT and agent.get_position() != agent.previous_position):
                    if agent.get_position()[1] == 2:
                        reward[1] += Norms.NO_UNCIVILITY
        elif action == Agent.UP + Agent.NB_ACTIONS/2:
            if agent.get_position()[0] == 2:
                reward[1] += Norms.NO_UNCIVILITY

        # Value 1

        if agent.being_civic:
            reward[1] += Values.TO_TRASHCAN
            #self.improving()
            agent.being_civic = False

        return reward

    def improving(self):
        pos = self.items[0].get_position()[1] ## to check if the garbage is in the wastebasket, which can only have y=1 or 3

        if pos == 3:
            pos = 1

        if pos == self.where_garbage:
           # print(pos)
            self.in_which_wastebasket[pos] += 1
        else:
            self.original_garbage_position[pos] += 1
            self.in_which_wastebasket[pos] += 1

    def civility_calculator(self):

        results = [0, 0]
        for i in range(len(self.in_which_wastebasket)):

            if self.original_garbage_position[i] == 0:
                results[i] = -1.0
            else:
                results[i] = float(self.in_which_wastebasket[i])/float(self.original_garbage_position[i])
        return results

    def get_state(self):
        """
        Wrapper to get the needed information of the state to the q-learning algorithm.
        :return:
        """
        stator = list()

        for item in self.agents:
            stator.append(self.translate(item.get_position()))

        for item in self.items:
            stator.append(self.translate(item.get_position()))

        return stator

    def step(self, actions):
        """
        Produces a step in the game, letting every agent try to act consecutively.
        :param action: the action that the player will perform
        :return: the information needed for the q-learning algorithm, copying the GYM style.
        """
        self.act(actions)

        rewards = list()
        dones = list()

        for agent, action in zip(self.agents, actions):

            reward = self.to_reward(agent, action)

            done = False
            if agent.get_position() == agent.goal:
                done = True

            rewards.append(reward)
            dones.append(done)

        # rewards[0] only takes rewards from left agent
        return self.get_state(), rewards, dones


    def set_stats(self, episode, r_big, mean_score, fourth=0, fifth=0):
        self.window.stats = episode, r_big, mean_score, fourth, fifth

    def eval_stats(self):

        mean_time = 0
        mean_tiredness = 0
        mean_damage = 0

        n = 0
        for agent in self.agents:
            n += 1
            mean_time += agent.time_taken
            mean_tiredness += agent.tiredness
            if self.norm_activated:
                mean_damage += 1
                self.norm_activated = False

        mean_time /= n
        mean_tiredness /= n
        mean_damage /= n

        civility = 0

        pos = self.items[0].get_position()
        if pos == self.waste_basket or pos == self.waste_basket2:
            civility = 1

        return mean_time, mean_tiredness, mean_damage, civility

    def create_cells(self):

        map_struct = list()
        for i in range(len(self.map_tileset)):
            map_struct.append(list())
            for j in range(len(self.map_tileset[0])):
                map_struct[i].append(Cell(self.map_tileset[i, j]))

        return np.array(map_struct)

    def map_clone(self):
        map_struct = list()
        for i in range(len(self.map_tileset)):
            map_struct.append(list())
            for j in range(len(self.map_tileset[0])):
                cell_created = Cell(self.map_tileset[i, j])
                cell_created.items = self.map[i, j].items[:]
                map_struct[i].append(cell_created)

        return np.array(map_struct)

    def translate(self, pos):
        """
        A method to simplify the state encoding for the q-learning algorithms. Transforms a map 2-dimensional location
        into a 1-dimensional one.
        :param pos: the position in the map (x, y)
        :return: an integer
        """

        counter = 0
        for i in range(self.map.shape[0]):
            for j in range(1, 3):
                if self.map[i, j].tile != Environment.IC:
                    if i == pos[0] and j == pos[1]:
                        return counter
                    counter += 1

        if pos[0] == self.waste_basket[0] and pos[1] == self.waste_basket[1]:
            return counter
        elif pos[0] == self.waste_basket2[0] and pos[1] == self.waste_basket2[1]:
            return counter + 1

        print("This should never occur")
        return counter

    def translate_state_cell(self, cell):

        pos = [0, 0]

        if cell < 12:
            pos[0] = int(cell//2)
            if cell % 2 == 0:
                pos[1] = 1
            else:
                pos[1] = 2
        elif cell == 12:
            pos = self.waste_basket
        elif cell == 13:
            pos = self.waste_basket2

        return pos

    def translate_state(self, cell_left, cell_right, cell_garbage):

        pos_left = self.translate_state_cell(cell_left)
        pos_right = self.translate_state_cell(cell_right)
        pos_garbage = self.translate_state_cell(cell_garbage)

        return pos_left, pos_right, pos_garbage


def create_base_map(waste_basket, waste_basket2):
    """
    The numpy array representing the map.  Change it as you consider it.
    :return:
    """
    base_map = np.array([
        [Environment.IC, Environment.GC, Environment.GC, Environment.IC],
        [Environment.IC, Environment.AC, Environment.GC, Environment.IC],
        [Environment.IC, Environment.AC, Environment.AC, Environment.IC],
        [Environment.IC, Environment.AC, Environment.AC, Environment.IC],
        [Environment.IC, Environment.AC, Environment.AC, Environment.IC],
        [Environment.IC, Environment.AC, Environment.AC, Environment.IC],
        [Environment.IC, Environment.IC, Environment.IC, Environment.IC]])

    base_map[waste_basket[0], waste_basket[1]] = Environment.GC
    base_map[waste_basket2[0], waste_basket2[1]] = Environment.GC

    return base_map


def generate_garbage(seed=-1, where_garbage=-1):

    if where_garbage != -1:
        return where_garbage

    possible_points = [[4, 1], [4, 2]]
    where = np.random.randint(len(possible_points))

    if seed > -1:
        return possible_points[seed]

    return possible_points[where]
