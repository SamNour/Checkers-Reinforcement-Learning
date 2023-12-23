'''
Created on Dec 3, 2016

@author: Sam Ragusa
'''
import numpy as np
import random
import json
from ast import literal_eval
from Board import Board
import matplotlib.pyplot as plt

class Player:
    """
    A class to be inherited by any class representing a checkers player.
    This is used so that other functions can be written for more general use,
    without worry of crashing (e.g. play_n_games).
    
    NOTES:
    1) Create set playerID method
    """
    
    def set_board(self, the_board):
        """
        Sets the Board object which is known by the AI.
        """
        self.board = the_board
    
    def game_completed(self):
        """
        Should be overridden if AI implementing this class should be notified 
        of when a game ends, before the board is wiped.
        """
        pass
    
    def get_next_move(self):
        """
        Gets the desired next move from the AI.
        """
        pass


def reward_function(state_info1, state_info2):
    """
    Reward for transitioning from state with state_info1 to state with state_info2.
    
    NOTE:
    1) do something better with where/how this is implemented
    2) should give some kind of negative for tieing
    """
    if state_info2[1] == 0 and state_info2[3] == 0:
        return 12
    if state_info2[0] == 0 and state_info2[2] == 0:
        return -12
    return state_info2[0]-state_info1[0] + 2*(state_info2[2]-state_info1[2])-(state_info2[1]-state_info1[1])-2*(state_info2[3]-state_info1[3])


class Q_Learning_AI(Player):
    """
    TO-DO:
    1) add ability to not train when wanted in a more efficient way
        A) when not training also don't look for/add currently unknown states
        B) do this by having an instance variable saying if it's testing or training
        C) and let a function set that parameter
    2) handle the rewards function which is coded as if the function were already defined
    """


    def __init__(self, the_player_id, the_learning_rate, the_discount_factor, info_location=None, the_random_move_probability=0, the_board=None):
        """
        Initialize the instance variables to be stored by the AI. 
        """
        self.random_move_probability = the_random_move_probability
        self.learning_rate = the_learning_rate    
        self.discount_factor = the_discount_factor
        self.player_id = the_player_id
        self.board = the_board
        self.pre_last_move_state = None
        self.post_last_move_state = None 
        if not info_location is None:
            self.load_transition_information(info_location)
        else:
            self.transitions = {}

    def set_random_move_probability(self, probability):
        """
        Sets the random move probability for the AI.
        """
        self.random_move_probability = probability


    def set_learning_rate(self, the_learning_rate):
        """
        Sets the learning rate for the AI.
        """
        self.learning_rate = the_learning_rate
    

    def get_states_from_boards_positions(self, game_board_positions):
        """
        Gets an array of tuples from the given set of board spots,
        each tuple representing the characteristics which define the
        state the board is in. 

        Format of returned data:
        a lower vertical center of mass could indicate that a player's pieces are more defensive, 
        while a higher vertical center of mass could indicate a more aggressive or advanced position.
        [(own_pieces, opp_pieces, own_kings, OP_kings, own_edges, MY_V_COM, OP_V_COM), ...]
        """
        def is_own_piece(piece):
            return (self.player_id and (piece == 1 or piece == 3)) or (not self.player_id and (piece == 2 or piece == 4))

        def is_edge_piece(i, j):
            return (i==0 and j%2==0) or (i==3 and j%2==1)

        def calculate_center_of_mass(piece_counters):
            for k in range(len(piece_counters)):
                own_pieces = piece_counters[k][0] + piece_counters[k][2]
                opp_pieces = piece_counters[k][1] + piece_counters[k][3]

                if own_pieces != 0:
                    average_own_piece_position = piece_counters[k][5] / own_pieces
                    piece_counters[k][5] = int(average_own_piece_position)
                else:
                    piece_counters[k][5] = 0

                if opp_pieces != 0:
                    average_opp_piece_position = piece_counters[k][6] / opp_pieces
                    piece_counters[k][6] = int(average_opp_piece_position)
                else:
                    piece_counters[k][6] = 0

        piece_counters = [[0,0,0,0,0,0,0] for _ in range(len(game_board_positions))]

        for k, board in enumerate(game_board_positions):
            for j, row in enumerate(board):
                for i, piece in enumerate(row):
                    if piece != 0:
                        piece_counters[k][piece-1] += 1
                        if is_own_piece(piece):
                            if is_edge_piece(i, j):
                                piece_counters[k][4] += 1
                            piece_counters[k][5] += j
                        else:
                            piece_counters[k][6] += j

        calculate_center_of_mass(piece_counters)

        return [tuple(counter) for counter in piece_counters]
                                 

    def get_optimal_transition_between_states(self, possible_state_array, initial_transition_value=10):#%%%%%%%%%%%%%%%%%% FOR (1)
        """
        Gets the desired transition to taken for the current board configuration.
        If any possible transition does not exist, it will create it.
        """
        # Get the current state of the board
        cur_state = tuple(self.get_states_from_boards_positions([self.board.spots])[0])

        # Initialize an empty dictionary to store processed transitions
        done_transitions = {}

        # Iterate over each possible state
        for state in possible_state_array:
            # Convert the state to a tuple
            state_tuple = tuple(state)

            # Create a tuple representing the transition from the current state to the possible state
            transition = (cur_state, state_tuple)

            # If the transition has not been processed before
            if transition not in done_transitions:
                # If the transition does not exist in self.transitions, add it with an initial value
                if transition not in self.transitions:
                    self.transitions[transition] = initial_transition_value

                # Add the transition and its value to done_transitions
                done_transitions[transition] = self.transitions[transition]
        """
        When it is the reinforcement learning AI's turn, it looks at a list of possible moves it could make, 
        and if their respective state transitions are not yet known, it adds them to the dictionary of 
        transitions. Then it will chose to do one of two things, based on a specified probability. 
        The first is that it could chose a move at random, this is used during training so that transitions 
        other then the current highest valued transition will still be trained. If the learning AI does not 
        move randomly, it will chose the move who's state transition has the highest value associated with it.

        "state transitions" are the pairs of initial and final states resulting from an action. 
        They are often used in reinforcement learning algorithms to learn which actions are beneficial 
        in which states, by associating rewards with transitions.
        """
        if random != 0 and random.random() < self.random_move_probability:
            try:
                return list(done_transitions.keys())[random.randint(0, len(done_transitions)-1)]
            except:   
                return []
    
        try:
            reverse_dict = {j:i for i,j in done_transitions.items()}
            return reverse_dict.get(max(reverse_dict))
        except:
            return []    
   
   
    def game_completed(self):
        """
        Update self.transitions with a completed game before the board
        is cleared.
        """
        cur_state = self.get_states_from_boards_positions([self.board.spots])[0]
        transition = (self.pre_last_move_state ,self.post_last_move_state)
        # ! the game is terminal so the reward is the value of the final state
        # T(S_n, S_{n+1}) = T(S_n, S_{n+1}) + alpha * (R(S_n, S_{n+1}))
        self.transitions[transition] = self.transitions[transition] + self.learning_rate * reward_function(transition[0],cur_state)

        self.pre_last_move_state = None
        self.post_last_move_state = None



    def get_transitions_information(self):
        """
        Get an array of of information about the dictionary self.transitions .
        It returns the information in the form:
        # ! Transition formate
        [num_transitions, num_start_of_transitions, avg_value, max_value, min_value]
        
        NOTES:
        1) Should use a dictionary here so this runs much faster 
        """
        start_of_transitions = {}
        max_value = float("-inf")
        min_value = float("inf")
        total_value = 0
        for k,v in self.transitions.items():
            if start_of_transitions.get(k[0]) is None:
                start_of_transitions.update({k[0]:0})
            #if k[0] not in start_of_transitions:
                #start_of_transitions.append(k[0])
            if v > max_value:
                max_value = v
            if v < min_value:
                min_value = v
            total_value = total_value + v
            
        return [len(self.transitions), len(start_of_transitions), float(total_value/len(self.transitions)), max_value, min_value]
    
    
    def print_transition_information(self, info):
        """
        Prints the output of get_transitions_information in a easy to understand format.
        """
        print("Total number of transitions: ".ljust(35), info[0])        
        print("Total number of visited states: ".ljust(35), info[1])
        print("Average value for transition: ".ljust(35), info[2])
        print("Maximum value for transition: ".ljust(35), info[3])
        print("Minimum value for transition: ".ljust(35), info[4])
    
        
    def save_game_transitions_to_file(self, file_name="data.json"):
        """
        Saves the current transitions information to a specified
        json file. 
        """
        with open(file_name, 'w') as fp:
            json.dump({str(k): v for k,v in self.transitions.items()}, fp, indent= 4)
        
        
    def load_transition_information(self, file_name):
        """
        Loads transitions information from a desired json file.
        """
        with open(file_name, 'r') as fp:
            self.transitions = {literal_eval(k): v for k,v in json.load(fp).items()}
        
        
    def get_max_future_state_value(self, depth):
        """
        Look ahead a given number of moves and return the maximal value associated 
        with a move of that depth. 
        
        STRATEGY:
        1) Look forward in (actual) own transition states.  
        2) Look at board as self being the opponent and look forward in that situations transition states
        3) If not at depth go back to step (1)
        
        TODO:
        1) Approach this with algorithm similar to how minimax works
            a) look for set of transitions from (I think) current state of length depth by doing minimax
            b) Might also use alpha-beta pruning
            
        NOTES:
        1) depth is not actually looking ahead in possible moves, but actually simulating something similar (hopefully similar)
        2) ONLY WORKS FOR DEPTH OF 1 RIGHT NOW
        """
        answer = float("-inf")
        cur_state = self.get_states_from_boards_positions([self.board.spots])[0]
        for k,v in self.transitions.items():
            if v > answer and k[0] == cur_state:
                answer = v
        
        if answer == float("-inf"):
            return None
        return answer



    def get_next_move(self):#, new_board):
        """
        NOTES:
        If the variable names are confusing, think about them being named when you just call the method.
        
        PRECONDITIONS:
        1)  The board exists and is legal
        self.pre_last_move_state = self.get_states_from_boards_positions([self.board.spots])[0]: 
        This line is getting the current state of the game board and storing it as the "pre-last-move" state. This state represents the board configuration before the AI makes its move.
        """
        if self.pre_last_move_state is not None:#%%%%%%%%%%%%%%%%%%%%%%% FOR (1)
            cur_state = self.get_states_from_boards_positions([self.board.spots])[0]
    
            transition = (self.pre_last_move_state ,self.post_last_move_state)
            try:# self.transitions.get(transition) is not None:#%%%%%%%%%%%%%%%%%%%%%%%%%%%% FOR (1)
                max_future_state = self.get_max_future_state_value(1)
                # ! T(S_n, S_{n+1}) = T(S_n, S_{n+1}) + alpha * (R(S_n, S_{n+1}) + gamma * max_a(T(S_{n+1}, S_{n+2})) - T(S_n, S_{n+1}))
                self.transitions[transition] = self.transitions[transition] + self.learning_rate * (reward_function(transition[0],cur_state)+ self.discount_factor* max_future_state - self.transitions[transition])
            except:#%%%%%%%%%%%%%%%%%%%%%%%%%%%% FOR (1)
                self.transitions[transition] = self.transitions[transition] + self.learning_rate * (reward_function(transition[0],cur_state))
        
        # get the current board state
        self.pre_last_move_state = self.get_states_from_boards_positions([self.board.spots])[0]#%%%%%%%%%%%%%%%%%%%%%%%%%%%% FOR (1)
        # This line is getting all possible moves that the AI could make from the current board state.
        possible_next_moves = self.board.get_possible_next_moves_from_board()
        # This line is calculating the potential next states of the game board that would result from each of the possible moves.
        possible_next_states = self.get_states_from_boards_positions(self.board.get_potential_spots_from_moves(possible_next_moves))
        # This line is determining the most desirable next state according to the AI's learned values. This state is stored as the 
        # "post-last-move" state, representing the desired board configuration after the AI makes its move.
        self.post_last_move_state = self.get_optimal_transition_between_states(possible_next_states)[1]   
        
        considered_moves = []
        # The for loop is creating a list of all moves that would lead to the post-last-move state. 
        # It does this by iterating over all possible next states and checking if each one matches 
        # the post-last-move state. If a match is found, the corresponding move is added to the considered_moves list.
        for j in range(len(possible_next_states)):
            if tuple(possible_next_states[j]) == self.post_last_move_state:
                considered_moves.append(possible_next_moves[j])
                
        # Pick a random move from the considered moves
        return considered_moves[random.randint(0,len(considered_moves)-1)]

def get_number_of_pieces_and_kings(spots, player_id=None):
    """
    Gets the number of pieces and the number of kings that each player has on the current 
    board configuration represented in the given spots. The format of the function with defaults is:
    [P1_pieces, P2_pieces, P1_kings, P2_kings]
    and if given a player_id:
    [player_pieces, player_kings]
    """
    piece_counter = [0,0,0,0]  
    for row in spots:
        for element in row:
            if element != 0:
                piece_counter[element-1] = piece_counter[element-1] + 1
    
    if player_id == True:
        return [piece_counter[0], piece_counter[2]]
    elif player_id == False:
        return [piece_counter[1], piece_counter[3]]
    else:
        return piece_counter
    

class Alpha_beta(Player):
    """
    A class representing a checkers playing AI using Alpha-Beta pruning.   
    
    TO DO:
    1) Be able to take in any reward function (for when not win/loss) 
    so that you can make a more robust set of training AI
    """
    
    def __init__(self, the_player_id, the_depth, the_board=None):
        """
        Initialize the instance variables to be stored by the AI. 
        """
        self.board = the_board
        self.depth = the_depth
        self.player_id = the_player_id

    def alpha_beta(self, board, depth, alpha, beta, maximizing_player):
        """
        A method implementing alpha-beta pruning to decide what move to make given 
        the current board configuration. 
        """
        if board.is_game_over():
            if get_number_of_pieces_and_kings(board.spots, board.player_turn) == [0,0]:
                if maximizing_player:
                    #Using integers instead of float("inf") so it's less than float("inf") not equal to
                    return -10000000, None
                else:
                    return 10000000, None
            elif get_number_of_pieces_and_kings(board.spots, not board.player_turn) == [0,0]:
                if maximizing_player:
                    return 1000000, None
                else:
                    return -1000000, None
            else:
                return 0, None

        if depth == 0:
            players_info = get_number_of_pieces_and_kings(board.spots)
            if board.player_turn != maximizing_player:
                return  players_info[1] + 2 * players_info[3] - (players_info[0] + 2 * players_info[2]), None
            return  players_info[0] + 2 * players_info[2] - (players_info[1] + 2 * players_info[3]), None
        possible_moves = board.get_possible_next_moves_from_board()

        potential_spots = board.get_potential_spots_from_moves(possible_moves)
        desired_move_index = None
        if maximizing_player:
            v = float('-inf')
            for j in range(len(potential_spots)):
                cur_board = Board(potential_spots[j], not board.player_turn)
                alpha_beta_results = self.alpha_beta(cur_board, depth - 1, alpha, beta, False)
                if v < alpha_beta_results[0]: 
                    v = alpha_beta_results[0]
                    alpha = max(alpha, v)
                    desired_move_index = j
                if beta <= alpha: 
                    break
            if desired_move_index is None:
                return v, None
            return v, possible_moves[desired_move_index]
        else:
            v = float('inf')
            for j in range(len(potential_spots)):
                cur_board = Board(potential_spots[j], not board.player_turn)
                alpha_beta_results = self.alpha_beta(cur_board, depth - 1, alpha, beta, True)
                if v > alpha_beta_results[0]:  
                    v = alpha_beta_results[0]
                    desired_move_index = j
                    beta = min(beta, v)
                if beta <= alpha:
                    break
            if desired_move_index is None:
                return v, None
            return v, possible_moves[desired_move_index]
    
    def get_next_move(self):
        return self.alpha_beta(self.board, self.depth, float('-inf'), float('inf'), self.player_id)[1]
        

def play_n_games(player1, player2, num_games, move_limit):
    """
    Plays a specified amount of games of checkers between player1, who goes first,
    and player2, who goes second.  The games will be stopped after the given limit on moves.
    This function outputs an array of arrays formatted as followed (only showing game 1's info):
    [[game1_outcome, num_moves, num_own_pieces, num_opp_pieces, num_own_kings, num_opp_kings]...]
    gameN_outcome is 0 if player1 won, 1 if lost, 2 if tied, and 3 if hit move limit.
    
    PRECONDITIONS:
    1)Both player1 and player2 inherit the Player class
    2)Both player1 and player2 play legal moves only
    """
    game_board = Board()
    player1.set_board(game_board)
    player2.set_board(game_board)
     
    players_move = player1
    outcome_counter = [[-1,-1,-1,-1,-1,-1] for j in range(num_games)] 
    for j in range(num_games):
        #print(j)
        move_counter = 0
        while not game_board.is_game_over() and move_counter < move_limit:
            game_board.make_move(players_move.get_next_move())
             
            move_counter = move_counter + 1
            if players_move is player1:
                players_move = player2
            else:
                players_move = player1
        else:
            piece_counter = get_number_of_pieces_and_kings(game_board.spots)
            if piece_counter[0] != 0 or piece_counter[2] != 0:
                if piece_counter[1] != 0 or piece_counter[3] != 0:
                    if move_counter == move_limit:
                        outcome_counter[j][0] = 3
                    else:
                        outcome_counter[j][0] = 2
#                     if (j+1)%100==0:
#                         print("Tie game for game #" + str(j + 1) + " in " + str(move_counter) + " turns!")
                else:
                    outcome_counter[j][0] = 0
#                     if (j+1)%100==0:
#                         print("Player 1 won game #" + str(j + 1) + " in " + str(move_counter) + " turns!")
            else:
                outcome_counter[j][0] = 1
#                 if (j+1)%100==0:
#                     print("Player 2 won game #" + str(j + 1) + " in " + str(move_counter) + " turns!")
                
            outcome_counter[j][1] = move_counter
            outcome_counter[j][2] = piece_counter[0]
            outcome_counter[j][3] = piece_counter[1]
            outcome_counter[j][4] = piece_counter[2]
            outcome_counter[j][5] = piece_counter[3]
             
            player1.game_completed()
            player2.game_completed()
            #game_board.print_board()
            game_board.reset_board()
     
    return outcome_counter


def pretty_outcome_display(outcomes):
    """
    Prints the outcome of play_n_games in a easy to understand format.
    
    TO DO:
    1) Add functionality for pieces in each game
    2) Add ability to take other strings for AI info and display it
    """
    game_wins = [0,0,0,0]
    total_moves = 0
    max_moves_made = float("-inf")
    min_moves_made = float("inf")
    for outcome in outcomes:
        total_moves = total_moves + outcome[1]
        if outcome[1] < min_moves_made:
            min_moves_made = outcome[1]
        if outcome[1] > max_moves_made:
            max_moves_made = outcome[1]
        
        game_wins[outcome[0]] = game_wins[outcome[0]] + 1
    
    print("Games Played: ".ljust(35), len(outcomes))
    print("Player 1 wins: ".ljust(35), game_wins[0])
    print("Player 2 wins: ".ljust(35), game_wins[1])
    print("Games exceeded move limit: ".ljust(35), game_wins[3])
    print("Games tied: ".ljust(35), game_wins[2])
    print("Total moves made: ".ljust(35), total_moves)  
    print("Average moves made: ".ljust(35), total_moves/len(outcomes))
    print("Max moves made: ".ljust(35), max_moves_made)
    print("Min moves made: ".ljust(35), min_moves_made)
    
    
def plot_end_game_information(outcome, interval, title="End of Game Results"):
    """
    Plots the end game results.
    """
    # Calculate the number of intervals
    num_intervals = len(outcome) // interval

    # Initialize result lists
    player1_wins = [0] * num_intervals
    player2_wins = [0] * num_intervals
    ties = [0] * num_intervals
    move_limit = [0] * num_intervals

    # Iterate over each interval
    for j in range(num_intervals):
        # Iterate over each game in the interval
        for i in range(interval):
            game_result = outcome[j*interval + i][0]

            # Update the appropriate result list based on the game result
            if game_result == 0:
                player1_wins[j] += 1
            elif game_result == 1:
                player2_wins[j] += 1
            elif game_result == 2:
                ties[j] += 1
            else:
                move_limit[j] += 1

    bar_width = 0.2
    intervals = np.arange(num_intervals)

    plt.figure(title)

    plt.bar(intervals - bar_width*1.5, player1_wins, bar_width, label="Player 1 wins")
    plt.bar(intervals - bar_width/2, player2_wins, bar_width, label="Player 2 wins")
    plt.bar(intervals + bar_width/2, ties, bar_width, label="Ties")
    plt.bar(intervals + bar_width*1.5, move_limit, bar_width, label="Move limit reached")

    # Set the labels
    plt.ylabel(f"Occurrence per {interval} games")
    plt.xlabel("Interval")

    # Add a legend
    plt.legend()

    plt.show()


 
LEARNING_RATE = .005  
DISCOUNT_FACTOR = .3
NUM_GAMES_TO_TRAIN = 100
NUM_TRAINING_ROUNDS = 25
NUM_VALIDATION_GAMES = 5
NUM_GAMES_TO_TEST = 0
TRAINING_RANDOM_MOVE_PROBABILITY = .25
ALPHA_BETA_DEPTH = 2
TRAINING_MOVE_LIMIT = 500
VALIDATION_MOVE_LIMIT = 1000
TESTING_MOVE_LIMIT = 2000
PLAYER1 = Q_Learning_AI(True, LEARNING_RATE, DISCOUNT_FACTOR, the_random_move_probability=TRAINING_RANDOM_MOVE_PROBABILITY)#, info_location="data.json")
PLAYER2 = Alpha_beta(False, ALPHA_BETA_DEPTH)
#PLAYER3 = Alpha_beta(False, 1)
PLAYER4 = Alpha_beta(False, 3)
# PLAYER5 = Q_Learning_AI(False, LEARNING_RATE, DISCOUNT_FACTOR, the_random_move_probability=TRAINING_RANDOM_MOVE_PROBABILITY)
 
  
#PLAYER1.print_transition_information(PLAYER1.get_transitions_information())
 
training_info = []
validation_info = []
for j in range(NUM_TRAINING_ROUNDS):
    training_info.extend(play_n_games(PLAYER1, PLAYER2, NUM_GAMES_TO_TRAIN, TRAINING_MOVE_LIMIT))
    PLAYER1.print_transition_information(PLAYER1.get_transitions_information())
    PLAYER1.set_random_move_probability(0)
    PLAYER1.set_learning_rate(0)
    validation_info.extend(play_n_games(PLAYER1, PLAYER4, NUM_VALIDATION_GAMES, VALIDATION_MOVE_LIMIT))
    print("Round " + str(j+1) + " completed!")
    PLAYER1.set_random_move_probability(TRAINING_RANDOM_MOVE_PROBABILITY)
    PLAYER1.set_learning_rate(LEARNING_RATE)
    print("")
    PLAYER1.print_transition_information(PLAYER1.get_transitions_information())
    print("")
    PLAYER1.save_game_transitions_to_file()
 
    
plot_end_game_information(training_info, 200, "Training Information")
plot_end_game_information(validation_info, NUM_VALIDATION_GAMES, "Validation Information")
plt.show()
 
pretty_outcome_display(training_info)
print("")
pretty_outcome_display(validation_info)
  
"""
 
PLAYER1.set_random_move_probability(0)
pretty_outcome_display(play_n_games(PLAYER1, PLAYER2, NUM_GAMES_TO_TEST, TESTING_MOVE_LIMIT))
PLAYER1.print_transition_information(PLAYER1.get_transitions_information())
print(" ")
pretty_outcome_display(play_n_games(PLAYER1, PLAYER3, NUM_GAMES_TO_TEST, TESTING_MOVE_LIMIT))
PLAYER1.print_transition_information(PLAYER1.get_transitions_information())
print(" ")
pretty_outcome_display(play_n_games(PLAYER1, PLAYER4, NUM_GAMES_TO_TEST, TESTING_MOVE_LIMIT))
PLAYER1.print_transition_information(PLAYER1.get_transitions_information())
PLAYER1.save_game_transitions_to_file()
 
"""
 

