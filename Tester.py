"""
@author: Sam Ragusa

IMPORTANT NOTE:
-Since I am currently executing the AI checkers games from the file AI.py,
the code at the end of that file needs to be commented out before execution of these tests.  This includes all code that is not defining functions, or classes. 
"""


from Board import Board
from AI import Alpha_beta



def print_test_results(computed_outputs, desired_outputs, correctness_function=lambda a,b : a==b):
    """
    Prints the results of a test given the computed outputs, and the desired outputs (and
    if desired a correctness function comparing the outputs as equal or not).  It displays
    failed tests in an easy to understand way, and won't bother you with all the passed tests.
    """
    output_correctness = list(map(correctness_function, desired_outputs, computed_outputs))
    
    has_failed_test = False
    for j in range(len(output_correctness)):
        if not output_correctness[j]:
            print("Test number " + str(j+1) + " failed.")
            print("Calculated output: " + str(computed_outputs[j]))
            print("Desired output:  " + str(desired_outputs[j]))
            print("")
            has_failed_test = True
    
    if has_failed_test == False:
        print("All tests passed.")

def test_possible_next_moves():
    """
    Checks that the Board classes get_possible_next_moves method works properly
    by testing it against a specifically chosen set of test cases.  
    """
    test_inputs = []
    test_inputs.append([[4,1,1],[4,2,1],[5,1,2]])
    test_inputs.append([[3,2,1],[5,2,1],[6,1,2]])
    test_inputs.append([[4,2,1],[5,3,2],[6,0,2]])
    test_inputs.append([[2,3,1],[4,1,1],[4,3,1],[5,0,2],[5,3,2]])
    test_inputs.append([[2,1,1],[4,2,2],[6,1,2]])
    test_inputs.append([[3,0,2],[4,0,4],[5,0,2]])
    test_inputs.append([[3,3,1],[5,1,2],[6,1,2]])
    test_inputs.append([[2,1,1],[5,0,2],[5,1,2],[6,1,2]])
    test_inputs.append([[1,0,1],[1,1,1],[3,0,1],[3,1,3],[3,2,3],[5,0,1],[5,1,1],[5,2,1],[6,0,2]])
    test_inputs.append([[2,1,1],[2,2,3],[3,1,4],[4,1,1],[4,2,3],[7,1,1]])
    test_inputs.append([[2,1,1],[2,2,3],[3,1,2],[4,1,1],[4,2,3],[7,1,1]])
    #test_inputs.append([[1,0,1],[1,1,1],[3,0,1],[3,1,3],[3,2,3],[5,0,1],[5,1,1],[5,2,1],[6,0,4]])   commented this out because calculating it's desired output by hand takes more time than I have right now
    test_inputs.append([[2,2,1],[4,0,4]])
    
    desired_outputs = []
    desired_outputs.append([[[5,1],[3,2]],[[5,1],[3,0]]])
    desired_outputs.append([[[6,1],[5,1]],[[6,1],[5,0]]])
    desired_outputs.append([[[5,3],[4,3]],[[6,0],[5,0]]])
    desired_outputs.append([[[5,0],[3,1]],[[5,3],[3,2],[1,3]]])
    desired_outputs.append([[[4,2],[3,2]],[[4,2],[3,1]],[[6,1],[5,1]],[[6,1],[5,0]]])
    desired_outputs.append([[[3,0],[2,1]],[[3,0],[2,0]],[[5,0],[4,1]]])
    desired_outputs.append([[[5,1],[4,2]],[[5,1],[4,1]],[[6,1],[5,0]]])
    desired_outputs.append([[[5,0],[4,1]],[[5,0],[4,0]],[[5,1],[4,2]],[[5,1],[4,1]]])
    desired_outputs.append([[[6,0],[4,1],[2,2],[0,1]],[[6,0],[4,1],[2,0],[0,1]]])  
    desired_outputs.append([[[3,1],[5,2]],[[3,1],[5,0]],[[3,1],[1,2]],[[3,1],[1,0]]])
    desired_outputs.append([[[3,1],[1,2]],[[3,1],[1,0]]])
    #desired_outputs.append(TOO LONG TO DO RIGHT NOW)
    desired_outputs.append([[[4,0],[5,0]],[[4,0],[3,0]]])
    
    test_boards = [Board(the_player_turn=False) for j in range(len(test_inputs))]
    
    for j in range(len(test_boards)):
        test_boards[j].empty_board()
        test_boards[j].insert_pieces(test_inputs[j])
    
    computed_outputs = [test_board.get_possible_next_moves() for test_board in test_boards]
    
    print_test_results(computed_outputs, desired_outputs)
     
     
def test_alpha_beta_ai():
    """
    Checks that the alpha-beta pruning AI is functioning properly by computing
    the desired move for a few different implementations of alpha-beta pruning (different depths),
    and comparing it to the desired move to be outputted. 
    """
    test_inputs = []
    test_inputs.append([[1,1,4],[1,3,1],[2,3,3],[3,1,4],[4,3,1],[6,1,1],[6,3,1],[7,1,2],[7,2,4]])
    test_inputs.append([[2,0,1],[2,1,1],[2,2,3],[2,3,1],[4,1,2],[4,2,2],[4,3,2]])
    test_inputs.append([[2,1,1],[2,2,3],[2,3,1],[4,0,2],[4,1,2],[4,2,2]])
    test_inputs.append([[1,2,3],[1,3,3],[5,2,2]])
    test_inputs.append([[1,1,1],[2,1,2],[4,3,1],[5,2,2]])
    test_inputs.append([[2,2,1],[4,1,2]])
    test_inputs.append([[1,0,1],[1,2,1],[3,0,1],[3,3,1],[4,2,1],[5,0,1],[6,0,1],[7,2,3],[7,3,2]])
    
    desired_outputs = []
    desired_outputs.append([[7,2],[5,3],[3,2]])
    desired_outputs.append([[7,1],[5,0]])
    desired_outputs.append([[4,3],[3,3]])
    desired_outputs.append([[4,1],[3,0]])
    desired_outputs.append([[5,2],[4,3]])
    desired_outputs.append([[2,1],[0,2]])
    desired_outputs.append([[2,1],[0,2]])
    desired_outputs.append([[4,1],[3,0]])
    desired_outputs.append([[4,1],[3,0]])
    desired_outputs.append([[7,3],[6,3]])
    
    test_boards = [Board(the_player_turn=False) for j in range(len(test_inputs))]

    for j in range(len(test_boards)):
        test_boards[j].empty_board()
        test_boards[j].insert_pieces(test_inputs[j])
    
    alpha_betas = {
    1 : Alpha_beta(False, 1),
    2 : Alpha_beta(False, 2),
    4 : Alpha_beta(False, 4)
    }

    move_getter_instructions = [[1,2],[2],[2],[4],[1,2],[2,4],[2]]

    computed_outputs = []
    for board, instructions in zip(test_boards,move_getter_instructions):
        for instruction in instructions:
            alpha_betas.get(instruction).set_board(board)
            computed_outputs.append(alpha_betas.get(instruction).get_next_move())

    print_test_results(computed_outputs, desired_outputs)



print("Possible next move tests:")
test_possible_next_moves()
print("")
print("Alpha-beta Pruning tests:")
test_alpha_beta_ai()    
