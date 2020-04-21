from src.game import Game
from src.visualize import visualize_val_function, visualize_reward_shaping, visualize_optimal_policy


def main():
    start = (0, 1)
    goal = (2, 3)
    game = Game(5, start=start, goal=goal, action_cost=-1)
    val_func = game.calculate_value_function()
    visualize_val_function(val_func)
    visualize_reward_shaping(val_func)
    visualize_optimal_policy(val_func)



if __name__ == '__main__' :
    main()