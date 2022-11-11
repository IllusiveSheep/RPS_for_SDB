import random
from enum import IntEnum

settings = "normal"
hard_settings = 0
answer_user = 0


class Action(IntEnum):
    Rock = 0
    Paper = 1
    Scissors = 2


victories = {
    Action.Scissors: [Action.Paper],
    Action.Paper: [Action.Rock],
    Action.Rock: [Action.Scissors]
}


def get_user_selection():
    # choices = [f"{action.name}[{action.value}]" for action in Action]
    # choices_str = ", ".join(choices)
    selection = answer_user
    action = Action(selection)
    return action


def get_computer_selection():
    selection = random.randint(0, len(Action) - 1)
    action = Action(selection)
    return action


def determine_winner(user_action, computer_action):
    computer_action = random.randint(0, 2)
    action = Action(computer_action)
    computer_action = action
    defeats = victories[user_action]
    if user_action == computer_action:
        print(f"Оба игрока выбрали {user_action.name}. Это ничья!")
    elif computer_action in defeats:
        print(f"{user_action.name} бьет {computer_action.name}! Вы победили!")
    else:
        print(f"{computer_action.name} бьет {user_action.name}! Вы проиграли. :C")
    if settings == 'hard':
        return hard_settings
    else:
        return normal_settings
