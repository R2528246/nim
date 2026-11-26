import random
import sys


class Nim:
    def __init__(self):
        self.piles = [1, 3, 5, 7]
        self.player = 0
        self.winner = None

    @staticmethod
    def available_actions(piles):
        """Return all legal (pile, count) pairs for the given piles."""
        actions = set()
        for pile, count in enumerate(piles):
            for number in range(1, count + 1):
                actions.add((pile, number))
        return actions

    def switch_player(self):
        self.player = 1 - self.player

    def move(self, action):
        """Make the given move, if valid, and update the game state."""
        pile, count = action

        if self.winner is not None:
            raise ValueError("Game already won")
        if action not in self.available_actions(self.piles):
            raise ValueError("Invalid move")

        self.piles[pile] -= count

        if all(pile_size == 0 for pile_size in self.piles):
            self.winner = self.player
        else:
            self.switch_player()


class NimAI:
    def __init__(self, alpha=0.5, epsilon=0.1):
        self.q = {}
        self.alpha = alpha
        self.epsilon = epsilon

    def get_q_value(self, state, action):
        """Return the Q-value for a state/action pair, defaulting to 0."""
        return self.q.get((tuple(state), action), 0.0)

    def update_q_value(self, state, action, old_q, reward, future_rewards):
        """Update a Q-value using the Q-learning update formula."""
        new_q = old_q + self.alpha * (reward + future_rewards - old_q)
        self.q[(tuple(state), action)] = new_q

    def best_future_reward(self, state):
        """Return the best Q-value for any legal action in this state."""
        possible_actions = Nim.available_actions(state)
        if not possible_actions:
            return 0.0
        return max(self.get_q_value(state, action) for action in possible_actions)

    def update(self, old_state, action, new_state, reward):
        """Update Q-values after moving from old_state to new_state."""
        old_q = self.get_q_value(old_state, action)
        future_rewards = self.best_future_reward(new_state)
        self.update_q_value(
            old_state, action, old_q, reward, future_rewards
        )

    def choose_action(self, state, epsilon=True):
        """Choose a legal action using epsilon-greedy selection."""
        possible_actions = Nim.available_actions(state)
        if not possible_actions:
            return None

        if epsilon and random.random() < self.epsilon:
            return random.choice(tuple(possible_actions))

        best_q = max(
            self.get_q_value(state, action) for action in possible_actions
        )
        best_actions = [
            action for action in possible_actions
            if self.get_q_value(state, action) == best_q
        ]
        return random.choice(best_actions)


def train(n):
    """Train an AI by playing n games against itself."""
    ai = NimAI()

    for _ in range(n):
        game = Nim()

        last = {
            0: {"state": None, "action": None},
            1: {"state": None, "action": None}
        }

        while True:
            state = game.piles.copy()
            current_player = game.player
            action = ai.choose_action(state)

            last[current_player]["state"] = state
            last[current_player]["action"] = action

            game.move(action)
            new_state = game.piles.copy()

            if game.winner is not None:
                ai.update(state, action, new_state, 1)
                previous = last[1 - current_player]
                ai.update(
                    previous["state"],
                    previous["action"],
                    new_state,
                    -1
                )
                break

            previous = last[1 - current_player]
            if previous["state"] is not None:
                ai.update(
                    previous["state"],
                    previous["action"],
                    new_state,
                    0
                )

    return ai


def play(ai, human_player=None):
    """Play a game of Nim against the trained AI."""
    game = Nim()

    while True:
        print()
        print("Piles:")
        for i, pile in enumerate(game.piles):
            print(f"Pile {i}: {'|' * pile}")

        if game.player == human_player:
            print("\nYour turn.")
            while True:
                try:
                    pile = int(input("Choose a pile: "))
                    count = int(input("Choose a number of objects to remove: "))
                    action = (pile, count)
                    if action in Nim.available_actions(game.piles):
                        break
                    print("Invalid move. Try again.")
                except (ValueError, EOFError):
                    print("Please enter valid numbers.")
        else:
            print("\nAI's turn.")
            action = ai.choose_action(game.piles, epsilon=False)
            print(f"AI chooses to take {action[1]} from pile {action[0]}.")

        game.move(action)

        if game.winner is not None:
            print()
            print("GAME OVER")
            if game.winner == human_player:
                print("You win!")
            else:
                print("AI wins!")
            return


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python nim.py number_of_training_games")

    try:
        number_of_games = int(sys.argv[1])
        if number_of_games <= 0:
            raise ValueError
    except ValueError:
        sys.exit("Please provide a positive integer number of games.")

    print(f"Playing against AI trained for {number_of_games} games...")
    ai = train(number_of_games)
    play(ai, human_player=0)
