"""Inspect a public game observation without exposing hidden ground truth."""

from PolyEnv import Bardur, GameEnv, Imperius


def main():
    engine = GameEnv(seed=2137, map_size=11, players=(Bardur, Imperius), map_type="lakes")
    observation = engine.model_request_numpy()
    actions = observation["actions"]["action_id"]
    print({"public_map_shape": observation["map_tokens"].shape,
           "legal_actions": len(actions)})
    if len(actions):
        result = engine.step_fast(int(actions[0]))
        print({"first_legal_action_result": result})


if __name__ == "__main__":
    main()
