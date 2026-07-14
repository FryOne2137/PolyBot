from PolyEnv import GameEnv, Bardur,Imperius, Lakes
from PolyEnv import clone_with_predictions


env = GameEnv(seed=2137,players=[Imperius,Bardur],map_size=14,map_type=Lakes)

data=env.model_request_numpy()
# print(data['actions'])

for action in data:
    print(action)


print()
print()

print(data['obs']
)
print()
print()
print()
print()

# print(data['observation'])
# print(data['reward'])
env.step(0)
# print(data)

# env.clone_with_predictions()