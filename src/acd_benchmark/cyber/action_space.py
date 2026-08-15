from CybORG.Simulator.Actions import Sleep, Analyse, Remove, Restore, Monitor

from acd_benchmark.cyber.blue_table_env import HOST_NAMES

TARGET_ACTION = {
    "Analyse": Analyse,
    "Restore": Restore,
    "Remove" : Remove, 
}
ACTION_SPACE = [
    (action, hostname) for action in TARGET_ACTION for hostname in HOST_NAMES
]
ACTION_SPACE.insert(0, ("Monitor", None))
ACTION_SPACE.insert(0, ("Sleep", None))

def build_action(action_id):
    action_name, action_host = ACTION_SPACE[action_id]

    if action_name == "Sleep":
        return Sleep()
    if action_name == "Monitor":
        return Monitor(session = 0, agent = "Blue")
    action_class = TARGET_ACTION[action_name]
    return action_class(session = 0, agent = "Blue", hostname = action_host) # exp: Remove(session =0, agent = "Blue",hostnamne = action_host)

def describe_action(action_id):
    action_name, hostname = ACTION_SPACE[action_id]
    return action_name if hostname is None else f"{action_name}({hostname})"

if __name__ == '__main__':
    for i in range(len(ACTION_SPACE)):
        print(f"|{i + 1}| {ACTION_SPACE[i]}")