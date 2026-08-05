import numpy as np

from acd_benchmark.cyber.env_factory import make_cc2_env
from acd_benchmark.wrappers.shared.flat_obs import encode_flat_observation
from acd_benchmark.wrappers.shared.graph_obs import build_graph_encoder, encode_graph_observation
from acd_benchmark.wrappers.shared.partial_obs import mask_dynamic_features, mask_feature_groups
from CybORG.Agents import BlueReactRemoveAgent


def test_raw_blue_obs_graph_and_flat_share_host_features():
    cyborg = make_cc2_env()
    result = cyborg.reset("Blue")

    encoder = build_graph_encoder(result.observation)
    graph_obs = encode_graph_observation(encoder, result.observation)
    flat_obs = encode_flat_observation(graph_obs)

    num_hosts, feature_dim = graph_obs.node_features.shape
    node_part = flat_obs[: num_hosts * feature_dim].reshape(num_hosts, feature_dim)

    assert num_hosts == 13
    assert feature_dim == len(graph_obs.feature_names)
    assert graph_obs.edge_index.shape[0] == 2
    assert graph_obs.adjacency.shape == (num_hosts, num_hosts)
    assert flat_obs.shape == (num_hosts * feature_dim + num_hosts * num_hosts,)
    np.testing.assert_array_equal(node_part, graph_obs.node_features)


def test_sparse_step_observation_preserves_host_schema():
    cyborg = make_cc2_env()
    blue_agent = BlueReactRemoveAgent()
    result = cyborg.reset("Blue")
    encoder = build_graph_encoder(result.observation)
    initial_graph = encode_graph_observation(encoder, result.observation)

    observation = result.observation
    for _ in range(5):
        action = blue_agent.get_action(observation, cyborg.get_action_space("Blue"))
        step_result = cyborg.step("Blue", action)
        observation = step_result.observation
        step_graph = encode_graph_observation(encoder, step_result.observation)

        assert step_graph.hostnames == initial_graph.hostnames
        assert step_graph.node_features.shape == initial_graph.node_features.shape
        assert step_graph.adjacency.shape == initial_graph.adjacency.shape
        np.testing.assert_array_equal(step_graph.adjacency, initial_graph.adjacency)


def test_partial_observation_masks_dynamic_features_only():
    cyborg = make_cc2_env()
    result = cyborg.reset("Blue")
    encoder = build_graph_encoder(result.observation)
    graph_obs = encode_graph_observation(encoder, result.observation)

    masked = mask_dynamic_features(graph_obs)
    dynamic_indices = list(graph_obs.dynamic_feature_indices)
    static_indices = [
        index for index in range(len(graph_obs.feature_names)) if index not in dynamic_indices
    ]

    assert dynamic_indices
    assert np.all(masked.node_features[:, dynamic_indices] == -1.0)
    np.testing.assert_array_equal(
        masked.node_features[:, static_indices],
        graph_obs.node_features[:, static_indices],
    )


def test_raw_encoder_derives_activity_and_compromise_from_connections():
    initial_observation = {
        "success": True,
        "HostA": {
            "Interface": [
                {
                    "Interface Name": "eth0",
                    "IP Address": "10.0.0.2",
                    "Subnet": "10.0.0.0/28",
                }
            ],
            "Sessions": [],
            "Processes": [],
            "User Info": [],
            "System info": {
                "Hostname": "HostA",
                "OSType": "LINUX",
                "OSDistribution": "UBUNTU",
                "OSVersion": "U18_04_3",
                "Architecture": "x64",
            },
        },
    }
    encoder = build_graph_encoder(initial_observation)

    exploit_observation = {
        "success": True,
        "HostA": {
            "Processes": [
                {
                    "Connections": [
                        {
                            "local_port": 22,
                            "remote_port": 4444,
                            "local_address": "10.0.0.2",
                            "remote_address": "10.0.0.9",
                        }
                    ]
                }
            ]
        },
    }
    graph_obs = encode_graph_observation(encoder, exploit_observation)
    feature_names = list(graph_obs.feature_names)

    assert graph_obs.node_features[0, feature_names.index("activity.is_exploit")] == 1.0
    assert graph_obs.node_features[0, feature_names.index("compromised.is_user")] == 1.0

    encoder.apply_blue_action(action_name="Remove", hostname="HostA")
    graph_obs = encoder.encode_current()

    assert graph_obs.node_features[0, feature_names.index("compromised.is_unknown")] == 1.0
    assert graph_obs.node_features[0, feature_names.index("compromised.is_user")] == 0.0


def test_feature_group_mask_can_mask_processes_without_masking_system_fields():
    cyborg = make_cc2_env()
    result = cyborg.reset("Blue")
    encoder = build_graph_encoder(result.observation)
    graph_obs = encode_graph_observation(encoder, result.observation)

    masked = mask_feature_groups(graph_obs, groups=("processes",))
    process_indices = [
        index for index, name in enumerate(graph_obs.feature_names) if name.startswith("processes.")
    ]
    system_indices = [
        index for index, name in enumerate(graph_obs.feature_names) if name.startswith("system.")
    ]

    assert process_indices
    assert system_indices
    assert np.all(masked.node_features[:, process_indices] == -1.0)
    np.testing.assert_array_equal(
        masked.node_features[:, system_indices],
        graph_obs.node_features[:, system_indices],
    )
