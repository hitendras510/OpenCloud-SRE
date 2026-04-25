"""
env — OpenCloudEnv simulation, state tensor, and FastAPI server.

IMPORTANT: This __init__ imports NOTHING by default to prevent torch
import errors in non-training contexts (graph, UI, evaluation tests).
Import sub-modules explicitly where needed:

    from env.environment   import OpenCloudEnv, VALID_ACTIONS   # needs torch
    from env.state_tensor  import CloudStateTensor               # needs torch
    from env.fault_injection import ChaosMonkey, FaultCategory  # needs torch
    from env.observability import ObservabilityBus               # no torch
    from env.action_space  import ActionRegistry                 # needs torch
"""
# Intentionally empty — see docstring above.
