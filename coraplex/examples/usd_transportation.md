# USD scene transportation

This integration branch combines the small import, placement and transportation PRs. The public test in `test/coraplex_test/test_transporting.py` builds its scene through `test/dataset/usd_transportation.py`, imports it with `WorldSpecification.from_usd_scene`, and executes a core transportation plan. It uses PR2 as a two-finger compatibility case and Armar7 as a five-finger humanoid case.

The scene contains labelled floor and worktops, separate visual and collision shapes, a graspable object and a revolute cabinet door. The robot receives its own instance prefix. Explicit support bodies set its standing height after the starting joint configuration is applied. Selected hand–object and object–worktop contacts permit manipulation while collision avoidance remains enabled. A carrying pose precedes continuous path navigation; placement releases the object and retracts the hand before arm parking.

The test observes movement while the object is attached, checks that the attachment transform remains rigid, and verifies release at the requested pose, standing height and restoration of temporary collision rules. It also checks that the imported door retains its articulated connection. Public robot descriptions must be installed; unavailable descriptions or an older USD installation without `UsdSemantics` cause an explicit skip. The focused geometry, grasp and policy regressions run without proprietary assets or live services.

Run the integration check with the configured project interpreter using `pytest test/coraplex_test/test_transporting.py -k transportation_in_an_imported`. Follow the repository's normal ORM-generation setup; generated interfaces stay ignored.

USD coordinates retain their authored units and axes. Supply `world_T_scene` to place the stage frame in the world. Joints start at zero; apply configured starting joint states through the world API. Supported articulations form a tree with fixed, revolute or prismatic joints and rigid joint frames. This series supplies kinematic humanoid transport, not walking or balance control. Continuous simulation and whole-posture clearance are explicit options, and collision-free reachability still depends on the scene layout and robot configuration.

## Merge order

Dependent PRs target their prerequisite branches to keep each review focused. Merge the prerequisite into `main`, then retarget its dependent PR to `main`. Keep the branches until their dependent PRs are retargeted. The integration PR is the combined acceptance snapshot; merge its remaining test and documentation additions last after the component PRs.

| PR | Change | Prerequisite |
| --- | --- | --- |
| [7](https://github.com/maltehue/cognitive_robot_abstract_machine/pull/7) | Concurrent ROS action waits | — |
| [8](https://github.com/maltehue/cognitive_robot_abstract_machine/pull/8) | TF callback initialization | — |
| [10](https://github.com/maltehue/cognitive_robot_abstract_machine/pull/10) | Consistent robot instance prefixes | — |
| [11](https://github.com/maltehue/cognitive_robot_abstract_machine/pull/11) | USD placement and scale | — |
| [13](https://github.com/maltehue/cognitive_robot_abstract_machine/pull/13) | Separate static scene objects | 11 |
| [14](https://github.com/maltehue/cognitive_robot_abstract_machine/pull/14) | Authored collision geometry | 13 |
| [12](https://github.com/maltehue/cognitive_robot_abstract_machine/pull/12) | Articulated scene objects | 14 |
| [17](https://github.com/maltehue/cognitive_robot_abstract_machine/pull/17) | Support labels | 12 |
| [19](https://github.com/maltehue/cognitive_robot_abstract_machine/pull/19) | Scene world specifications | 17 |
| [28](https://github.com/maltehue/cognitive_robot_abstract_machine/pull/28) | Unspecified USD mass properties | 19 |
| [16](https://github.com/maltehue/cognitive_robot_abstract_machine/pull/16) | Body annotation lookup | — |
| [15](https://github.com/maltehue/cognitive_robot_abstract_machine/pull/15) | Ray-based grasp confidence | — |
| [18](https://github.com/maltehue/cognitive_robot_abstract_machine/pull/18) | Thumb-opposed multi-finger detection | 15 |
| [9](https://github.com/maltehue/cognitive_robot_abstract_machine/pull/9) | Standing root height during navigation | — |
| [20](https://github.com/maltehue/cognitive_robot_abstract_machine/pull/20) | Support-geometry placement | 9 |
| [26](https://github.com/maltehue/cognitive_robot_abstract_machine/pull/26) | Bounds of frame-only mobile bases | — |
| [21](https://github.com/maltehue/cognitive_robot_abstract_machine/pull/21) | Temporary collision policy restoration | — |
| [22](https://github.com/maltehue/cognitive_robot_abstract_machine/pull/22) | Selected manipulation contacts | 21 |
| [23](https://github.com/maltehue/cognitive_robot_abstract_machine/pull/23) | Carried-object navigation clearance | 9 |
| [24](https://github.com/maltehue/cognitive_robot_abstract_machine/pull/24) | Continuous simulated navigation | 23 |
| [27](https://github.com/maltehue/cognitive_robot_abstract_machine/pull/27) | Floors beneath offset base frames | 23 |
| [25](https://github.com/maltehue/cognitive_robot_abstract_machine/pull/25) | Carrying posture and deferred placement | 22 |

ROS waits concern independent client calls; they do not schedule competing goals on one server or coordinate robot motions. The transportation example runs one robot's plan sequentially. Cramera foundation work and UI features are outside this series.
