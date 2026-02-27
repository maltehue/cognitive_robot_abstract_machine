# Cognitive Robot Abstract Machine (CRAM)

Monorepo for the CRAM cognitive architecture.

A hybrid cognitive architecture enabling robots to accomplish everyday manipulation tasks.

This documentation serves as a central hub for all sub-packages within the CRAM ecosystem.

## About CRAM

The Cognitive Robot Abstract Machine (CRAM) ecosystem is a comprehensive cognitive architecture for autonomous robots, organized as a monorepo of interconnected components.
Together, they form a pipeline from abstract task descriptions to physically executable actions, bridging the gap between high-level intentions and low-level robot control.

[Installation](#Installation) | [Contributing](#Contributing) | [Github](https://github.com/cram2/cognitive_robot_abstract_machine)

### Architecture Overview

CRAM consists of the following sub-packages:

* **[PyCRAM](https://cram2.github.io/cognitive_robot_abstract_machine/pycram)**:
PyCRAM is the central control unit of the CRAM architecture.
It interprets and executes high-level action plans using the CRAM plan language (CPL).

* **[Semantic Digital Twin](https://cram2.github.io/cognitive_robot_abstract_machine/semantic_digital_twin)**
The semantic digital twin is a world representation that integrates sensor data, robot models, and external knowledge to provide a comprehensive understanding of the robot's environment and tasks.

* **[Giskardpy](https://github.com/SemRoCo/giskardpy)**
GiskardPy is a Python library for motion planning and control for robots.
It uses constraint- and optimization-based task-space control to control the whole body of a robot.

* **[KRROOD]((https://cram2.github.io/cognitive_robot_abstract_machine/krrood))**
KRROOD is a Python framework that integrates symbolic knowledge representation, powerful querying, and rule-based reasoning through intuitive, object-oriented abstractions.

* **[Probabilistic Model]((https://cram2.github.io/cognitive_robot_abstract_machine/probabilistic_model))**
Probabilistic Model is a Python library that offers a clean and unified API for probabilistic models, similar to scikit-learn for classical machine learning.

* **[Random Events](https://cram2.github.io/cognitive_robot_abstract_machine/random_events)**
Random Events is a Python library for modeling and simulating random events.

<img title="Architecture Diagram" alt="Architecture Diagram of the semantic digital twin monorepo" src="img/architecture_diagram.png">

## Installation

To install the CRAM architecture, follow these steps:

Set up the Python virtual environment:

```bash
sudo apt install -y virtualenv virtualenvwrapper && \
grep -qxF 'export WORKON_HOME=$HOME/.virtualenvs' ~/.bashrc || echo 'export WORKON_HOME=$HOME/.virtualenvs' >> ~/.bashrc && \
grep -qxF 'export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3' ~/.bashrc || echo 'export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3' >> ~/.bashrc && \
grep -qxF 'source /usr/share/virtualenvwrapper/virtualenvwrapper.sh' ~/.bashrc || echo 'source /usr/share/virtualenvwrapper/virtualenvwrapper.sh' >> ~/.bashrc && \
source ~/.bashrc && \
mkvirtualenv cram-env
```
Activate / deactivate

```
workon cram-env
deactivate
```

Pull the submodules: 
```bash
cd cognitive_robot_abstract_machine
git submodule update --init --recursive
```

### Install using UV 

To install the whole repo we use uv (https://github.com/astral-sh/uv), first to install uv:

```bash 
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh
```

then install packages:

```bash
uv sync --active
```


### Alternative: Poetry

Alternatively you can use poetry to install all packages in the repository.

Install poetry if you haven't already:

```bash
pip install poetry
``` 

Install the CRAM package along with its dependencies:

```bash 
poetry install
```


## Contributing

Before committing any changes, please navigate into the project root and install pre-commit hooks:

```bash
sudo apt install pre-commit
pre-commit install
```

If you have any questions or feedback, consider submitting a [GitHub Issue](https://github.com/cram2/cognitive_robot_abstract_machine/issues).

## Research & Publications

[1] A. Bassiouny and T. Schierenbeck, Knowledge Representation & Reasoning Through Object-Oriented Design. (Feb. 27, 2026). Python. CRAM (Cognitive Robot Abstract Machine). Accessed: Feb. 27, 2026. [Online]. Available: https://github.com/cram2/cognitive_robot_abstract_machine \
[2] M. Beetz, G. Kazhoyan, and D. Vernon, “The CRAM Cognitive Architecture for Robot Manipulation in Everyday Activities,” p. 20, 2021. \
[3] M. Beetz, G. Kazhoyan, and D. Vernon, “Robot manipulation in everyday activities with the CRAM 2.0 cognitive architecture and generalized action plans,” Cognitive Systems Research, vol. 92, p. 101375, Sep. 2025, doi: 10.1016/j.cogsys.2025.101375. \
[4] J. Dech, A. Bassiouny, T. Schierenbeck, V. Hassouna, L. Krohm, and D. Prüsser, PyCRAM: A Python framework for cognition-enbabled robtics. (2025). [Online]. Available: https://github.com/cram2/pycram \
[5] T. Schierenbeck, probabilistic_model: A Python package for probabilistic models. (Jul. 01, ). [Online]. Available: https://github.com/tomsch420/probabilistic_model \
[6] T. Schierenbeck, Random-Events. (Apr. 01, 2002). [Online]. Available: https://github.com/tomsch420/random-events \
[7] S. Stelter, “A Robot-Agnostic Kinematic Control Framework: Task Composition via Motion Statecharts and Linear Model Predictive Control,” Universität Bremen, 2025. doi: 10.26092/ELIB/3743.

## About the AICOR Institute for Artificial Intelligence

The AICOR Institute for Artificial Intelligence researches how robots can understand and perform everyday tasks using fundamental cognitive abilities – essentially teaching robots to think and act in practical, real-world situations.

The institute is headed by Prof. Michael Beetz, and is based at the [University of Bremen](https://www.uni-bremen.de/en/), where is is affiliated with the [Center for Computing and Communication Technologies (TZI)](https://www.uni-bremen.de/tzi/) and the high-profile area [Minds, Media and Machines (MMM)](https://minds-media-machines.de/en/).

Beyond Bremen, AICOR is also part of several research networks:

* [Robotics Institute Germany (RIG)](https://robotics-institute-germany.de/) – a national robotics research initiative
* [euROBIN](https://www.eurobin-project.eu/) – a European network focused on advancing robot learning and intelligence

[Website](https://ai.uni-bremen.de/) | [Github](https://github.com/code-iai)

## Acknowledgements

This work has been partially supported by the German Research Foundation DFG, as part of Collaborative Research Center (Sonderforschungsbereich) 1320 Project-ID 329551904 "EASE - Everyday Activity Science and Engineering", University of Bremen (http://www.ease-crc.org/).
