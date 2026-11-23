# Introduction to Robot Learning and VLA

The purpose of this note is not to make you memorize the terms encountered in recent days one by one, but to build a mental map of the field.

First we will separate the concepts into the right layers. Then we will place names such as OpenVLA, OpenVLA-OFT, LIBERO, MuJoCo, ROS, LeRobot and Microduck inside this structure.

---

# 1. The biggest picture

Think of the robot learning problem in its simplest form like this:

```text
OBSERVATION
    |
    v
  POLICY
    |
    v
 ACTION
    |
    v
WORLD CHANGES
    |
    v
NEW OBSERVATION
```

A robot observes the world, makes a decision, produces an action, this action changes the world, and the robot receives a new observation.

This loop continues constantly.

Compared with an image classification problem:

```text
Image
  |
  v
 CNN
  |
  v
Class
```

In robot learning it is:

```text
Image + Robot State + Language
             |
             v
           Policy
             |
             v
           Action
```

The key concept here is the word **policy**.

A policy is the function that decides what the robot will do by looking at its current observations.

---

# 2. What is Robot Learning?

Robot learning is the field that tries to learn robot behavior from data instead of defining it with hand-written rules.

For example, a classical system might look like this:

```text
if object_is_red:
    move_arm_left()
if gripper_distance < threshold:
    close_gripper()
```

In the robot learning approach, the system tries to learn the following relation from a large number of examples:

```text
observation -> action
```

The observation may contain:

- RGB camera image
- depth image
- robot joint angles
- gripper state
- force / torque information
- language command
- previous actions

The action may be, for example:

- move the arm 2 cm to the left
- rotate the wrist
- close the gripper
- drive the robot base forward

---

# 3. What is Embodied AI?

Embodied AI is a broader concept.

It studies an AI system solving tasks by perceiving and acting in a physical or simulated world.

A simple hierarchy:

```text
Artificial Intelligence
|
+-- Embodied AI
    |
    +-- Robot Learning
        |
        +-- Robot Policies
            |
            +-- Vision-Language-Action models
```

VLA is not the whole of this field.

VLA is one of the modern approaches used to produce robot policies.

---

# 4. The basic parts of a robot learning system

The easiest way to understand the field is to separate the following components from one another.

## 4.1 World

The environment the robot is in.

This environment can be:

- the real world
- a simulation

Example:

```text
table
mug
drawer
robot arm
camera
```

---

## 4.2 Task

What does the robot have to do?

For example:

```text
"Pick up the red cup."
"Open the drawer."
"Put the apple into the bowl."
```

A task can sometimes be an explicit language command.

Sometimes it may be defined implicitly during training.

---

## 4.3 Observation

Which part of the world does the robot see?

Example:

```text
RGB image
depth image
joint positions
gripper state
language instruction
previous actions
```

The robot does not always know the state of the whole world.

It only has the observation coming from its sensors.

---

## 4.4 Policy

The policy is the decision-making part of the system.

```text
observation
    |
    v
 policy
    |
    v
 action
```

The main model on the machine learning side is here.

A VLA is also a kind of policy.

---

## 4.5 Action

The physical command the robot will apply.

Example:

```text
delta_x = +0.02
delta_y = -0.01
delta_z = 0.00
gripper = close
```

Some systems directly produce low-level motor commands.

Others produce higher-level actions.

---

# 5. How does a robot learn?

Learning requires data.

Data can be produced in different ways.

## 5.1 Human Demonstration

A human shows the robot the task.

For example, with teleoperation:

```text
camera image
robot state
human action
```

are recorded.

A trajectory results.

---

## 5.2 Teleoperation

A human controls the robot with a joystick, a VR controller or another interface.

At every time step this pair is recorded:

```text
observation_t -> action_t
```

The training dataset is built from these.

---

## 5.3 Simulation Data

Robot tasks can be run in simulation.

Advantages:

- the real robot does not break
- data can be produced faster
- the environment is easy to change
- the same task can be repeated thousands of times

Disadvantage:

- simulation is not exactly the same as the real world

This difference is called the

**sim-to-real gap**.

---

# 6. Imitation Learning

One of the most basic robot learning paradigms.

Goal:

> Learn whatever the human or expert did.

Dataset:

```text
observation_1 -> action_1
observation_2 -> action_2
observation_3 -> action_3
...
```

From these the model learns the mapping:

```text
observation -> action
```

---

# 7. Behavior Cloning

One of the most basic forms of Imitation Learning.

It can be thought of as supervised learning.

```text
X = observation
y = expert action
```

The model learns:

```text
observation -> predicted action
```

This is one of the most important concepts for an introduction to robot learning.

---

# 8. Reinforcement Learning

In Reinforcement Learning the robot does not have to learn directly from correct action labels.

The robot learns through the loop:

```text
state
  |
action
  |
reward
```

Example:

```text
cup successfully grasped -> +1
cup dropped -> -1
```

But RL can be expensive on real robots because many trials are needed.

For this reason imitation learning is very important in the field of robot manipulation.

---

# 9. What is Manipulation?

Most VLA work is built on robot manipulation.

Manipulation:

The robot's physical interaction with objects.

Example tasks:

```text
pick
place
push
open
close
insert
pour
fold
stack
```

When you see a VLA paper, you will most likely encounter manipulation tasks.

---

# 10. What is a VLA?

VLA:

**Vision-Language-Action**

The basic idea:

```text
Vision
  +
Language
  +
Robot State
  |
  v
 VLA
  |
  v
Action
```

For example:

```text
camera image:
  red mug + blue mug

instruction:
  "pick up the blue mug"

robot state:
  arm position

        |
        v

       VLA

        |
        v

robot action
```

The critical idea here is:

A VLA should not only understand the image.

It must also connect the language command to robot behavior.

This is called

**language grounding**.

---

# 11. The difference between a VLM and a VLA

VLM:

Vision-Language Model

For example:

```text
image + question -> text answer
```

VLA:

Vision-Language-Action

```text
image + instruction -> robot action
```

So the output of a VLA is not only language.

It is robot behavior.

---

# 12. Examples of VLA models

VLA is not a model name, it is a model family.

Examples:

- RT-2
- OpenVLA
- Octo
- π0
- π0-Fast
- NORA
- UniVLA
- WorldVLA

They do not all have the same architecture.

But their common aim is to use vision and language information in producing robot actions.

---

# 13. What is OpenVLA?

OpenVLA is an open-source VLA model.

The simplest mental position:

```text
Robot Learning
    |
    +-- VLA
        |
        +-- OpenVLA
```

That is:

```text
VLA = category
OpenVLA = a specific model in this category
```

OpenVLA can be fine-tuned and evaluated on different robot manipulation tasks.

---

# 14. What is OpenVLA-OFT?

OpenVLA-OFT is not a separate VLA paradigm.

It is a version of OpenVLA improved with more effective fine-tuning and training methods.

Roughly:

```text
OpenVLA
   |
   | improved fine-tuning recipe
   v
OpenVLA-OFT
```

The OFT approach here aims for OpenVLA to perform better especially on downstream robot manipulation tasks.

So it is normal to come across the OpenVLA-OFT paper when working on OpenVLA with LIBERO.

Because the question is usually:

```text
How can OpenVLA be fine-tuned better on LIBERO?
```

OpenVLA-OFT is one of the important answers given to this question.

---

# 15. What is a Benchmark?

A benchmark is not a model.

A benchmark is an exam system.

A benchmark usually defines:

- environment
- tasks
- initial states
- demonstrations
- evaluation protocol
- success metric

A model is evaluated on a benchmark.

---

# 16. What is LIBERO?

LIBERO is a robot learning benchmark family.

It contains manipulation tasks in particular.

A simple mental model:

```text
OpenVLA = student
LIBERO  = exam
```

LIBERO contains different task suites.

They aim to measure different generalization properties of the robot.

For example, there are different task sets such as:

- object
- spatial
- goal
- long-horizon

---

# 17. What is LIBERO-Plus?

LIBERO-Plus is not a new VLA model.

It is an evaluation approach that makes the LIBERO benchmark harder in terms of robustness.

Goal:

> Did the model really learn the task, or is it using the easy shortcuts in the benchmark?

For example, the environment can be changed in these ways:

- language perturbation
- object appearance perturbation
- background change
- camera change
- distractor object
- different spatial configuration

In this way it is measured whether the model really generalizes.

---

# 18. OpenVLA-OFT and the language grounding problem

One of the interesting findings in the LIBERO-Plus paper is this:

In some tasks, the performance of OpenVLA-OFT can be largely preserved even when the language instruction is invalidated.

At first glance this looks very strange.

Because while the purpose of a VLA is:

```text
Vision + Language -> Action
```

the model may be behaving like this:

```text
Vision -> Action
```

and using the language part very little.

This can arise especially when every environment in the benchmark is very strongly paired with a particular task.

For example, if the model can already guess from the environment image which task will be done, the language instruction can become unnecessary.

This may be a kind of

**shortcut learning**

problem.

---

# 19. Do not confuse base OpenVLA with OpenVLA-OFT

The language insensitivity result found for OpenVLA-OFT cannot be directly generalized to base OpenVLA.

Because the fine-tuning method can change the model's behavior distribution significantly.

An important distinction:

```text
OpenVLA
   !=
OpenVLA-OFT
```

Even if they come from the same base model family, they can behave differently after downstream fine-tuning.

So when reading a paper, you need to pay attention to the exact checkpoint and training recipe of the model.

---

# 20. Why is generalization so important?

If a robot only memorizes the situations it saw during training, it is useless in the real world.

For example:

Training:

```text
red cup
wooden table
camera angle A
```

Test:

```text
green cup
white table
camera angle B
```

Can the model still do the task?

This is the generalization problem.

---

# 21. The basic research problems in robot learning

It is more useful to know the problems before the model names.

## 21.1 Manipulation

How does the robot physically interact with objects?

---

## 21.2 Generalization

Can it work in situations not seen during training?

---

## 21.3 Language Grounding

Does the robot really use the language instruction?

Example:

```text
red cup + blue cup

"pick up the red cup"
```

Is the robot really choosing according to the word "red"?

---

## 21.4 Robustness

Does the model break when the environment changes a little?

Example:

- lighting
- background
- camera position
- object texture
- distractor object
- instruction paraphrase

---

## 21.5 Long-Horizon Tasks

Not a single action, but a long chain of actions is needed.

Example:

```text
open fridge
   |
find apple
   |
pick apple
   |
close fridge
   |
put apple on table
```

Errors accumulate over time.

---

## 21.6 Reactive Behavior

Can the robot change its behavior when something unexpected happens in the environment?

For example:

```text
instruction:
"Take the apples to the kitchen."

robot carries bag

bag tears

apple falls
```

Can the robot notice this and change its plan?

This is a hard problem for current VLA systems.

---

## 21.7 Temporal Reasoning

The robot may need not only the current frame but also the past.

For example:

```text
the robot opened the drawer a moment ago
the camera does not fully see the drawer now
```

The robot must remember its past actions and observations.

---

## 21.8 Planning

In some tasks, producing actions directly is not enough.

The robot must first build a plan.

For example:

```text
Goal:
make coffee

Plan:
1. find cup
2. pick cup
3. place under machine
4. press button
5. wait
6. retrieve cup
```

---

# 22. What is Simulation?

Running the robot environment on a computer without using a real robot.

In simulation the following can be modeled:

- robot
- physics
- camera
- objects
- collision
- gravity
- friction

Some of the important simulators for this field:

- MuJoCo
- Isaac Sim
- PyBullet
- Gazebo

---

# 23. What is MuJoCo?

MuJoCo is a physics engine.

Its expansion:

**Multi-Joint dynamics with Contact**

It is very widely used for robot simulation.

MuJoCo simulates:

- rigid body dynamics
- joint movement
- contact
- friction
- collision
- gravity

Important:

```text
MuJoCo != VLA
MuJoCo != benchmark
MuJoCo != robot learning algorithm
```

MuJoCo is only the infrastructure that simulates the physical world.

---

# 24. A benchmark and a simulator are not the same thing

This distinction is very important.

Simulator:

```text
physics + world
```

Benchmark:

```text
tasks + evaluation protocol
```

For example:

```text
MuJoCo -> physics engine
LIBERO -> benchmark
```

A benchmark can run on a simulator such as MuJoCo.

But they are not the same thing.

---

# 25. What is ROS?

ROS:

**Robot Operating System**

Despite its name, it is not an operating system in the classical sense.

It is a middleware / framework structure that lets robotics software components communicate with each other.

For example:

```text
camera node
     |
     v
vision model
     |
     v
VLA policy
     |
     v
robot controller
     |
     v
motors
```

These components can exchange messages over ROS.

ROS is very important especially in real robot systems.

---

# 26. ROS2

Today ROS2 is the more common choice in new robotics projects.

ROS2 is stronger than ROS1 in:

- distributed communication
- real-time support
- multi-machine systems
- modern middleware

For a beginner it usually makes more sense to learn ROS2.

---

# 27. What is LeRobot?

LeRobot is an open-source robot learning ecosystem developed by Hugging Face.

Its aim is to make the robot learning workflow easier.

It includes:

- robot datasets
- data collection
- teleoperation
- policy training
- evaluation
- hardware integrations

It is more accurate to think of LeRobot like this:

```text
robot learning toolbox / ecosystem
```

It is not a VLA model.

It is not a simulator.

It is not a benchmark either.

---

# 28. What is Robomimic?

Robomimic is one of the open-source frameworks used for robot imitation learning.

It is used especially for policy training on demonstration datasets.

It is not the same as LeRobot but provides tools in a similar problem area.

---

# 29. What is Microduck?

Microduck is one of the real physical robot platforms.

Such robots can be used for small-scale experiments and education.

Mentally:

```text
Microduck = hardware
```

On it there can be components such as:

- camera
- motor
- robot arm
- gripper
- onboard computer

When you want to test a VLA model on a real robot, you need hardware platforms like this.

---

# 30. Robot Hardware examples

Robots frequently encountered in robot manipulation research:

- Franka Panda
- UR5 / UR10
- WidowX
- ALOHA
- xArm
- Kinova
- Fetch

These are not VLA models.

They are physical robot platforms.

---

# 31. What is ALOHA?

ALOHA is important as a robot system used especially in bimanual manipulation research.

It can be used for collecting human demonstrations with two robot arms and for imitation learning.

Mobile ALOHA combines a mobile platform with the dual-arm approach.

---

# 32. What is a Dataset?

A robot learning dataset usually does not have just an image-label structure.

A trajectory can look like this:

```text
episode_001

t=0
image
robot_state
action

t=1
image
robot_state
action

t=2
image
robot_state
action
```

An episode usually represents running a task from beginning to end.

---

# 33. What is a Trajectory?

Trajectory:

The record of the movements the robot made over time.

Roughly:

```text
(o1, a1)
(o2, a2)
(o3, a3)
...
(oT, aT)
```

Here:

```text
o = observation
a = action
```

It is encountered very often in robot learning papers.

---

# 34. What is an Episode?

An episode usually means a single task attempt.

For example:

```text
Task:
pick up red cup

Episode starts

robot moves
robot grasps cup
robot lifts cup

Episode ends
```

It can be successful or unsuccessful.

---

# 35. Success Rate

One of the most basic metrics in robot benchmarks.

For example:

```text
100 task attempt
87 successful

success rate = 87%
```

But a high success rate does not always mean real intelligence.

Works such as LIBERO-Plus question exactly this.

---

# 36. Shortcut Learning

Instead of learning the real cause of the task, the model can use easy correlations in the dataset.

For example:

```text
environment A -> always pick red cup
environment B -> always open drawer
```

The model can predict the task from the environment image without using the language instruction.

Result:

```text
benchmark score = high
true language grounding = weak
```

This is a very important problem for VLA evaluation.

---

# 37. Distribution Shift

If the training and test distributions are not the same, distribution shift occurs.

For example:

```text
training:
bright room

test:
dark room
```

or:

```text
training:
"pick up the red mug"

test:
"could you grab the crimson cup?"
```

Can the model handle this?

It is one of the basic questions of robustness work.

---

# 38. Domain Randomization

In simulation, environment properties are randomized.

For example:

- lighting
- texture
- object position
- camera angle
- friction

Goal:

To prevent the model from over-adapting to a single environment configuration.

---

# 39. Sim-to-Real

The model is trained in simulation.

Then it is transferred to the real robot.

Problem:

```text
simulation != reality
```

Because of this difference the model can perform badly in the real world.

This problem is called

**sim-to-real transfer**.

---

# 40. Real-to-Sim

There is also the reverse approach.

A digital copy of the real-world robot or environment is created.

This is especially important in the digital twin approach.

---

# 41. World Model

A world model is a model that tries to predict how the environment will change.

For example:

```text
current observation
+
candidate action
    |
    v
predicted next observation
```

A robot can use a world model for planning.

A VLA and a world model are not the same thing.

But in the future these two approaches may increasingly merge.

---

# 42. Diffusion Policy

Diffusion Policy is one of the important policy families for robot manipulation.

It is not a VLA.

It generally uses a diffusion model approach to produce continuous action sequences.

A simple comparison:

```text
VLA
vision + language -> action

Diffusion Policy
observation -> continuous action trajectory
```

Some modern systems can also combine language with the diffusion policy approach.

---

# 43. Transformer Robot Policy

Many modern robot policies use the Transformer architecture.

Because robot tasks involve time series, multimodal input and long context.

For example:

```text
image tokens
language tokens
robot state tokens
previous action tokens
```

can be processed in the same model.

---

# 44. Action Chunking

Instead of producing a single action at every timestep, the model can produce several actions at once.

For example:

```text
input:
current observation

output:
a_t
a_t+1
a_t+2
...
a_t+9
```

This approach can provide more stable behavior in robot control.

This idea is important in models such as ACT.

---

# 45. Closed-Loop and Open-Loop

## Open-loop

The model produces an action sequence and the robot applies it without receiving a new observation.

```text
observe once
   |
predict 10 actions
   |
execute all
```

## Closed-loop

The robot continuously receives new observations.

```text
observe
act
observe
act
observe
act
```

For the real world, closed-loop control is generally safer.

---

# 46. The big distinction in Robot Learning

When you see a paper, first ask this:

```text
What does this work change?
```

The options are usually:

```text
MODEL
DATA
TRAINING
ENVIRONMENT
BENCHMARK
HARDWARE
CONTROL
EVALUATION
```

Asking this question reduces terminology confusion significantly.

---

# 47. Placing terms in the right category

## Model / Policy

```text
OpenVLA
π0
RT-2
Octo
Diffusion Policy
ACT
```

## Training / Fine-tuning Method

```text
OpenVLA-OFT
behavior cloning
imitation learning
RL
```

## Benchmark / Task Suite

```text
LIBERO
LIBERO-Plus
CALVIN
RLBench
```

## Simulator / Physics

```text
MuJoCo
Isaac Sim
PyBullet
Gazebo
```

## Robot Learning Framework / Tooling

```text
LeRobot
robomimic
```

## Robot Middleware

```text
ROS
ROS2
```

## Hardware

```text
Franka
UR5
ALOHA
Microduck
```

---

# 48. How should a new paper be read?

When reading a VLA paper for the first time, it may be better not to start with the architecture diagram.

First answer these questions.

## 1. Task

What is the robot doing?

```text
pick?
place?
open?
long-horizon manipulation?
navigation?
```

## 2. Environment

```text
simulation?
real robot?
both?
```

## 3. Data

What does the model learn from?

```text
human demonstrations?
robot trajectories?
internet data?
simulation data?
```

## 4. Input

What does the model see?

```text
RGB?
depth?
language?
robot state?
history?
```

## 5. Output

What does the model produce?

```text
joint action?
end-effector pose?
action chunk?
high-level plan?
```

## 6. Model

Which policy is used?

```text
VLA?
diffusion?
transformer?
world model?
```

## 7. Training

How was it optimized?

```text
behavior cloning?
fine-tuning?
RL?
LoRA?
full fine-tuning?
```

## 8. Benchmark

Where was it tested?

```text
LIBERO?
CALVIN?
real robot?
```

## 9. Generalization Claim

What does the author claim generalizes?

```text
new objects?
new tasks?
new language?
new environments?
```

## 10. Ablation

What happens when the author removes which input?

This is very important especially for VLAs.

---

# 49. What to pay special attention to in VLA papers

The success metric alone is not enough.

For example:

```text
Model A
LIBERO success = 98%
```

This does not tell us:

```text
is language really being used?
```

So a strong evaluation should test:

```text
wrong language
blank language
paraphrased language
different object layout
different camera
different texture
distractor objects
new environment
```

---

# 50. Why did OpenVLA-OFT come up?

When you do research on the OpenVLA + LIBERO line, the literature naturally leads you to OpenVLA-OFT.

Because:

```text
OpenVLA
   |
fine-tune on robot task
   |
LIBERO evaluation
```

on this subject OpenVLA-OFT is an important reference.

So it is not wrong that the OFT paper came up.

But the critical point is this:

```text
OpenVLA-OFT result
```

does not automatically mean:

```text
base OpenVLA behavior
```

After fine-tuning the model may give different weight to language, vision or environment shortcuts.

---

# 51. Why is the LIBERO-Plus finding interesting?

Normal expectation:

```text
Vision + Language -> Action
```

If performance stays the same when the language input is removed, this question arises:

```text
Does the model really use language?
```

A possible explanation:

```text
Environment image
        |
        v
task identity already obvious
        |
        v
language becomes redundant
```

This is an important problem in VLA benchmark design.

---

# 52. The right learning order for this field

A starting order could be as follows.

## Stage 1

Learn these:

```text
observation
action
policy
trajectory
episode
behavior cloning
imitation learning
```

## Stage 2

Robot manipulation:

```text
gripper
end-effector
joint
pose
control
```

## Stage 3

Simulation:

```text
MuJoCo
environment
task
reset
step
reward
```

## Stage 4

Modern robot policies:

```text
ACT
Diffusion Policy
Octo
OpenVLA
π0
```

## Stage 5

Benchmarks:

```text
LIBERO
CALVIN
RLBench
```

## Stage 6

Robustness and generalization:

```text
language grounding
visual perturbation
distribution shift
shortcut learning
```

---

# 53. A practical starting strategy

You can start without a real robot.

The first goal:

```text
existing benchmark
+
existing policy
+
existing checkpoint
```

is to run them.

For example:

```text
LIBERO
+
existing OpenVLA / policy implementation
```

The first aim is not to develop a new model.

It is to understand this:

```text
how does the dataset come in?
how is the environment reset?
how does the observation go to the model?
how is the action applied to the environment?
how is success computed?
```

Once this pipeline runs once, the field becomes much more understandable.

---

# 54. The full flow of a VLA experiment

```text
TASK
"pick up the red cup"

        |
        v

ENVIRONMENT
robot + table + objects

        |
        v

OBSERVATION
RGB + robot state + language

        |
        v

POLICY
OpenVLA / π0 / another model

        |
        v

ACTION

        |
        v

ENVIRONMENT STEP

        |
        v

NEW OBSERVATION

        |
        v

repeat

        |
        v

SUCCESS / FAILURE
```

---

# 55. The connection between simulation and the real robot

```text
                    POLICY
                       |
           +-----------+-----------+
           |                       |
           v                       v
      SIMULATION               REAL ROBOT
       MuJoCo                   Franka
      Isaac Sim                  UR5
      PyBullet                  Microduck
```

The same policy can in theory run on both simulation and a real robot.

But the action space, sensors and robot morphology must be compatible.

---

# 56. Where does ROS come in?

A typical structure on a real robot:

```text
CAMERA
  |
  v
ROS IMAGE TOPIC
  |
  v
VLA MODEL
  |
  v
ACTION
  |
  v
ROS CONTROLLER
  |
  v
ROBOT
```

ROS here is not a model.

It connects the components to each other.

---

# 57. Where does LeRobot come in?

Typical flow:

```text
REAL ROBOT
   |
teleoperation
   |
   v
LeRobot Dataset
   |
   v
Policy Training
   |
   v
Evaluation
   |
   v
Deployment
```

LeRobot can be thought of as a tool ecosystem that makes the whole workflow easier.

---

# 58. The most critical conceptual distinction

It is enough to keep the following table in mind.

| Question | Concept |
|---|---|
| What will the robot do? | Task |
| What does the robot see? | Observation |
| Who makes the decision? | Policy |
| What does the robot apply? | Action |
| How does the model learn? | Training method |
| Where does it run? | Environment |
| Where is the physics simulated? | Simulator |
| How is the model tested? | Benchmark |
| How do the parts of a real robot talk? | ROS |
| What makes the robot learning workflow easier? | LeRobot |
| What is the physical machine? | Hardware |

---

# 59. The field in one sentence

Robot learning:

> Studies learning from data the policies that produce robot actions from observations.

VLA:

> The modern policy family that produces robot actions using vision and language information.

Simulator:

> Imitates the robot's physical world on a computer.

Benchmark:

> Measures how well a model works on particular tasks.

ROS:

> Connects the software components in a real robot system.

LeRobot:

> Provides data collection, training and deployment tools for robot learning.

---

# 60. When you see a new term from now on

The first question:

```text
What is this thing?
```

Put it into one of the following categories:

```text
MODEL
TRAINING METHOD
DATASET
BENCHMARK
SIMULATOR
FRAMEWORK
MIDDLEWARE
HARDWARE
CONTROL METHOD
EVALUATION METHOD
```

After making this classification, the meaning of the term usually settles in much more easily.

---

# Final mental map

```text
EMBODIED AI
|
+-- ROBOT LEARNING
    |
    +-- DATA
    |   +-- demonstrations
    |   +-- trajectories
    |
    +-- LEARNING
    |   +-- behavior cloning
    |   +-- imitation learning
    |   +-- reinforcement learning
    |
    +-- POLICY
    |   |
    |   +-- ACT
    |   +-- Diffusion Policy
    |   +-- VLA
    |       |
    |       +-- OpenVLA
    |       +-- π0
    |       +-- Octo
    |       +-- RT-2
    |
    +-- ENVIRONMENT
    |   |
    |   +-- Simulation
    |   |   +-- MuJoCo
    |   |   +-- Isaac Sim
    |   |   +-- PyBullet
    |   |
    |   +-- Real World
    |       +-- Franka
    |       +-- UR5
    |       +-- ALOHA
    |       +-- Microduck
    |
    +-- BENCHMARK
    |   |
    |   +-- LIBERO
    |   +-- LIBERO-Plus
    |   +-- CALVIN
    |   +-- RLBench
    |
    +-- TOOLING
        |
        +-- LeRobot
        +-- robomimic
        +-- ROS / ROS2
```

This map can be used to read the field independently of model names.

When a new paper comes, first find which box on this map the paper changes.

Then moving on to the technical details becomes much easier.
