# Ingenieurie Logicielle pour l'IA et la robotique (Software Engineering for AI and Robotics) Project

## Project Overview

In this project, we had to model and implement a neural network that could, from a live video feed, tell a simulated car wich action take to stay in her lane.

To do that, we had to create a dataset consisting of video frames and the corresponding actions.

We then trained a neural network on this dataset to predict the appropriate action for each frame, allowing the simulated car to stay in its lane based on the live video feed.

A detailled explanation (in French) of our method and results can be found in the [Final Report](Final_Report.pdf) file.

## Launching the Project

To launch the project, you need to have *ROS*, *RVIZ* and *GAZEBO* installed on your system.

You can then run the simulation by executing the autopilot_teleop.launch file.