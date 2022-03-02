# pixelate-autonomous-
This is an autonomous robot system in webots simulation which uses open cv for path image and path planning.The sample path is given and is a nXn matrix with blocks of different colour denoting different costs of moving to the block
The blocks containing shapes have certain conditions such as circle denoting a patient.


![sample](https://github.com/aniketjohri23/pixelate-autonomous-/blob/main/sample.jpg)

The objective is to pick up patients from their respective pickup point and release them at their stated hospitals with the condition that the other patient should  not be in the path.The cost of the path should be minimum
For shortest path we used the Dijkstra's algorithm since it works with positve weights and for tracking the robot's position we have used the aruco marker which comes under the open cv library
