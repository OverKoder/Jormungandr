#Jormungandr
Hello! This is project I thought of, so i made it!

The AI uses an A* algorithm to find the path to the food, when it is finally calculated, the snake does a sequence of moves to get there. 

Some modifications have been made. There is no cost to travel from node to node. And the heuristic value is calculated as the euclidean distance between the food and the head position. Also, the path is calculated considering the number of moves it takes to get a certain node. For this, each square of the snake's body is assigned a value which represents the number of moves the snake has to do in order for that specific spot to be no longer an obstacle. If the number of moves the snake has to do is higher than the obstacle's assigned value, it is accepted as a valid path.
