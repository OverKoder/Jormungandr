#Jormungandr
Hello! This is project I thought of, so i made it!

The AI uses an A* algorythm to find the path to the food, however, some improvements can be made, because, the AI calculates the path given the position of the food and snake's body position and does not try to predict the future.

When it is finally calculated, the snake does a sequence of moves to get there, but, if you run it, you may see that sometimes it does not take the optimal path, that is because the AI thinks that the obstacles (the snake's body) is still at the same place from when the path was being calculated, but in reality, the body moved with the head, and the path that was previously blocked it is now cleared. I am aware of this and i will fix it if I have time for it.
