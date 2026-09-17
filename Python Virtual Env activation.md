2333 neurons all need to get processed in order for one epoch to happen.  

So first loop or epoch, we have a randomization of the 2333 neurons (shuffling) 

then those 2333 neurons that are now randomized are now passed through an iterative loop 

This nested loop basically says from 0 all the way to the 50th batch of neurons. 

The first random batch is put into idx it's start:start +batch_size due to the 50 I guess? 

idx is I assume is the iterating array of 50 row numbers. 

x_all[idx] and y_all[idx] are then the array from the shuffled idx so x_all and y_all with their idx will return an array of each log x and y value of the random idx row number which will be stored in x and y. 

from there we measure error. 