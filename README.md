# Enabler--A summer project by Ben Kurzion

## Problem Statement
A lot of people are colorblind and it makes crafting an outfit a pain in the butt. Sometimes, you get lucky and choose something cohesive, but more often than not, your pants clash with your shirt and you have no idea. 
Rather than have to contend with random chance, Enabler helps colorblind people identify colors and calculates whether two items clash with one another!

## Technical Information
Enabler takes a user inputted image and runs Graph Cut (Ford-Fulkerson min cut/max flow) to separate the item of inquiry from the background. Once the item has been isolated from the background, Enabler calculates the
item's color (via averaging RGB) and can computes whether other colors clash with this color. 
