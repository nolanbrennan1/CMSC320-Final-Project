# Texas Hold’em Analysis
### Fall 2024 Data Science Project
### Nolan Brennan
## Contributions:
This was a solo project, all portions of the project were created by Nolan Brennan.

## Background information:
Texas Hold’em is the most common variant of poker. The game is played by each player being dealt a 2-card hand. The goal is for each player to create the best 5-card hand out of their 2 cards and 5 shared community cards. Cards are dealt sequentially, with each player first being dealt their hand, followed by 3 community cards (called the flop), then a 4th community card (called the turn), and finally the last community card (called the river). Between each step, all remaining players in the round bet. All players still in the round after betting on the river then reveal their hand to determine the winner (called the showdown). A player can either raise – which means to add more money, call – which means to put in equal money as the previous raise, check – which means to call given no previous raises, or fold – which means to not call and forfeit the round.

## Introduction:
This project will be analyzing a database of 10,000 hands of No-Limit Texas Hold’em poker conducted in a research study. This analysis will attempt to investigate several observed gameplay patterns, attempting to note features of an optimal playstyle. In the data exploration phase of this project, in addition to an observation on finishing stack amounts (the amount of money at the end of the hand), two gameplay features will be investigated. These gameplay features are the proportion of winning compared to table position and which winning hand type wins the most money.  Knowledge of these gameplay concepts will contribute to a winning playstyle, earning a player more money in the long run. Additionally, a machine learning model will be created to determine which hands a player should play vs fold prior to any board information. This concept is called “ranges” in poker, which is a gameplay strategy to only play hands that have a higher than average likelihood of winning before investing in the round.

## Data Curation:
This study observed 5 professional poker players and an AI poker player the researchers developed. The study was conducted over a period of 12 days, with 14 total players swapping in and out. The game was 6-player No-Limit Texas Hold’em, with blinds set at 50/100 points. Individual player points were reset after every round, meaning there was no continuity between hands. This database was selected as it is the largest publicly accessible No-Limit Texas Hold’em database easily findable on the internet. Hand information is given in a standardized format seen below. This format lacks several important observations required for this analysis, though these observations can be extrapolated based on what is given (Ex. Winning player based on which player is remaining). 

## Citations:
Brown, Noam, and Tuomas Sandholm. “Superhuman AI for Multiplayer Poker.” Science, vol. 365, no. 6456, 11 July 2019, pp. 885–890, https://doi.org/10.1126/science.aay2400.

uoftcprg. “Phh-Dataset/Data/Pluribus at Main · Uoftcprg/Phh-Dataset.” GitHub, 2024, github.com/uoftcprg/phh-dataset/tree/main/data/pluribus. Accessed 3 Dec. 2024.



## Exploratory data analysis:
See graphs in .ipynb file.



## Primary Analysis:
A Random Forest Model was used to predict whether a player should play or not play a hand prior to any information being given. This is a classification problem, meaning that the model should either give a 0 or 1 to not play or play respectively. Random Forest was selected as it allows for the data to be trained and tested amongst the whole dataset, which despite being 10,000 hands, is still size constrained due to the total number of possible hands. Random forest is also a classification model, intended for similar problems.
	
## Results of Primary Analysis:
This model is able to predict not-playing hands with a high F1 score. However, the model is not able to predict playing hands very accurately, with only a 0.45 F1 score. This model may be improved upon with a larger dataset, as not playing hands are overrepresented in the dataset.

## Insights and Conclusions:
This project successfully informs an uninformed and informed audience. Primarily, insights into three specific gameplay features 

7. Insights and Conclusions. After reading through the project, does an uninformed
reader feel informed about the topic? Would a reader who already knew about the
topic feel like they learned more about it?

