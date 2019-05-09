# Character Generator for Dungeons and Dragons Characters

My movtivation from this project comes from my experience playing in roleplaying
groups. Dungeons and Dragons (D&D) is a roleplaying game where a small group of players
will each assume the role of a fantasy character and act out adventures and combat
with terrible monsters via the medium of spoken word and dice rolls. The monsters
have text descriptions and statistics that allow the player characters to interact
with them. Anyone who has played in a D&D group before will know
that there is advantage to be gained by understanding the strategies the monsters
are designed to employ. And so novelty becomes essential to the D&D experience
(especially for a seasoned group of nerds). Hence, I designed this project to 
allow me to take some state of the art natural language processing technologies 
for a spin to help enchance the novelty and gaming experience for my roleplaying
groups. 

I would be remiss in not saying that the inspiration for this project came from 
both my interest in NLP and deep learning, as well as
<a href='https://aiweirdness.com/post/165373096197/a-neural-network-learns-to-create-better-dd'>
an aiweirdness.com post</a> on using neural networks to generate the names for 
novel D&D spells.

## What Am I Actually Doing Here?

I will use Python's Scrapy library to acquire all the monster descriptions for the
latest edition of Dungeons and Dragons. Then I will use the Keras library's LSTM
paired with ULMFiT to train a network on a Spark EMR cluster. The end product will
be a model that can create novel monsters for Dungeons and Dragons adventures.

## External Links

Coming Soon!
