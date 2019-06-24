# Character Generator for Dungeons and Dragons Characters

<strong>The project is currently in progress</strong>

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
5th edition (current) of Dungeons and Dragons. Then I will use the Tensorflow and
Keras libraries to build RNNs, such as LSTMs and GRUs to auto-generate text that
should closely mimic D&D monsters. The end product will be a model that can create 
novel monsters for Dungeons and Dragons adventures.

The initial approach will be to generate new monsters with character level models.
However, I will also plan to test sentence level models that leverage the power
of transformers.

## External Links

<a href='https://www.tensorflow.org/beta/tutorials/text/text_generation'>Tensorflow Text Generation with an RNN</a>: a 
helpful walkthrough on the tensorflow site to accomplish character level text generation.
