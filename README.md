# E. coli case prediction with LSTMs
2019 sci fair project.

## Why?
I was fascinated with applications of deep learning towards predicting epidemics. Considering that E. coli was hot in the news at the time
and my town relies heavily on the affected crop, I thought it'd be a cool topic.

## How?
I scraped data off the CDC's NNDSS (National Notifiable Diseases Surveillance System) & took search popularity data from Google Trends. I then trained it under an LSTM to use the latter to predict the former.

## What's an LSTM?
A fancy type of neural network. Neural networks basically let you throw a bunch of data into it as an input, (algorithmically) tinker around with the calculations a bit, then slowly approach an accurate output. LSTMs let you throw your outputs back into the input recursively. For example, in language processing, this allows the network to take previous outputs into context by feeding it as an input.

## Why an LSTM?
I thought it'd be cool if I could recursively predict cases into the future, but yes, it probably would have been far better and easier with another algorithm

## Does it work?
Depends on your definition of "work". It had an MSE in the single digits, which sounds great, but a lot of the data was missing/just straight up had no infected people at some points. I would have estimated the data with a spline curve but a) too lazy and, more importantly, b) I finished this like a day before the fair. That being said, it did follow the trend pretty accurately, so it did "learn" a little bit.
